"""
Telegram Bot integration for Price Action Trading Bot
Implements show/hide keyboard and command handlers for account, positions,
orders, trading actions, performance, history, alerts, and news.
"""

import asyncio
import threading
import logging
from typing import Optional, List, Dict

from config import (
    TELEGRAM_BOT_TOKEN,
    SYMBOL,
    SYMBOLS,
    TIMEFRAME,
    RISK_PERCENTAGE,
    DEFAULT_SL_POINTS,
    DEFAULT_TP_POINTS,
    TE_API_CLIENT,
    TE_COUNTRY,
    TE_IMPORTANCE,
    NEWS_API_KEY,
    NEWS_COUNTRY,
    NEWS_CATEGORY,
)
from mt5_connector import MT5Connector
from database import db_manager
try:
    from ai.auto_trainer import AutoTrainer
except Exception:
    AutoTrainer = None

# python-telegram-bot v20
from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    BotCommand,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

logger = logging.getLogger(__name__)


def _build_main_reply_keyboard(news_count: int = 0, is_admin: bool = False) -> ReplyKeyboardMarkup:
    news_label = "📰 News" if news_count <= 0 else f"📰 News ({news_count})"
    keyboard_layout = [
        ["ℹ️ Info", "👤 Account"],
        ["📊 Positions", "📋 Orders"],
        ["🟢 Start Trade", "🔴 End Trade"],
        ["📈 Performance", "🧾 History"],
        ["⚠️ Close Reasons","🧠 Analyze Now"],
        ["🤖 AI Status"],
        ["🚀 AI Train", "📈 AI Performance"],
        [news_label],
    ]
    
    # Add admin-only buttons if user is admin
    if is_admin:
        keyboard_layout.extend([
            ["👑 Admin Panel"],
            ["➕ Add User", "📋 List Users"],
            ["📊 DB Stats"]
        ])
    
    return ReplyKeyboardMarkup(
        keyboard_layout,
        resize_keyboard=True,
        one_time_keyboard=False,
        is_persistent=True,
    )

def _build_minimal_reply_keyboard(is_admin: bool = False) -> ReplyKeyboardMarkup:
    keyboard_layout = [
        ["ℹ️ Info"],
        ["🔑 Login", "👤 Account"],
    ]
    
    # Add admin-only buttons if user is admin
    if is_admin:
        keyboard_layout.extend([
            ["👑 Admin Panel"],
            ["➕ Add User", "📋 List Users"],
            ["📊 DB Stats"]
        ])
    
    return ReplyKeyboardMarkup(
        keyboard_layout,
        resize_keyboard=True,
        one_time_keyboard=False,
        is_persistent=True,
    )


def _build_show_hide_inline() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(text="Show Keyboard", callback_data="show_keyboard"),
                InlineKeyboardButton(text="Hide Keyboard", callback_data="hide_keyboard"),
            ]
        ]
    )


class TelegramBot:
    """Telegram Bot wrapper using python-telegram-bot v20."""

    def __init__(self, mt5_connector, controller):
        self.mt5 = mt5_connector
        self.controller = controller  # PriceActionTradingBot instance for start/stop and stats
        self.application = None
        self.alerts_enabled = True
        self.current_risk_percentage = RISK_PERCENTAGE
        self.default_sl_points = DEFAULT_SL_POINTS
        self.default_tp_points = DEFAULT_TP_POINTS
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Per-chat MT5 sessions
        self._sessions: Dict[int, MT5Connector] = {}
        # Per-chat login state machine: chat_id -> {stage, login, password, server}
        self._login_states: Dict[int, Dict[str, str]] = {}
        # Auto-trainer instance
        self.auto_trainer: Optional[AutoTrainer] = None

    def _get_session(self, chat_id: int) -> Optional[MT5Connector]:
        """Return per-chat MT5Connector if exists, else None (force login)."""
        return self._sessions.get(chat_id)
    
    def _is_user_authorized(self, chat_id: int) -> bool:
        """Check if telegram chat ID is authorized to use the bot."""
        return db_manager.is_telegram_user_authorized(chat_id)
    
    def _is_user_admin(self, chat_id: int) -> bool:
        """Check if telegram chat ID is an admin."""
        return db_manager.is_user_admin(chat_id)
    
    async def _check_user_authorization(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Check user authorization and send error message if not authorized."""
        chat_id = update.effective_chat.id
        
        if not self._is_user_authorized(chat_id):
            await update.message.reply_text(
                "❌ Access Denied\n\n"
                "You are not authorized to use this bot. Please contact the administrator to get access."
            )
            return False
        return True
    
    async def _check_admin_authorization(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Check admin authorization and send error message if not admin."""
        chat_id = update.effective_chat.id
        
        if not self._is_user_admin(chat_id):
            await update.message.reply_text(
                "❌ Admin Access Required\n\n"
                "This command is only available to administrators."
            )
            return False
        return True

    async def _cmd_login(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/login <login> <password> <server>  OR interactive when no args."""
        try:
            # Check user authorization first
            if not await self._check_user_authorization(update, context):
                return
                
            args = context.args
            if len(args) < 3:
                # Start interactive wizard
                chat_id = update.effective_chat.id
                self._login_states[chat_id] = {"stage": "account"}
                await update.message.reply_text("Please enter your Account (login) number:")
                return
            login = int(args[0])
            password = args[1]
            server = args[2]
            
            # Check if user is already logged in with a different account
            existing_session = self._sessions.get(update.effective_chat.id)
            if existing_session and existing_session._login != login:
                await update.message.reply_text(
                    f"You are already logged in with account {existing_session._login}. "
                    f"Please logout first before switching to account {login}."
                )
                return
            
            # Create and connect new session
            session = MT5Connector(login=login, password=password, server=server,
                                   terminal_path=None)
            if not session.connect():
                # Provide specific MT5 error if available
                try:
                    msg = session.get_last_error_message()
                except Exception:
                    msg = "MT5 login failed. Check credentials/server."
                await update.message.reply_text(f"❌ {msg}")
                return
            
            # Store session and update database
            self._sessions[update.effective_chat.id] = session
            
            # Get bot user info and store MT account in database
            chat_id = update.effective_chat.id
            bot_user = db_manager.get_bot_user_by_telegram_chat_id(chat_id)
            if bot_user:
                db_manager.add_mt_account(bot_user['bot_user_id'], login)
                logger.info(f"Stored MT account {login} for bot_user_id {bot_user['bot_user_id']}")
            
            info = session.get_account_info() or {}
            await update.message.reply_text(
                f"✅ Logged in to account: {info.get('login', login)}\n"
                f"Balance: {info.get('balance', 0):.2f} {info.get('currency', '')}"
            )
            # Show full keyboard after login
            try:
                count = await self._get_upcoming_count()
                is_admin = self._is_user_admin(chat_id)
                await update.message.reply_text(
                    "Keyboard updated.", reply_markup=_build_main_reply_keyboard(count, is_admin)
                )
            except Exception:
                pass
        except Exception as e:
            await update.message.reply_text(f"Login error: {e}")

    async def _cmd_logout(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        session = self._sessions.get(chat_id)
        if not session:
            await update.message.reply_text("No session to logout.")
            return
        
        account_login = session._login
        try:
            # Only disconnect if this session is currently active
            if session.is_active_connection():
                session.disconnect()
            else:
                # Just remove from sessions if not active
                logger.info(f"Removing inactive session for account {account_login}")
        except Exception:
            pass
        
        # Remove from database
        bot_user = db_manager.get_bot_user_by_telegram_chat_id(chat_id)
        if bot_user:
            db_manager.remove_mt_account(bot_user['bot_user_id'])
            logger.info(f"Removed MT account from database for bot_user_id {bot_user['bot_user_id']}")
        
        self._sessions.pop(chat_id, None)
        await update.message.reply_text(f"✅ Logged out of MT5 account {account_login} for this chat.")

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Check user authorization first
        if not await self._check_user_authorization(update, context):
            return
            
        await update.message.reply_text(
            "Welcome to Price Action Bot. Use the buttons or commands.",
            reply_markup=_build_show_hide_inline(),
        )

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Stop trading and close all positions with summary"""
        # Check user authorization first
        if not await self._check_user_authorization(update, context):
            return
            
        chat_id = update.effective_chat.id
        session = self._get_session(chat_id)
        
        if not session:
            await update.message.reply_text("Please /login first to stop trading.")
            return
        
        # Send initial message
        await update.message.reply_text("🛑 Stopping trade and closing all positions...")
        
        try:
            # Disable only this chat's trading and close positions
            try:
                if hasattr(self.controller, 'stop_trading_for_chat'):
                    self.controller.stop_trading_for_chat(chat_id)
                else:
                    # Fallback: legacy behavior affects global trading
                    self.controller.set_mt5_connector(session)
                    self.controller.disable_trading()
            except Exception:
                pass

            # First cancel all pending orders
            try:
                cancel_res = session.cancel_all_orders()
                if cancel_res.get('total', 0) > 0:
                    await update.message.reply_text(
                        f"⛔ Cancelled pending orders: {cancel_res.get('success', 0)}/{cancel_res.get('total', 0)}"
                    )
            except Exception:
                pass

            # Close all positions for this chat/account
            try:
                if hasattr(self.controller, 'close_all_positions_for_chat'):
                    close_results = self.controller.close_all_positions_for_chat(chat_id)
                else:
                    close_results = session.close_all_positions()
            except Exception:
                close_results = self.controller.close_all_positions_for_chat(chat_id)
            
            # Build summary message
            summary_lines = ["📊 **Trade Stop Summary**\n"]
            
            if 'message' in close_results:
                summary_lines.append(f"ℹ️ {close_results['message']}")
            else:
                summary_lines.append(f"📈 **Positions Closed:** {close_results['success']}/{close_results['total']}")
                summary_lines.append(f"💰 **Total Profit/Loss:** {close_results['total_profit']:.2f}")
                
                if close_results['failed'] > 0:
                    summary_lines.append(f"❌ **Failed to Close:** {close_results['failed']}")
                
                # Add details for each position
                if close_results['details']:
                    summary_lines.append("\n📋 **Position Details:**")
                    for detail in close_results['details']:
                        status_icon = "✅" if detail['status'] == 'closed' else "❌"
                        profit_icon = "📈" if detail['profit'] >= 0 else "📉"
                        summary_lines.append(
                            f"{status_icon} #{detail['ticket']} {detail['type'].upper()} "
                            f"{detail['volume']} {detail['symbol']} {profit_icon} {detail['profit']:.2f}"
                        )
            
            # Add bot status
            summary_lines.append(f"\n🤖 **Bot Status:** Trading Disabled")
            
            # Send summary
            summary_text = "\n".join(summary_lines)
            await update.message.reply_text(summary_text, parse_mode='Markdown')
            
            # Keep bot running; only trading is disabled
            try:
                self.controller.unsubscribe_telemetry(chat_id)
            except Exception:
                pass

            # Stop AutoTrainer if running
            try:
                if self.auto_trainer:
                    self.auto_trainer.stop_auto_training()
            except Exception:
                pass
            
        except Exception as e:
            logger.exception("Error during stop trade")
            await update.message.reply_text(f"❌ Error stopping trade: {e}")
        
        # Show keyboard options
        await update.message.reply_text("Use ▶️ Start Trade to enable trading again.", reply_markup=_build_show_hide_inline())

    async def _cmd_close_all(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Close all positions without stopping the bot"""
        chat_id = update.effective_chat.id
        session = self._get_session(chat_id)
        
        if not session:
            await update.message.reply_text("Please /login first.")
            return
        
        # Send initial message
        await update.message.reply_text("🔄 Closing all positions...")
        
        try:
            # Close all positions and update statistics
            close_results = self.controller.close_all_positions_for_chat(chat_id)
            
            # Build summary message
            summary_lines = ["📊 **Close All Positions Summary**\n"]
            
            if 'message' in close_results:
                summary_lines.append(f"ℹ️ {close_results['message']}")
            else:
                summary_lines.append(f"📈 **Positions Closed:** {close_results['success']}/{close_results['total']}")
                summary_lines.append(f"💰 **Total Profit/Loss:** {close_results['total_profit']:.2f}")
                
                if close_results['failed'] > 0:
                    summary_lines.append(f"❌ **Failed to Close:** {close_results['failed']}")
                
                # Add details for each position
                if close_results['details']:
                    summary_lines.append("\n📋 **Position Details:**")
                    for detail in close_results['details']:
                        status_icon = "✅" if detail['status'] == 'closed' else "❌"
                        profit_icon = "📈" if detail['profit'] >= 0 else "📉"
                        summary_lines.append(
                            f"{status_icon} #{detail['ticket']} {detail['type'].upper()} "
                            f"{detail['volume']} {detail['symbol']} {profit_icon} {detail['profit']:.2f}"
                        )
            
            # Send summary
            summary_text = "\n".join(summary_lines)
            await update.message.reply_text(summary_text, parse_mode='Markdown')
            
        except Exception as e:
            logger.exception("Error during close all positions")
            await update.message.reply_text(f"❌ Error closing positions: {e}")

    async def _cmd_analyze_now(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Force an immediate analysis pass and send snapshot to Telegram."""
        try:
            # Ensure we use this chat's MT5 session for data access
            chat_id = update.effective_chat.id
            session = self._get_session(chat_id)
            if not session:
                await update.message.reply_text("Please /login first.")
                return
            # Reconnect if the session is not currently connected
            try:
                if not getattr(session, 'connected', False):
                    ok = session.connect()
                    if not ok:
                        await update.message.reply_text("❌ Could not connect to MT5. Please /login again.")
                        return
            except Exception:
                await update.message.reply_text("❌ MT5 connection error. Please /login again.")
                return
            # Attach the session to the controller temporarily
            try:
                self.controller.set_mt5_connector(session)
            except Exception:
                pass
            snap = self.controller.generate_analysis_snapshot()
            if snap:
                # Attach AI explainability if available
                try:
                    ai = getattr(self.controller, 'ai_strategy', None)
                    if ai and ai.last_prediction and ai.last_prediction.get('explain'):
                        ex = ai.last_prediction['explain']
                        top_pos = ex.get('top_positive') or []
                        top_neg = ex.get('top_negative') or []
                        lines = ["\nTop AI drivers:"]
                        if top_pos:
                            lines.append("+ " + ", ".join([f"{k} ({v:.3f})" for k, v in top_pos[:3]]))
                        if top_neg:
                            lines.append("- " + ", ".join([f"{k} ({v:.3f})" for k, v in top_neg[:3]]))
                        snap = f"{snap}\n" + "\n".join(lines)
                except Exception:
                    pass
                await update.message.reply_text(snap)
            else:
                await update.message.reply_text("No snapshot available (no data or error).")
        except Exception as e:
            logger.exception("Error triggering analysis")
            await update.message.reply_text(f"❌ Failed to analyze now: {e}")
    async def _cmd_start_trade(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enable auto trading, using this chat's MT5 session if available."""
        # Check user authorization first
        if not await self._check_user_authorization(update, context):
            return
            
        chat_id = update.effective_chat.id
        session = self._get_session(chat_id)
        if not session:
            await update.message.reply_text("Please /login first with 🔑 Login or /login.")
            return
        try:
            # Enable trading only for this chat
            try:
                if hasattr(self.controller, 'start_trading_for_chat'):
                    self.controller.start_trading_for_chat(chat_id, session)
                else:
                    # Legacy fallback: global
                    self.controller.set_mt5_connector(session)
                    self.controller.enable_trading()
            except Exception as e:
                logger.exception("Error enabling trading for chat")
                await update.message.reply_text(f"❌ Failed to enable trading: {e}")
                return
            # Subscribe this chat to telemetry
            try:
                self.controller.subscribe_telemetry(chat_id)
            except Exception:
                pass
            is_admin = self._is_user_admin(chat_id)
            await update.message.reply_text(
                "✅ Auto trading enabled. Use 🧠 Analyze Now anytime to see the latest analysis.",
                reply_markup=_build_main_reply_keyboard(is_admin=is_admin),
            )

            # Start AutoTrainer for continuous retraining if available
            try:
                if AutoTrainer and hasattr(self.controller, 'ai_strategy') and self.controller.ai_strategy:
                    if self.auto_trainer is None:
                        self.auto_trainer = AutoTrainer(self.controller.ai_strategy, session)
                    self.auto_trainer.start_auto_training()
            except Exception:
                logger.exception("Failed to start AutoTrainer after enabling trading")
        except Exception as e:
            logger.exception("Error enabling trading")
            await update.message.reply_text(f"❌ Failed to enable trading: {e}")

    async def _cmd_restart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.controller.stop()
        await asyncio.sleep(0.5)
        self.controller.start()
        await update.message.reply_text("Bot restarted.")

    async def _on_inline_toggle(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        if query.data == "show_keyboard":
            # If user not logged in, show minimal keyboard
            chat_id = query.message.chat_id
            is_admin = self._is_user_admin(chat_id)
            if chat_id in getattr(self, '_sessions', {}):
                count = await self._get_upcoming_count()
                kb = _build_main_reply_keyboard(count, is_admin)
            else:
                kb = _build_minimal_reply_keyboard(is_admin)
            await query.message.reply_text("Keyboard shown.", reply_markup=kb)
        elif query.data == "hide_keyboard":
            await query.message.reply_text(
                "Keyboard hidden.", reply_markup=ReplyKeyboardRemove()
            )

    async def _cmd_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        is_admin = self._is_user_admin(chat_id)
        if chat_id in getattr(self, '_sessions', {}):
            count = await self._get_upcoming_count()
            kb = _build_main_reply_keyboard(count, is_admin)
        else:
            kb = _build_minimal_reply_keyboard(is_admin)
        await update.message.reply_text("Keyboard shown.", reply_markup=kb)

    async def _cmd_close(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Keyboard hidden.", reply_markup=ReplyKeyboardRemove()
        )

    async def _get_upcoming_count(self) -> int:
        """Return number of upcoming economic events for badge."""
        try:
            import requests
            from datetime import datetime, timedelta
            country = TE_COUNTRY
            importance = TE_IMPORTANCE
            imp_map = {"low": "1", "medium": "2", "high": "3"}
            imp_param = imp_map.get(importance, None)

            d1 = (datetime.utcnow() - timedelta(days=2)).strftime('%Y-%m-%d')
            d2 = (datetime.utcnow() + timedelta(days=7)).strftime('%Y-%m-%d')
            base = "https://api.tradingeconomics.com/calendar/country/"
            url = (
                f"{base}{requests.utils.quote(country)}?c={requests.utils.quote(TE_API_CLIENT)}&format=json"
                f"&d1={d1}&d2={d2}"
            )
            if imp_param:
                url += f"&importance={imp_param}"
            resp = requests.get(url, timeout=8)
            if resp.status_code != 200:
                return 0
            data = resp.json() or []
            def parse_iso(dt_str: str):
                try:
                    return datetime.fromisoformat(dt_str.replace('Z', ''))
                except Exception:
                    return None
            now = datetime.utcnow()
            return sum(1 for e in data if (parse_iso(e.get('Date') or e.get('DateUtc') or '') or now) >= now)
        except Exception:
            return 0

    async def _cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        session = self._get_session(update.effective_chat.id)
        if not session:
            await update.message.reply_text("Please /login first.")
            return
        info = session.get_account_info()
        if not info:
            await update.message.reply_text("Unable to fetch account info.")
            return
        msg = (
            f"Balance: {info['balance']:.2f}\n"
            f"Equity: {info['equity']:.2f}\n"
            f"Margin: {info['margin']:.2f}\n"
            f"Free Margin: {info['free_margin']:.2f}"
        )
        await update.message.reply_text(msg)

    async def _cmd_account(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        session = self._get_session(update.effective_chat.id)
        if not session:
            await update.message.reply_text("Please /login first.")
            return
        info = session.get_account_info()
        if not info:
            await update.message.reply_text("Unable to fetch account info.")
            return
        # Detect account type heuristically using server/company/name
        broker_meta = (info.get('server') or '') + ' ' + (info.get('company') or '') + ' ' + (info.get('name') or '')
        broker_meta_l = broker_meta.lower()
        if 'ecn' in broker_meta_l or 'raw' in broker_meta_l or 'zero' in broker_meta_l or 'prime' in broker_meta_l:
            account_type = 'ECN'
        elif 'pro' in broker_meta_l or 'vip' in broker_meta_l:
            account_type = 'Pro'
        elif 'swap' in broker_meta_l or 'islam' in broker_meta_l or 'islamic' in broker_meta_l:
            account_type = 'Swap-free'
        elif 'standard' in broker_meta_l or 'classic' in broker_meta_l:
            account_type = 'Standard'
        else:
            account_type = 'Unknown'
        # Detect symbol variant for this account
        try:
            from config import SYMBOLS as CFG_SYMBOLS
            base_symbols = CFG_SYMBOLS if isinstance(CFG_SYMBOLS, list) else [CFG_SYMBOLS]
        except Exception:
            base_symbols = ['EURUSD','GBPUSD','USDJPY','AUDUSD','XAUUSD']
        detected_symbols = []
        try:
            detected_symbols = session.detect_symbol_variant(base_symbols)
        except Exception:
            detected_symbols = base_symbols
        symbol_variant = ','.join(detected_symbols)
        # If unknown from metadata but '+' symbols detected, hint ECN/Raw
        if account_type == 'Unknown' and any(s.endswith('+') for s in detected_symbols):
            account_type = 'ECN'

        # Check if this session is currently active
        is_active = session.is_active_connection()
        status = "🟢 Active" if is_active else "🟡 Inactive (another user connected)"
        
        msg = (
            "Account Information\n"
            "----------------------------\n"
            f"Account: {info['login']}\n"
            f"Type: {account_type}\n"
            f"Server: {info.get('server','-')}\n"
            f"Status: {status}\n"
            f"Leverage: {info['leverage']}\n"
            f"Currency: {info['currency']}\n"
            "\nBalance Information\n"
            "----------------------------\n"
            f"Balance: {info['balance']:.2f}\n"
            f"Equity: {info['equity']:.2f}\n"
            f"Margin: {info['margin']:.2f}\n"
            f"Free Margin: {info['free_margin']:.2f}"
        )
        # Append symbol mapping info
        msg2 = (
            "\n\nSymbol Mapping\n"
            "----------------------------\n"
            f"Detected: {symbol_variant}"
        )
        await update.message.reply_text(msg + msg2)

    async def _cmd_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        session = self._get_session(update.effective_chat.id)
        if not session:
            await update.message.reply_text("Please /login first.")
            return
        info = session.get_account_info()
        if not info:
            await update.message.reply_text("Unable to fetch account info.")
            return
        # Determine bot running state (based on MT5 connection and schedule loop being active)
        is_running = getattr(self.controller.mt5, 'connected', False)
        status = "Running" if is_running else "Stopped"

        # Map open positions to strategy names via magic numbers
        positions = session.get_positions()
        magic_in_positions = {p.get('magic') for p in positions} if positions else set()
        active_strategies: List[str] = []
        for s in getattr(self.controller, 'strategies', []):
            if getattr(s, 'magic_number', None) in magic_in_positions:
                active_strategies.append(getattr(s, 'name', 'Strategy'))
        if not active_strategies:
            active_strategies.append("None")

        # Compose symbols string for multi-symbol trading
        symbols_text = ", ".join(SYMBOLS) if isinstance(SYMBOLS, list) and SYMBOLS else SYMBOL

        msg = (
            "Bot Info\n"
            "----------------------------\n"
            f"Status: {status}\n"
            f"Symbols: {symbols_text}\n"
            f"Open Positions: {len(positions) if positions else 0}\n"
            f"Strategies in Use: {', '.join(sorted(set(active_strategies)))}\n"
            "\nBalance Snapshot\n"
            "----------------------------\n"
            f"Balance: {info['balance']:.2f}\n"
            f"Equity: {info['equity']:.2f}\n"
            f"Free Margin: {info['free_margin']:.2f}"
        )
        await update.message.reply_text(msg)

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        session = self._get_session(update.effective_chat.id)
        if not session:
            await update.message.reply_text("Please /login first.")
            return
        positions = session.get_positions()
        if not positions:
            await update.message.reply_text("No open positions.")
            return
        lines: List[str] = []
        for p in positions:
            lines.append(
                f"#{p['ticket']} {p['type'].upper()} {p['symbol']} vol={p['volume']} PnL={p['profit']:.2f}"
            )
        await update.message.reply_text("\n".join(lines))

    async def _cmd_orders(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        session = self._get_session(update.effective_chat.id)
        if not session:
            await update.message.reply_text("Please /login first.")
            return
        orders = session.get_orders()
        if not orders:
            await update.message.reply_text("No pending orders.")
            return
        lines: List[str] = []
        for o in orders:
            lines.append(
                f"#{o['ticket']} {o['symbol']} type={o['type']} vol={o['volume']} price={o['price_open']}"
            )
        await update.message.reply_text("\n".join(lines))

    async def _cmd_close_all(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        session = self._get_session(update.effective_chat.id)
        if not session:
            await update.message.reply_text("Please /login first.")
            return
        positions = session.get_positions()
        if not positions:
            await update.message.reply_text("No open positions to close.")
            return
        closed = 0
        for p in positions:
            try:
                if self.mt5.close_position(p['ticket']):
                    closed += 1
            except Exception:
                logger.exception("Error closing position")
        await update.message.reply_text(f"Closed {closed} positions.")

    async def _cmd_buy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Trading buttons are temporarily disabled.")

    async def _cmd_sell(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Trading buttons are temporarily disabled.")

    async def _place_market(self, update: Update, order_type: str):
        # Best-effort TP/SL using defaults and account-based lot sizing via controller logic
        session = self._get_session(update.effective_chat.id)
        if not session:
            await update.message.reply_text("Please /login first.")
            return
        prices = session.get_current_price()
        if not prices:
            await update.message.reply_text("Unable to get current price.")
            return
        point = session.symbol_info.point if session.symbol_info else 0.0001
        entry_price = prices['ask'] if order_type == 'buy' else prices['bid']
        if order_type == 'buy':
            sl = entry_price - self.default_sl_points * point
            tp = entry_price + self.default_tp_points * point
        else:
            sl = entry_price + self.default_sl_points * point
            tp = entry_price - self.default_tp_points * point

        # Risk-based lot size
        account_info = session.get_account_info()
        if not account_info:
            await update.message.reply_text("Account info unavailable.")
            return
        risk_amount = account_info['balance'] * (self.current_risk_percentage / 100)
        sl_points = int(abs(entry_price - sl) / max(point, 1e-9))
        volume = session.calculate_lot_size(risk_amount, sl_points)

        result = session.place_market_order(
            order_type=order_type,
            volume=volume,
            sl=sl,
            tp=tp,
            comment=f"Telegram {order_type}",
            magic=9999,
        )
        if result:
            try:
                sym = session.get_symbol()
            except Exception:
                sym = SYMBOL
            ticket = result.get('order') if isinstance(result, dict) else None
            await update.message.reply_text(
                f"{order_type.title()} order placed: {sym} vol={volume} | Ticket: {ticket if ticket is not None else 'n/a'}"
            )
        else:
            # Surface last_error info to user for troubleshooting
            try:
                import MetaTrader5 as mt5
                err = mt5.last_error()
                await update.message.reply_text(f"Order failed. MT5: {err}")
            except Exception:
                await update.message.reply_text("Order failed. Check MT5 and permissions.")

    async def _cmd_set_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Usage: /set_risk 1.5
        try:
            arg = context.args[0]
            value = float(arg)
            if value <= 0 or value > 10:
                raise ValueError
            self.current_risk_percentage = value
            await update.message.reply_text(f"Risk per trade set to {value}%")
        except Exception:
            await update.message.reply_text("Usage: /set_risk <percent>. Example: /set_risk 1.5")

    async def _cmd_set_tp_sl(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Usage: /set_tp_sl 100 50
        try:
            tp = int(context.args[0])
            sl = int(context.args[1])
            if tp <= 0 or sl <= 0:
                raise ValueError
            self.default_tp_points = tp
            self.default_sl_points = sl
            await update.message.reply_text(
                f"Defaults set. TP={tp} points, SL={sl} points"
            )
        except Exception:
            await update.message.reply_text("Usage: /set_tp_sl <tp_points> <sl_points>")

    async def _cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        
        # Get user-specific session stats
        user_session = self.controller._user_sessions.get(chat_id)
        if not user_session:
            await update.message.reply_text("❌ No active trading session. Please /login first.")
            return
            
        stats = user_session.get('stats', {})
        if not stats:
            await update.message.reply_text("No stats available.")
            return
            
        total = int(stats.get('total_trades', 0) or 0)
        win = int(stats.get('winning_trades', 0) or 0)
        loss = int(stats.get('losing_trades', 0) or 0)
        profit = float(stats.get('total_profit', 0.0) or 0.0)
        
        # Prefer computed total from outcomes if it's larger/more accurate
        outcomes_total = win + loss
        denom = outcomes_total if outcomes_total > 0 else max(total, 1)
        # Clamp win to denominator to avoid >100%
        win_clamped = min(max(win, 0), denom)
        win_rate = (win_clamped / denom) * 100.0
        
        # Get session info
        session_start = stats.get('session_start', 'Unknown')
        if isinstance(session_start, str):
            session_display = session_start
        else:
            session_display = session_start.strftime('%Y-%m-%d %H:%M') if session_start else 'Unknown'
            
        msg = (
            f"📊 **Performance Report**\n\n"
            f"**Trades:** {total} (W:{win} L:{loss})\n"
            f"**Win Rate:** {win_rate:.1f}%\n"
            f"**Total P/L:** {profit:.2f}\n"
            f"**Session Started:** {session_display}"
        )
        await update.message.reply_text(msg, parse_mode='Markdown')

    async def _cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Usage: /history [day|week|month], default day
        period = 'day'
        if context.args:
            arg = context.args[0].lower()
            if arg in ('day', 'week', 'month'):
                period = arg
        session = self._get_session(update.effective_chat.id)
        if not session:
            await update.message.reply_text("Please /login first.")
            return
        
        # Get history deals - ensure we get all symbols
        deals = session.get_recent_history(period)
        if not deals:
            await update.message.reply_text("No recent history found.")
            return
        
        # Group by symbol and show totals
        symbol_totals = {}
        lines = []
        
        # Show all deals, not just last 20, to ensure we see all symbols
        for d in deals:
            symbol = d['symbol']
            if symbol not in symbol_totals:
                symbol_totals[symbol] = {'count': 0, 'total_pnl': 0.0}
            symbol_totals[symbol]['count'] += 1
            symbol_totals[symbol]['total_pnl'] += d['profit']
            
            dt = d['time'].strftime('%Y-%m-%d %H:%M')
            lines.append(
                f"{dt} {symbol} vol={d['volume']} price={d['price']} P/L={d['profit']:.2f}"
            )
        
        # Add summary at the top for better visibility
        if symbol_totals:
            summary_lines = ["📊 History Summary:"]
            for symbol, totals in symbol_totals.items():
                summary_lines.append(f"{symbol}: {totals['count']} deals, P/L: {totals['total_pnl']:.2f}")
            summary_lines.append("")  # Empty line separator
            
            # Combine summary with deal details
            full_message = "\n".join(summary_lines + lines)
        else:
            full_message = "\n".join(lines)
        
        # Split message if too long (Telegram has 4096 char limit)
        if len(full_message) > 4000:
            # Send summary first
            await update.message.reply_text("\n".join(summary_lines))
            # Then send deals in chunks
            chunk_size = 20
            for i in range(0, len(lines), chunk_size):
                chunk = lines[i:i+chunk_size]
                await update.message.reply_text("\n".join(chunk))
        else:
            await update.message.reply_text(full_message)

    async def _cmd_debug(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Debug command to show session and symbol information"""
        session = self._get_session(update.effective_chat.id)
        if not session:
            await update.message.reply_text("Please /login first.")
            return
        
        # Get current session info
        current_symbol = session.get_symbol()
        account_info = session.get_account_info()
        
        # Get recent history to check what symbols are found
        deals = session.get_recent_history('day')
        symbols_found = set(d['symbol'] for d in deals) if deals else set()
        
        # Get configured symbols from config
        from config import SYMBOLS, SYMBOL
        
        debug_info = [
            "🐛 Debug Information:",
            f"Current session symbol: {current_symbol}",
            f"Configured symbols: {SYMBOLS}",
            f"Default symbol: {SYMBOL}",
            f"Account: {account_info['login'] if account_info else 'N/A'}",
            f"History deals found: {len(deals) if deals else 0}",
            f"Symbols in history: {list(symbols_found)}",
            f"Session connected: {session.connected}",
        ]
        
        await update.message.reply_text("\n".join(debug_info))

    async def _cmd_alerts_on(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.alerts_enabled = True
        await update.message.reply_text("Alerts enabled.")

    async def _cmd_alerts_off(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.alerts_enabled = False
        await update.message.reply_text("Alerts disabled.")

    async def _cmd_alerts_toggle(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.alerts_enabled = not self.alerts_enabled
        state = "enabled" if self.alerts_enabled else "disabled"
        await update.message.reply_text(f"Alerts {state}.")

    async def _cmd_news(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        import base64
        import requests
        try:
            # Mode selection: 'headlines' -> NewsAPI; otherwise TradingEconomics calendar
            if context.args and context.args[0].lower() == 'headlines':
                country = NEWS_COUNTRY
                category = NEWS_CATEGORY
                if len(context.args) >= 2:
                    country = context.args[1]
                if len(context.args) >= 3:
                    category = context.args[2]
                if not NEWS_API_KEY:
                    await update.message.reply_text("NEWS_API_KEY not configured.")
                    return
                url = (
                    f"https://newsapi.org/v2/top-headlines?country={country}"
                    f"&category={category}&pageSize=5&apiKey={NEWS_API_KEY}"
                )
                resp = requests.get(url, timeout=10)
                if resp.status_code != 200:
                    await update.message.reply_text("Failed to fetch headlines.")
                    return
                data = resp.json()
                articles = data.get('articles', [])[:5]
                if not articles:
                    await update.message.reply_text("No headlines available.")
                    return
                lines = []
                for a in articles:
                    title = a.get('title', 'Untitled')
                    source = (a.get('source') or {}).get('name', '')
                    lines.append(f"- {title} ({source})")
                await update.message.reply_text("\n".join(lines))
                return

            # Default: Economic calendar via TradingEconomics API
            country = TE_COUNTRY
            importance = TE_IMPORTANCE
            if context.args:
                # Allow override: /news <country> <importance>
                if len(context.args) >= 1:
                    country = " ".join(context.args[:-1]) if len(context.args) > 1 else context.args[0]
                if len(context.args) >= 2:
                    imp_candidate = context.args[-1].lower()
                    if imp_candidate in ("low", "medium", "high", "all"):
                        importance = imp_candidate

            # Importance mapping to TE levels (1=low,2=medium,3=high)
            imp_map = {"low": "1", "medium": "2", "high": "3"}
            imp_param = imp_map.get(importance, None)

            # TradingEconomics calendar with date window: last 2 days to next 7 days
            import datetime as dt
            d1 = (dt.datetime.utcnow() - dt.timedelta(days=2)).strftime('%Y-%m-%d')
            d2 = (dt.datetime.utcnow() + dt.timedelta(days=7)).strftime('%Y-%m-%d')

            base = "https://api.tradingeconomics.com/calendar/country/"
            url = (
                f"{base}{requests.utils.quote(country)}?c={requests.utils.quote(TE_API_CLIENT)}&format=json"
                f"&d1={d1}&d2={d2}"
            )
            if imp_param:
                url += f"&importance={imp_param}"

            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                await update.message.reply_text("Failed to fetch calendar. Check TE_API_CLIENT.")
                return
            data = resp.json()
            if not data:
                await update.message.reply_text("No upcoming events.")
                return
            from datetime import datetime

            def format_dt(dt_str: str) -> str:
                try:
                    # Expecting ISO-like strings from TE
                    dt = datetime.fromisoformat(dt_str.replace('Z', ''))
                    return dt.strftime('%b %d, %Y %H:%M UTC')
                except Exception:
                    return dt_str

            def format_importance(val) -> str:
                # TE Importance may be 1/2/3 or text; normalize
                s = str(val).strip().lower()
                if s in ('3', 'high'):
                    return '🔴 High'
                if s in ('2', 'medium'):
                    return '🔸 Medium'
                if s in ('1', 'low'):
                    return '🔹 Low'
                return s.capitalize() if s else '—'

            def parse_iso(dt_str: str):
                try:
                    return datetime.fromisoformat(dt_str.replace('Z', ''))
                except Exception:
                    return None

            now = datetime.utcnow()
            upcoming: List[tuple] = []
            past: List[tuple] = []
            for e in data:
                dt_raw = e.get('Date') or e.get('DateUtc') or ''
                dt_obj = parse_iso(dt_raw)
                if not dt_obj:
                    continue
                if dt_obj >= now:
                    upcoming.append((dt_obj, e))
                else:
                    past.append((dt_obj, e))

            upcoming.sort(key=lambda x: x[0])
            past.sort(key=lambda x: x[0], reverse=True)

            def render(ev):
                when = format_dt(ev.get('Date') or ev.get('DateUtc') or '')
                event = ev.get('Event', 'Event')
                cur = ev.get('Currency') or ''
                imp = format_importance(ev.get('Importance'))
                actual = ev.get('Actual', '-') or '-'
                forecast = ev.get('Forecast', '-') or '-'
                prevv = ev.get('Previous', '-') or '-'
                title = f"📅 {when} — {event}"
                if cur:
                    title += f" ({cur})"
                title += f"  [{imp}]"
                details = f"Actual: {actual}  |  Forecast: {forecast}  |  Previous: {prevv}"
                return f"{title}\n{details}"

            blocks: List[str] = []
            if upcoming:
                blocks.append("🔜 Upcoming\n" + "\n\n".join(render(e) for _, e in upcoming[:5]))
            else:
                blocks.append("🔜 Upcoming\nCurrently no upcoming News Event")
            if past:
                blocks.append("🕘 Past\n" + "\n\n".join(render(e) for _, e in past[:5]))

            await update.message.reply_text("\n\n".join(blocks))

            # Update keyboard with badge
            try:
                is_admin = self._is_user_admin(update.effective_chat.id)
                await update.message.reply_text(
                    "Menu updated.", reply_markup=_build_main_reply_keyboard(len(upcoming), is_admin)
                )
            except Exception:
                pass
        except Exception:
            logger.exception("Error fetching news/calendar")
            await update.message.reply_text("Error fetching news/calendar.")

    async def _cmd_sessions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show all active MT5 sessions"""
        try:
            if not self._sessions:
                await update.message.reply_text("No active MT5 sessions.")
                return
            
            # Get current active account
            current_account = None
            try:
                import MetaTrader5 as mt5
                account_info = mt5.account_info()
                if account_info:
                    current_account = account_info.login
            except Exception:
                pass
            
            message_lines = ["🔗 **Active MT5 Sessions**\n"]
            
            for chat_id, session in self._sessions.items():
                is_active = session.is_active_connection()
                status = "🟢 Active" if is_active else "🟡 Inactive"
                
                # Get account info
                try:
                    info = session.get_account_info()
                    if info:
                        account_display = f"{info['login']} ({info['balance']:.2f} {info['currency']})"
                    else:
                        account_display = f"{session._login} (info unavailable)"
                except Exception:
                    account_display = f"{session._login} (error)"
                
                message_lines.append(f"Chat {chat_id}: {account_display} - {status}")
            
            if current_account:
                message_lines.append(f"\n🌐 Currently connected to: {current_account}")
            
            message = "\n".join(message_lines)
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in sessions command: {e}")
            await update.message.reply_text(f"❌ Error getting sessions: {e}")

    async def _cmd_switch(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Switch to your MT5 account (force connection)"""
        try:
            session = self._get_session(update.effective_chat.id)
            if not session:
                await update.message.reply_text("Please /login first.")
                return
            
            # Force connection to this user's account
            if session.ensure_connection():
                info = session.get_account_info()
                if info:
                    await update.message.reply_text(
                        f"✅ Switched to account: {info['login']}\n"
                        f"Balance: {info['balance']:.2f} {info['currency']}"
                    )
                else:
                    await update.message.reply_text(f"✅ Switched to account: {session._login}")
            else:
                await update.message.reply_text("❌ Failed to switch to your account. Please try /login again.")
                
        except Exception as e:
            logger.error(f"Error in switch command: {e}")
            await update.message.reply_text(f"❌ Error switching account: {e}")

    async def _cmd_ai_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show AI strategy status"""
        try:
            # Check if AI is available
            if not hasattr(self.controller, 'ai_strategy') or self.controller.ai_strategy is None:
                await update.message.reply_text("❌ AI Strategy not available. Please ensure AI components are properly installed.")
                return
            
            ai_strategy = self.controller.ai_strategy
            
            # Get strategy info with error handling
            try:
                status = ai_strategy.get_strategy_info()
            except Exception as e:
                logger.error(f"Error getting strategy info: {e}")
                status = {
                    'is_trained': False,
                    'enabled': False,
                    'prediction_horizon': 5,
                    'min_confidence_threshold': 0.6,
                    'risk_reward_ratio': 2.0,
                    'available_models': []
                }
            
            # Get performance info with error handling
            try:
                performance = ai_strategy.get_model_performance()
            except Exception as e:
                logger.error(f"Error getting performance info: {e}")
                performance = {
                    'prediction_accuracy': {
                        'total_predictions': 0,
                        'accuracy': 0.0
                    },
                    'last_prediction': None
                }
            
            # Build status message
            message_lines = [
                "🤖 <b>AI Strategy Status</b>",
                "",
                f"<b>Training Status:</b> {'✅ Trained' if status.get('is_trained', False) else '❌ Not Trained'}",
                f"<b>Enabled:</b> {'✅ Yes' if status.get('enabled', False) else '❌ No'}",
                f"<b>Prediction Horizon:</b> {status.get('prediction_horizon', 5)} periods",
                f"<b>Confidence Threshold:</b> {status.get('min_confidence_threshold', 0.6):.2f}",
                f"<b>Risk/Reward Ratio:</b> {status.get('risk_reward_ratio', 2.0):.1f}",
                "",
                "<b>Performance Metrics:</b>",
                f"• Total Predictions: {performance.get('prediction_accuracy', {}).get('total_predictions', 0)}",
                f"• Accuracy: {performance.get('prediction_accuracy', {}).get('accuracy', 0.0):.3f}",
            ]
            
            # Add last confidence if available
            last_prediction = performance.get('last_prediction')
            if last_prediction and isinstance(last_prediction, dict):
                confidence = last_prediction.get('confidence', 0)
                message_lines.append(f"• Last Confidence: {confidence:.3f}")
            else:
                message_lines.append("• Last Confidence: N/A")
            
            message_lines.extend([
                "",
                "<b>Available Models:</b>",
            ])
            
            available_models = status.get('available_models', [])
            if available_models:
                for model in available_models:
                    message_lines.append(f"• {model}")
            else:
                message_lines.append("• No models available")
            
            message = "\n".join(message_lines)
            await update.message.reply_text(message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Error in AI status command: {e}")
            await update.message.reply_text(f"❌ Error getting AI status: {e}")

    async def _cmd_ai_train(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Train AI models"""
        try:
            # Check if AI is available
            if not hasattr(self.controller, 'ai_strategy') or self.controller.ai_strategy is None:
                await update.message.reply_text("❌ AI Strategy not available. Please ensure AI components are properly installed.")
                return
            
            # Get data periods parameter
            data_periods = 2000
            if context.args and len(context.args) > 0:
                try:
                    data_periods = int(context.args[0])
                    data_periods = max(500, min(data_periods, 10000))  # Limit between 500-10000
                except ValueError:
                    await update.message.reply_text("❌ Invalid data periods. Using default 2000.")
            
            await update.message.reply_text(f"🤖 Starting AI training with {data_periods} data points...")
            
            # Get historical data and train
            session = self._get_session(update.effective_chat.id)
            if not session:
                await update.message.reply_text("Please /login first.")
                return
            
            # Send progress update
            progress_msg = await update.message.reply_text("📊 Fetching historical data...")
            
            # Get historical data for training
            historical_data = []
            for symbol in SYMBOLS:
                try:
                    session.change_symbol(symbol)
                    df = session.get_rates(symbol, TIMEFRAME, data_periods)
                    if df is not None and not df.empty:
                        df['symbol'] = symbol
                        historical_data.append(df)
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {e}")
                    continue
            
            if not historical_data:
                await update.message.reply_text("❌ Failed to get historical data for training")
                return
            
            # Combine data
            import pandas as pd
            if historical_data:
                combined_data = pd.concat(historical_data, ignore_index=True)
                
                # Handle time column properly
                if 'time' in combined_data.columns:
                    combined_data = combined_data.sort_values('time')
                else:
                    # If no time column, use index
                    combined_data = combined_data.sort_index()
                
                combined_data = combined_data.drop_duplicates()
                
                # Reset index to avoid RangeIndex issues
                combined_data = combined_data.reset_index(drop=True)
                
                # Ensure we have the required OHLC columns
                required_columns = ['open', 'high', 'low', 'close']
                if not all(col in combined_data.columns for col in required_columns):
                    await update.message.reply_text("❌ Historical data missing required OHLC columns")
                    return
            else:
                await update.message.reply_text("❌ No historical data available for training")
                return
            
            # Update progress
            await progress_msg.edit_text("🧠 Training AI models... (This may take a few minutes)")
            
            # Train models
            result = self.controller.ai_strategy.train_models(combined_data)
            
            if 'error' in result:
                await update.message.reply_text(f"❌ Training failed: {result['error']}")
            else:
                # Compute per-symbol accuracy using the trained ensemble
                try:
                    ai_strategy = self.controller.ai_strategy
                    per_symbol = {}
                    symbols_in_data = sorted(set(combined_data.get('symbol', []))) if 'symbol' in combined_data.columns else []
                    for sym in symbols_in_data:
                        df_sym = combined_data[combined_data['symbol'] == sym]
                        # Build features and targets for this symbol
                        feat_df = ai_strategy.data_processor.create_features(df_sym)
                        features, targets = ai_strategy.data_processor.prepare_training_data(
                            feat_df, df_sym, prediction_horizon=ai_strategy.prediction_horizon
                        )
                        if features.empty or targets is None or len(targets) == 0:
                            continue
                        X_sym = ai_strategy.data_processor.transform_features(features)
                        if X_sym.size == 0:
                            continue
                        preds, _ = ai_strategy.model_manager.predict(X_sym, 'ensemble')
                        if preds.size == 0:
                            continue
                        import numpy as np
                        acc = float((preds == targets.values[:len(preds)]).mean())
                        per_symbol[sym] = acc
                    # Persist into model metadata and save
                    if per_symbol:
                        ai_strategy.model_manager.model_metadata['per_symbol_accuracy'] = per_symbol
                        ai_strategy.model_manager.save_models()
                except Exception as ex:
                    logger.warning(f"Failed to compute per-symbol accuracy: {ex}")

                # Export combined dataset to ai/data
                try:
                    import os
                    from datetime import datetime
                    os.makedirs('ai/data', exist_ok=True)
                    # Build a short symbols tag (first 5 symbols if long)
                    sym_list = SYMBOLS if isinstance(SYMBOLS, list) else [SYMBOLS]
                    sym_tag = ",".join(sym_list[:5])
                    if len(sym_list) > 5:
                        sym_tag += ",…"
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    safe_tag = sym_tag.replace('/', '').replace(' ', '')
                    csv_path = os.path.join('ai/data', f'train_{safe_tag}_{timestamp}.csv')
                    combined_data.to_csv(csv_path, index=False)
                    await update.message.reply_text("💾 Training dataset saved successfully")
                except Exception as ex:
                    logger.warning(f"Failed to export training dataset: {ex}")

                message_lines = [
                    "✅ <b>AI Training Completed</b>",
                    "",
                    f"<b>Status:</b> {result.get('status', 'Unknown')}",
                    f"<b>Samples:</b> {result.get('n_samples', 0)}",
                    f"<b>Features:</b> {result.get('n_features', 0)}",
                    f"<b>Models Trained:</b> {result.get('models_trained', 0)}",
                ]
                
                # Add training results
                training_results = result.get('training_results', {})
                if training_results:
                    message_lines.extend(["", "<b>Model Performance:</b>"])
                    for model_name, model_result in training_results.items():
                        if 'error' not in model_result:
                            val_score = model_result.get('val_score', 0)
                            cv_mean = model_result.get('cv_mean', 0)
                            message_lines.append(f"• {model_name}: {val_score:.3f} (CV: {cv_mean:.3f})")
                # Append per-symbol accuracy if available
                try:
                    per_symbol = self.controller.ai_strategy.model_manager.model_metadata.get('per_symbol_accuracy', {})
                    if per_symbol:
                        message_lines.extend(["", "<b>Per-Symbol Accuracy:</b>"])
                        for sym, acc in per_symbol.items():
                            message_lines.append(f"• {sym}: {acc:.3f}")
                except Exception:
                    pass
                
                message = "\n".join(message_lines)
                await update.message.reply_text(message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Error in AI train command: {e}")
            await update.message.reply_text(f"❌ Training error: {e}")

    async def _cmd_ai_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show AI performance report"""
        try:
            # Check if AI is available
            if not hasattr(self.controller, 'ai_strategy') or self.controller.ai_strategy is None:
                await update.message.reply_text("❌ AI Strategy not available. Please ensure AI components are properly installed.")
                return
            
            ai_strategy = self.controller.ai_strategy
            
            # Get performance info with error handling
            try:
                performance = ai_strategy.get_model_performance()
            except Exception as e:
                logger.error(f"Error getting performance info: {e}")
                performance = {
                    'prediction_accuracy': {
                        'total_predictions': 0,
                        'correct_predictions': 0,
                        'accuracy': 0.0
                    },
                    'is_trained': False,
                    'total_predictions': 0,
                    'last_prediction': None
                }
            
            message_lines = [
                "📊 **AI Performance Report**",
                "",
                "**Prediction Accuracy:**",
                f"• Total Predictions: {performance.get('prediction_accuracy', {}).get('total_predictions', 0)}",
                f"• Correct Predictions: {performance.get('prediction_accuracy', {}).get('correct_predictions', 0)}",
                f"• Accuracy: {performance.get('prediction_accuracy', {}).get('accuracy', 0.0):.3f}",
                "",
                "**Model Information:**",
                f"• Is Trained: {'✅ Yes' if performance.get('is_trained', False) else '❌ No'}",
                f"• Total Predictions Made: {performance.get('total_predictions', 0)}",
            ]
            
            # Add last prediction info
            last_prediction = performance.get('last_prediction')
            if last_prediction and isinstance(last_prediction, dict):
                message_lines.extend([
                    "",
                    "**Last Prediction:**",
                    f"• Time: {last_prediction.get('timestamp', 'N/A')}",
                    f"• Prediction: {last_prediction.get('prediction', 'N/A')}",
                    f"• Confidence: {last_prediction.get('confidence', 0.0):.3f}",
                ])
                # Add drivers
                try:
                    ex = last_prediction.get('explain') or {}
                    top_pos = ex.get('top_positive') or []
                    top_neg = ex.get('top_negative') or []
                    if top_pos or top_neg:
                        message_lines.append("• Top Drivers:")
                        if top_pos:
                            message_lines.append("  + " + ", ".join([f"{k} ({v:.3f})" for k, v in top_pos[:3]]))
                        if top_neg:
                            message_lines.append("  - " + ", ".join([f"{k} ({v:.3f})" for k, v in top_neg[:3]]))
                except Exception:
                    pass
            else:
                message_lines.extend(["", "**Last Prediction:**", "• No predictions made yet"]) 

            # Global feature importance summary
            try:
                feature_names = ai_strategy.data_processor.feature_names or []
                if feature_names:
                    global_imps = ai_strategy.model_manager.get_global_feature_importance(feature_names)
                    avg = {}
                    for _, m in global_imps.items():
                        for f, v in m.items():
                            avg[f] = avg.get(f, 0.0) + v
                    if avg:
                        total_models = max(1, len(global_imps))
                        for f in list(avg.keys()):
                            avg[f] /= total_models
                        top = sorted(avg.items(), key=lambda x: x[1], reverse=True)[:5]
                        message_lines.extend(["", "**Global Feature Importance (avg):**", "• " + ", ".join([f"{k} ({v:.3f})" for k, v in top])])
            except Exception:
                pass
            
            message = "\n".join(message_lines)
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in AI performance command: {e}")
            await update.message.reply_text(f"❌ Error getting performance: {e}")
    
    async def _cmd_close_reasons(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show close reasons statistics"""
        try:
            if not self.controller or not hasattr(self.controller, 'get_close_reasons_stats'):
                await update.message.reply_text("❌ Close reasons tracking not available")
                return
            
            stats = self.controller.get_close_reasons_stats()
            
            if stats['total_closes'] == 0:
                await update.message.reply_text("📊 **Close Reasons Report**\n\nNo positions closed yet.")
                return
            
            message_parts = [
                "📊 <b>Close Reasons Report</b>",
                f"Total Closes: {stats['total_closes']}",
                "",
                "<b>Close Reasons:</b>"
            ]
            
            # Show reason counts
            for reason, count in stats['reasons'].items():
                avg_profit = stats['avg_profit_by_reason'].get(reason, 0)
                profit_emoji = "💰" if avg_profit > 0 else "📉" if avg_profit < 0 else "➖"
                message_parts.append(f"{profit_emoji} {reason}: {count} times (Avg P/L: {avg_profit:.2f})")
            
            message_parts.extend([
                "",
                "<b>By Strategy:</b>"
            ])
            
            # Show strategy counts
            for strategy, count in stats['strategies'].items():
                message_parts.append(f"• {strategy}: {count} closes")
            
            # Show recent closes
            if stats['recent_closes']:
                message_parts.extend([
                    "",
                    "<b>Recent Closes:</b>"
                ])
                for close in stats['recent_closes'][-3:]:  # Show last 3
                    profit_emoji = "💰" if close['profit'] > 0 else "📉"
                    timestamp = close['timestamp'].strftime("%H:%M")
                    message_parts.append(
                        f"{profit_emoji} #{close['ticket']} {close['symbol']} "
                        f"({close['type']}) - {close['reason']} - P/L: {close['profit']:.2f} [{timestamp}]"
                    )
            
            message = "\n".join(message_parts)
            await update.message.reply_text(message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Error in close reasons command: {e}")
            await update.message.reply_text("❌ Error retrieving close reasons data")

    async def _cmd_db_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show database statistics (admin command)"""
        try:
            # Check admin authorization first
            if not await self._check_admin_authorization(update, context):
                return
            stats = db_manager.get_database_stats()
            
            message_lines = [
                "📊 Database Statistics",
                "",
                f"Total Bot Users: {stats['total_bot_users']}",
                f"Total MT Accounts: {stats['total_mt_accounts']}",
                f"Active Sessions: {stats['active_sessions']}",
                f"Database Path: {stats['database_path']}",
            ]
            
            if 'error' in stats:
                message_lines.append(f"Error: {stats['error']}")
            
            message = "\n".join(message_lines)
            await update.message.reply_text(message)
            
        except Exception as e:
            logger.error(f"Error in database stats command: {e}")
            await update.message.reply_text(f"❌ Error getting database stats: {e}")

    async def _cmd_add_user(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Add a new bot user (admin command)"""
        try:
            # Check admin authorization first
            if not await self._check_admin_authorization(update, context):
                return
            if not context.args:
                await update.message.reply_text("Usage: /add_user <telegram_chat_id> [admin]")
                return
            
            try:
                telegram_chat_id = int(context.args[0])
            except ValueError:
                await update.message.reply_text("❌ Invalid telegram_chat_id. Must be a number.")
                return
            
            # Check if admin flag is provided
            is_admin = len(context.args) > 1 and context.args[1].lower() in ['admin', 'true', '1', 'yes']
            
            # Add user to database
            bot_user_id = db_manager.add_bot_user(telegram_chat_id, is_admin)
            
            if bot_user_id:
                admin_status = "Admin" if is_admin else "Regular User"
                await update.message.reply_text(
                    f"✅ Added bot user successfully!\n"
                    f"Bot User ID: {bot_user_id}\n"
                    f"Telegram Chat ID: {telegram_chat_id}\n"
                    f"Role: {admin_status}"
                )
            else:
                await update.message.reply_text("❌ Failed to add bot user. User might already exist.")
                
        except Exception as e:
            logger.error(f"Error in add user command: {e}")
            await update.message.reply_text(f"❌ Error adding user: {e}")

    async def _cmd_list_users(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List all bot users (admin command)"""
        try:
            # Check admin authorization first
            if not await self._check_admin_authorization(update, context):
                return
            users = db_manager.get_all_bot_users()
            
            if not users:
                await update.message.reply_text("📋 Bot Users\n\nNo users found.")
                return
            
            message_lines = ["📋 Bot Users", ""]
            
            for user in users:
                # Check if user has an MT account
                mt_account = db_manager.get_mt_account_by_bot_user_id(user['bot_user_id'])
                account_status = f"MT Account: {mt_account['mt_account_number']}" if mt_account else "No MT Account"
                role_status = "👑 Admin" if user['is_admin'] else "👤 User"
                
                message_lines.append(
                    f"User {user['bot_user_id']}:\n"
                    f"• Telegram Chat ID: {user['telegram_chat_id']}\n"
                    f"• Role: {role_status}\n"
                    f"• {account_status}\n"
                    f"• Created: {user['created_at']}\n"
                )
            
            message = "\n".join(message_lines)
            
            # Split message if too long
            if len(message) > 4000:
                # Send first part
                first_part = "\n".join(message_lines[:10])  # First 10 lines
                await update.message.reply_text(first_part)
                
                # Send remaining parts
                remaining_lines = message_lines[10:]
                for i in range(0, len(remaining_lines), 10):
                    chunk = "\n".join(remaining_lines[i:i+10])
                    await update.message.reply_text(chunk)
            else:
                await update.message.reply_text(message)
                
        except Exception as e:
            logger.error(f"Error in list users command: {e}")
            await update.message.reply_text(f"❌ Error listing users: {e}")

    async def _cmd_admin_panel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show admin panel with available admin commands"""
        try:
            # Check admin authorization first
            if not await self._check_admin_authorization(update, context):
                return
            
            message_lines = [
                "👑 Admin Panel",
                "",
                "Available Admin Commands:",
                "• /add_user <telegram_chat_id> [admin] - Add new user",
                "• /list_users - List all users",
                "• /db_stats - View database statistics",
                "",
                "Admin Buttons:",
                "• ➕ Add User - Add new user (interactive)",
                "• 📋 List Users - Show all users",
                "• 📊 DB Stats - Database statistics",
                "",
                "Usage Examples:",
                "• Add regular user: /add_user 123456789",
                "• Add admin user: /add_user 123456789 admin",
                "• View users: /list_users",
                "• Database stats: /db_stats"
            ]
            
            message = "\n".join(message_lines)
            await update.message.reply_text(message)
            
        except Exception as e:
            logger.error(f"Error in admin panel command: {e}")
            await update.message.reply_text(f"❌ Error showing admin panel: {e}")

    async def _cmd_add_user_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Interactive add user via button (admin command)"""
        try:
            # Check admin authorization first
            if not await self._check_admin_authorization(update, context):
                return
            
            await update.message.reply_text(
                "👑 Add New User\n\n"
                "Please send the Telegram Chat ID of the user you want to add.\n\n"
                "Format: Just send the number (e.g., 123456789)\n\n"
                "To add as admin: Send 'admin' after the chat ID (e.g., 123456789 admin)"
            )
            
            # Set state for interactive user addition
            chat_id = update.effective_chat.id
            self._login_states[chat_id] = {"stage": "add_user"}
            
        except Exception as e:
            logger.error(f"Error in add user button command: {e}")
            await update.message.reply_text(f"❌ Error: {e}")

    async def _on_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text:
            return
        text = update.message.text.strip().lower()
        # Handle interactive login wizard and admin functions
        chat_id = update.effective_chat.id
        state = self._login_states.get(chat_id)
        if state:
            stage = state.get("stage")
            if stage == "add_user":
                # Handle interactive user addition
                try:
                    parts = update.message.text.strip().split()
                    if len(parts) < 1:
                        await update.message.reply_text("❌ Please provide a Telegram Chat ID.")
                        return
                    
                    telegram_chat_id = int(parts[0])
                    is_admin = len(parts) > 1 and parts[1].lower() in ['admin', 'true', '1', 'yes']
                    
                    # Add user to database
                    bot_user_id = db_manager.add_bot_user(telegram_chat_id, is_admin)
                    
                    if bot_user_id:
                        admin_status = "Admin" if is_admin else "Regular User"
                        await update.message.reply_text(
                            f"✅ Added bot user successfully!\n"
                            f"Bot User ID: {bot_user_id}\n"
                            f"Telegram Chat ID: {telegram_chat_id}\n"
                            f"Role: {admin_status}"
                        )
                    else:
                        await update.message.reply_text("❌ Failed to add bot user. User might already exist.")
                    
                except ValueError:
                    await update.message.reply_text("❌ Invalid Telegram Chat ID. Must be a number.")
                except Exception as e:
                    await update.message.reply_text(f"❌ Error adding user: {e}")
                finally:
                    self._login_states.pop(chat_id, None)
                return
            elif stage == "account":
                try:
                    state["login"] = int(update.message.text.strip())
                    state["stage"] = "password"
                    await update.message.reply_text("Enter Password:")
                except Exception:
                    await update.message.reply_text("Invalid account. Enter numeric account:")
                return
            if stage == "password":
                state["password"] = update.message.text.strip()
                state["stage"] = "server"
                await update.message.reply_text("Enter Server (e.g., VantageInternational-Demo):")
                return
            if stage == "server":
                state["server"] = update.message.text.strip()
                # Attempt login
                try:
                    # Check if user is already logged in with a different account
                    existing_session = self._sessions.get(chat_id)
                    if existing_session and existing_session._login != state["login"]:
                        await update.message.reply_text(
                            f"You are already logged in with account {existing_session._login}. "
                            f"Please logout first before switching to account {state['login']}."
                        )
                        self._login_states.pop(chat_id, None)
                        return
                    
                    session = MT5Connector(login=state["login"], password=state["password"], server=state["server"])
                    if not session.connect():
                        try:
                            msg = session.get_last_error_message()
                        except Exception:
                            msg = "Login failed. Check credentials/server and try /login again."
                        await update.message.reply_text(f"❌ {msg}")
                    else:
                        self._sessions[chat_id] = session
                        
                        # Get bot user info and store MT account in database
                        bot_user = db_manager.get_bot_user_by_telegram_chat_id(chat_id)
                        if bot_user:
                            db_manager.add_mt_account(bot_user['bot_user_id'], state["login"])
                            logger.info(f"Stored MT account {state['login']} for bot_user_id {bot_user['bot_user_id']}")
                        
                        info = session.get_account_info() or {}
                        await update.message.reply_text(
                            f"✅ Logged in to account: {info.get('login', state['login'])}\n"
                            f"Balance: {info.get('balance', 0):.2f} {info.get('currency', '')}"
                        )
                        try:
                            count = await self._get_upcoming_count()
                            is_admin = self._is_user_admin(chat_id)
                            await update.message.reply_text(
                                "Keyboard updated.", reply_markup=_build_main_reply_keyboard(count, is_admin)
                            )
                        except Exception:
                            pass
                except Exception as e:
                    await update.message.reply_text(f"Login error: {e}")
                finally:
                    self._login_states.pop(chat_id, None)
                return
        # Map keyboard labels to actions
        if text in ("info", "ℹ️ info"):
            await self._cmd_info(update, context)
        elif text in ("account", "👤 account"):
            await self._cmd_account(update, context)
        elif text in ("positions", "📊 positions"):
            await self._cmd_positions(update, context)
        elif text in ("orders", "📋 orders"):
            await self._cmd_orders(update, context)
        # Buy/Sell commands removed
        elif text in ("start trade", "▶️ start trade", "🟢 start trade"):
            await self._cmd_start_trade(update, context)
        elif text in ("end trade", "⏹️ end trade", "🔴 end trade"):
            await self._cmd_stop(update, context)
        elif text in ("performance", "📈 performance"):
            await self._cmd_performance(update, context)
        elif text in ("history", "🧾 history"):
            await self._cmd_history(update, context)
        # Alerts button removed
        elif text in ("news", "📰 news"):
            await self._cmd_news(update, context)
        elif text in ("analyze now", "🔎 analyze now", "🧠 analyze now"):
            await self._cmd_analyze_now(update, context)
        elif text in ("ai status", "🤖 ai status"):
            await self._cmd_ai_status(update, context)
        elif text in ("ai train", "🤖 ai train", "🚀 ai train"):
            await self._cmd_ai_train(update, context)
        elif text in ("ai performance", "🤖 ai performance", "📈 ai performance"):
            await self._cmd_ai_performance(update, context)
        elif text in ("close reasons", "📊 close reasons", "⚠️ close reasons"):
            await self._cmd_close_reasons(update, context)
        elif text in ("debug", "🐛 debug"):
            await self._cmd_debug(update, context)
        elif text in ("login", "🔑 login"):
            # kick off interactive login
            self._login_states[chat_id] = {"stage": "account"}
            await update.message.reply_text("Please enter your Account (login) number:")
        # Admin-only button handlers
        elif text in ("admin panel", "👑 admin panel"):
            await self._cmd_admin_panel(update, context)
        elif text in ("add user", "➕ add user"):
            await self._cmd_add_user_button(update, context)
        elif text in ("list users", "📋 list users"):
            await self._cmd_list_users(update, context)
        elif text in ("db stats", "📊 db stats"):
            await self._cmd_db_stats(update, context)
        elif text == "show keyboard":
            await self._cmd_menu(update, context)
        elif text == "hide keyboard":
            await self._cmd_close(update, context)
        else:
            # Ignore unknown texts
            return

    async def _register(self):
        application = (
            ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        )

        # Command handlers
        application.add_handler(CommandHandler("start", self._cmd_start))
        application.add_handler(CommandHandler("stop", self._cmd_stop))
        application.add_handler(CommandHandler("close_all", self._cmd_close_all))
        application.add_handler(CommandHandler("restart", self._cmd_restart))
        application.add_handler(CommandHandler("info", self._cmd_info))
        application.add_handler(CommandHandler("login", self._cmd_login))
        application.add_handler(CommandHandler("logout", self._cmd_logout))
        application.add_handler(CommandHandler("analyze_now", self._cmd_analyze_now))
        application.add_handler(CommandHandler("balance", self._cmd_balance))
        application.add_handler(CommandHandler("account", self._cmd_account))
        application.add_handler(CommandHandler("positions", self._cmd_positions))
        application.add_handler(CommandHandler("orders", self._cmd_orders))
        application.add_handler(CommandHandler("close_all", self._cmd_close_all))
        # Buy/Sell handlers removed per request
        application.add_handler(CommandHandler("set_risk", self._cmd_set_risk))
        application.add_handler(CommandHandler("set_tp_sl", self._cmd_set_tp_sl))
        application.add_handler(CommandHandler("performance", self._cmd_performance))
        application.add_handler(CommandHandler("history", self._cmd_history))
        # Alerts handlers removed per request
        application.add_handler(CommandHandler("news", self._cmd_news))
        application.add_handler(CommandHandler("sessions", self._cmd_sessions))
        application.add_handler(CommandHandler("switch", self._cmd_switch))
        # Admin commands
        application.add_handler(CommandHandler("db_stats", self._cmd_db_stats))
        application.add_handler(CommandHandler("add_user", self._cmd_add_user))
        application.add_handler(CommandHandler("list_users", self._cmd_list_users))

        # Inline callbacks for show/hide keyboard
        application.add_handler(CallbackQueryHandler(self._on_inline_toggle))

        self.application = application

    async def _post_init(self, app):
        try:
            await app.bot.set_my_commands([
                BotCommand("menu", "Show Keyboard"),
                BotCommand("close", "Hide Keyboard"),
            ])
        except Exception:
            logger.exception("Failed setting bot commands")

    def _run_blocking(self):
        try:
            # Create and set event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            # Build and run polling in this thread
            application = (
                ApplicationBuilder()
                .token(TELEGRAM_BOT_TOKEN)
                .post_init(self._post_init)
                .build()
            )

            # Command handlers
            application.add_handler(CommandHandler("start", self._cmd_start))
            application.add_handler(CommandHandler("stop", self._cmd_stop))
            application.add_handler(CommandHandler("close_all", self._cmd_close_all))
            application.add_handler(CommandHandler("restart", self._cmd_restart))
            application.add_handler(CommandHandler("info", self._cmd_info))
            application.add_handler(CommandHandler("login", self._cmd_login))
            application.add_handler(CommandHandler("logout", self._cmd_logout))
            application.add_handler(CommandHandler("analyze_now", self._cmd_analyze_now))
            application.add_handler(CommandHandler("balance", self._cmd_balance))
            application.add_handler(CommandHandler("account", self._cmd_account))
            application.add_handler(CommandHandler("positions", self._cmd_positions))
            application.add_handler(CommandHandler("orders", self._cmd_orders))
            application.add_handler(CommandHandler("close_all", self._cmd_close_all))
            application.add_handler(CommandHandler("menu", self._cmd_menu))
            application.add_handler(CommandHandler("close", self._cmd_close))
            # Buy/Sell handlers removed per request
            application.add_handler(CommandHandler("set_risk", self._cmd_set_risk))
            application.add_handler(CommandHandler("set_tp_sl", self._cmd_set_tp_sl))
            application.add_handler(CommandHandler("performance", self._cmd_performance))
            application.add_handler(CommandHandler("history", self._cmd_history))
            # Alerts handlers removed per request
            application.add_handler(CommandHandler("news", self._cmd_news))
            application.add_handler(CommandHandler("sessions", self._cmd_sessions))
            # Admin commands
            application.add_handler(CommandHandler("db_stats", self._cmd_db_stats))
            application.add_handler(CommandHandler("add_user", self._cmd_add_user))
            application.add_handler(CommandHandler("list_users", self._cmd_list_users))
            application.add_handler(CallbackQueryHandler(self._on_inline_toggle))
            # Text handler for reply keyboard buttons
            application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text))

            self.application = application
            application.run_polling(allowed_updates=None, stop_signals=None)
        except Exception:
            logger.exception("Telegram bot polling crashed")

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_blocking, name="TelegramBot", daemon=True)
        self._thread.start()

    def stop(self):
        try:
            if self.application:
                # Graceful stop; run_polling will exit. Ensure coroutine is awaited from thread loop.
                if self._loop and self._loop.is_running():
                    fut = asyncio.run_coroutine_threadsafe(self.application.stop(), self._loop)
                    fut.result(timeout=5)
                else:
                    # Fallback: call stop synchronously if loop not available
                    try:
                        self.application.stop_running()
                    except Exception:
                        pass
        except Exception:
            logger.exception("Error stopping Telegram application")

    # External notification helpers
    def notify(self, chat_id: int, text: str):
        try:
            if not self.application or not self._loop:
                return
            fut = asyncio.run_coroutine_threadsafe(
                self.application.bot.send_message(chat_id=chat_id, text=text, disable_web_page_preview=True),
                self._loop,
            )
            try:
                # Allow more time for Telegram API under load
                result = fut.result(timeout=15)
                return result
            except Exception as e:
                # On timeout or cancellation, attempt to cancel and log at warning level
                try:
                    fut.cancel()
                except Exception:
                    pass
                logger.warning(f"Telegram notify timed out/cancelled for chat {chat_id}: {e}")
                return None
        except Exception:
            logger.exception("Failed to send Telegram notification")
            return None

    def edit_message(self, chat_id: int, message_id: int, text: str):
        """Edit an existing message"""
        try:
            if not self.application or not self._loop:
                return
            fut = asyncio.run_coroutine_threadsafe(
                self.application.bot.edit_message_text(
                    chat_id=chat_id, 
                    message_id=message_id, 
                    text=text
                ),
                self._loop,
            )
            try:
                result = fut.result(timeout=10)
                return result
            except Exception as e:
                try:
                    fut.cancel()
                except Exception:
                    pass
                logger.warning(f"Telegram edit_message timed out/cancelled for chat {chat_id}: {e}")
                return None
        except Exception:
            logger.exception(f"Failed to edit message {message_id} in chat {chat_id}")
            return None

    def delete_message(self, chat_id: int, message_id: int):
        """Delete a message"""
        try:
            if not self.application or not self._loop:
                return
            fut = asyncio.run_coroutine_threadsafe(
                self.application.bot.delete_message(
                    chat_id=chat_id, 
                    message_id=message_id
                ),
                self._loop,
            )
            try:
                result = fut.result(timeout=10)
                return result
            except Exception as e:
                try:
                    fut.cancel()
                except Exception:
                    pass
                logger.warning(f"Telegram delete_message timed out/cancelled for chat {chat_id}: {e}")
                return None
        except Exception:
            logger.exception(f"Failed to delete message {message_id} in chat {chat_id}")
            return None

    def notify_all(self, text: str):
        try:
            chat_ids = list(self._sessions.keys())
            for cid in chat_ids:
                self.notify(cid, text)
        except Exception:
            logger.exception("Failed broadcasting Telegram notifications")


