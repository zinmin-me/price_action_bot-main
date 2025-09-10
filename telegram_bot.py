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


def _build_main_reply_keyboard(news_count: int = 0) -> ReplyKeyboardMarkup:
    news_label = "📰 News" if news_count <= 0 else f"📰 News ({news_count})"
    keyboard_layout = [
        ["ℹ️ Info", "👤 Account"],
        ["📊 Positions", "📋 Orders"],
        ["🟢 Buy", "🔴 Sell"],
        ["▶️ Start Trade", "⏹️ End Trade"],
        ["📈 Performance", "🧾 History"],
        ["🔔 Alerts On/Off", news_label],
        ["🧠 Analyze Now"],
    ]
    return ReplyKeyboardMarkup(
        keyboard_layout,
        resize_keyboard=True,
        one_time_keyboard=False,
        is_persistent=True,
    )

def _build_minimal_reply_keyboard() -> ReplyKeyboardMarkup:
    keyboard_layout = [
        ["ℹ️ Info"],
        ["🔑 Login", "👤 Account"],
    ]
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

    def _get_session(self, chat_id: int) -> Optional[MT5Connector]:
        """Return per-chat MT5Connector if exists, else None (force login)."""
        return self._sessions.get(chat_id)

    async def _cmd_login(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/login <login> <password> <server>  OR interactive when no args."""
        try:
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
            # Create and connect new session
            session = MT5Connector(login=login, password=password, server=server,
                                   terminal_path=None)
            if not session.connect():
                await update.message.reply_text("MT5 login failed. Check credentials/server.")
                return
            self._sessions[update.effective_chat.id] = session
            info = session.get_account_info() or {}
            await update.message.reply_text(
                f"Logged in: {info.get('login', login)}"
            )
            # Show full keyboard after login
            try:
                count = await self._get_upcoming_count()
                await update.message.reply_text(
                    "Keyboard updated.", reply_markup=_build_main_reply_keyboard(count)
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
        try:
            session.disconnect()
        except Exception:
            pass
        self._sessions.pop(chat_id, None)
        await update.message.reply_text("Logged out of MT5 for this chat.")

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Welcome to Price Action Bot. Use the buttons or commands.",
            reply_markup=_build_show_hide_inline(),
        )

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Stop trading and close all positions with summary"""
        chat_id = update.effective_chat.id
        session = self._get_session(chat_id)
        
        if not session:
            await update.message.reply_text("Please /login first to stop trading.")
            return
        
        # Send initial message
        await update.message.reply_text("🛑 Stopping trade and closing all positions...")
        
        try:
            # Point controller to this session's connector, then disable trading
            try:
                self.controller.set_mt5_connector(session)
            except Exception:
                pass
            self.controller.disable_trading()

            # First cancel all pending orders
            try:
                cancel_res = session.cancel_all_orders()
                if cancel_res.get('total', 0) > 0:
                    await update.message.reply_text(
                        f"⛔ Cancelled pending orders: {cancel_res.get('success', 0)}/{cancel_res.get('total', 0)}"
                    )
            except Exception:
                pass

            # Close all positions
            close_results = session.close_all_positions()
            
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
            # Close all positions
            close_results = session.close_all_positions()
            
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
            snap = self.controller.generate_analysis_snapshot()
            if snap:
                await update.message.reply_text(snap)
            else:
                await update.message.reply_text("No snapshot available (no data or error).")
        except Exception as e:
            logger.exception("Error triggering analysis")
            await update.message.reply_text(f"❌ Failed to analyze now: {e}")
    async def _cmd_start_trade(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enable auto trading, using this chat's MT5 session if available."""
        chat_id = update.effective_chat.id
        session = self._get_session(chat_id)
        if not session:
            await update.message.reply_text("Please /login first with 🔑 Login or /login.")
            return
        try:
            # Ensure controller uses this user's connector
            self.controller.set_mt5_connector(session)
            # Enable trading
            self.controller.enable_trading()
            # Subscribe this chat to telemetry
            try:
                self.controller.subscribe_telemetry(chat_id)
            except Exception:
                pass
            await update.message.reply_text(
                "✅ Auto trading enabled. Use 🧠 Analyze Now anytime to see the latest analysis.",
                reply_markup=_build_main_reply_keyboard(),
            )
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
            if chat_id in getattr(self, '_sessions', {}):
                count = await self._get_upcoming_count()
                kb = _build_main_reply_keyboard(count)
            else:
                kb = _build_minimal_reply_keyboard()
            await query.message.reply_text("Keyboard shown.", reply_markup=kb)
        elif query.data == "hide_keyboard":
            await query.message.reply_text(
                "Keyboard hidden.", reply_markup=ReplyKeyboardRemove()
            )

    async def _cmd_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        if chat_id in getattr(self, '_sessions', {}):
            count = await self._get_upcoming_count()
            kb = _build_main_reply_keyboard(count)
        else:
            kb = _build_minimal_reply_keyboard()
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
        msg = (
            "Account Information\n"
            "----------------------------\n"
            f"Account: {info['login']}\n"
            f"Leverage: {info['leverage']}\n"
            f"Currency: {info['currency']}\n"
            "\nBalance Information\n"
            "----------------------------\n"
            f"Balance: {info['balance']:.2f}\n"
            f"Equity: {info['equity']:.2f}\n"
            f"Margin: {info['margin']:.2f}\n"
            f"Free Margin: {info['free_margin']:.2f}"
        )
        await update.message.reply_text(msg)

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
        await self._place_market(update, order_type="buy")

    async def _cmd_sell(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self._place_market(update, order_type="sell")

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
        stats = self.controller.stats if hasattr(self.controller, 'stats') else None
        if not stats:
            await update.message.reply_text("No stats available.")
            return
        total = stats.get('total_trades', 0)
        win = stats.get('winning_trades', 0)
        loss = stats.get('losing_trades', 0)
        profit = stats.get('total_profit', 0.0)
        win_rate = (win / max(total, 1)) * 100
        msg = (
            f"Trades: {total} (W:{win} L:{loss})\n"
            f"Win rate: {win_rate:.1f}%\n"
            f"Total P/L: {profit:.2f}"
        )
        await update.message.reply_text(msg)

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
                await update.message.reply_text(
                    "Menu updated.", reply_markup=_build_main_reply_keyboard(len(upcoming))
                )
            except Exception:
                pass
        except Exception:
            logger.exception("Error fetching news/calendar")
            await update.message.reply_text("Error fetching news/calendar.")

    async def _on_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text:
            return
        text = update.message.text.strip().lower()
        # Handle interactive login wizard
        chat_id = update.effective_chat.id
        state = self._login_states.get(chat_id)
        if state:
            stage = state.get("stage")
            if stage == "account":
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
                    session = MT5Connector(login=state["login"], password=state["password"], server=state["server"])
                    if not session.connect():
                        await update.message.reply_text("Login failed. Check credentials/server and try /login again.")
                    else:
                        self._sessions[chat_id] = session
                        info = session.get_account_info() or {}
                        await update.message.reply_text(f"Logged in: {info.get('login', state['login'])}")
                        try:
                            count = await self._get_upcoming_count()
                            await update.message.reply_text(
                                "Keyboard updated.", reply_markup=_build_main_reply_keyboard(count)
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
        elif text in ("buy", "🟢 buy"):
            await self._cmd_buy(update, context)
        elif text in ("sell", "🔴 sell"):
            await self._cmd_sell(update, context)
        elif text in ("start trade", "▶️ start trade"):
            await self._cmd_start_trade(update, context)
        elif text in ("end trade", "⏹️ end trade"):
            await self._cmd_stop(update, context)
        elif text in ("performance", "📈 performance"):
            await self._cmd_performance(update, context)
        elif text in ("history", "🧾 history"):
            await self._cmd_history(update, context)
        elif text in ("alerts on", "🔔 alerts on"):
            await self._cmd_alerts_on(update, context)
        elif text in ("alerts off", "🔔 alerts off"):
            await self._cmd_alerts_off(update, context)
        elif text in ("alerts on/off", "🔔 alerts on/off"):
            await self._cmd_alerts_toggle(update, context)
        elif text in ("news", "📰 news"):
            await self._cmd_news(update, context)
        elif text in ("analyze now", "🔎 analyze now", "🧠 analyze now"):
            await self._cmd_analyze_now(update, context)
        elif text in ("debug", "🐛 debug"):
            await self._cmd_debug(update, context)
        elif text in ("login", "🔑 login"):
            # kick off interactive login
            self._login_states[chat_id] = {"stage": "account"}
            await update.message.reply_text("Please enter your Account (login) number:")
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
        application.add_handler(CommandHandler("buy", self._cmd_buy))
        application.add_handler(CommandHandler("sell", self._cmd_sell))
        application.add_handler(CommandHandler("set_risk", self._cmd_set_risk))
        application.add_handler(CommandHandler("set_tp_sl", self._cmd_set_tp_sl))
        application.add_handler(CommandHandler("performance", self._cmd_performance))
        application.add_handler(CommandHandler("history", self._cmd_history))
        application.add_handler(CommandHandler("alerts_on", self._cmd_alerts_on))
        application.add_handler(CommandHandler("alerts_off", self._cmd_alerts_off))
        application.add_handler(CommandHandler("news", self._cmd_news))

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
            application.add_handler(CommandHandler("buy", self._cmd_buy))
            application.add_handler(CommandHandler("sell", self._cmd_sell))
            application.add_handler(CommandHandler("set_risk", self._cmd_set_risk))
            application.add_handler(CommandHandler("set_tp_sl", self._cmd_set_tp_sl))
            application.add_handler(CommandHandler("performance", self._cmd_performance))
            application.add_handler(CommandHandler("history", self._cmd_history))
            application.add_handler(CommandHandler("alerts_on", self._cmd_alerts_on))
            application.add_handler(CommandHandler("alerts_off", self._cmd_alerts_off))
            application.add_handler(CommandHandler("news", self._cmd_news))
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
                # Graceful stop; run_polling will exit
                self.application.stop()
        except Exception:
            logger.exception("Error stopping Telegram application")

    # External notification helpers
    def notify(self, chat_id: int, text: str):
        try:
            if not self.application or not self._loop:
                return
            fut = asyncio.run_coroutine_threadsafe(
                self.application.bot.send_message(chat_id=chat_id, text=text),
                self._loop,
            )
            fut.result(timeout=5)
        except Exception:
            logger.exception("Failed to send Telegram notification")

    def notify_all(self, text: str):
        try:
            chat_ids = list(self._sessions.keys())
            for cid in chat_ids:
                self.notify(cid, text)
        except Exception:
            logger.exception("Failed broadcasting Telegram notifications")


