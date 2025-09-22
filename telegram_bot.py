"""
Telegram Bot integration for Price Action Trading Bot
Implements show/hide keyboard and command handlers for account, positions,
orders, trading actions, performance, history, alerts, and news.
"""

import asyncio
import threading
import logging
import time
from typing import Optional, List, Dict
from queue import Queue, Empty

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
    AIORateLimiter,
)
from telegram.error import RetryAfter, TelegramError

logger = logging.getLogger(__name__)


def _build_main_reply_keyboard(news_count: int = 0, is_admin: bool = False) -> ReplyKeyboardMarkup:
    news_label = "📰 News" if news_count <= 0 else f"📰 News ({news_count})"
    keyboard_layout = [
        ["ℹ️ Info", "👤 Account"],
        ["📊 Positions", "📋 Orders"],
        ["🟢 Start Trade", "🔴 End Trade"],
        ["📈 Performance", "🧾 History"],
        ["⚠️ Close Reasons","🧠 Analyze Now"],
        [news_label],
    ]
    
    # Add admin-only buttons if user is admin
    if is_admin:
        keyboard_layout.extend([
            ["👑 Admin Panel"],
            ["➕ Add User", "📋 List Users"],
            ["🗑️ Delete User", "📊 DB Stats"],
            ["🖥️ Terminals", "🔄 Sessions"],
            ["🤖 AI Status"],
            ["🚀 AI Train", "📈 AI Performance"],
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
            ["🗑️ Delete User", "📊 DB Stats"],
            ["🖥️ Terminals", "🔄 Sessions"]
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
        # Message queue for rate limiting
        self._message_queue: Queue = Queue()
        self._queue_worker_running = False
        # Initialize terminal manager
        self._init_terminal_manager()

    def _init_terminal_manager(self):
        """Initialize automatic terminal manager from database"""
        try:
            from auto_terminal_manager import auto_terminal_manager
            
            # Initialize auto terminal manager (loads from database)
            if auto_terminal_manager.initialize():
                logger.info("Auto Terminal Manager initialized successfully")
            else:
                logger.warning("Failed to initialize Auto Terminal Manager, using shared terminal mode")
                
        except ImportError:
            logger.warning("Auto Terminal Manager not available, using shared terminal mode")
        except Exception as e:
            logger.error(f"Error initializing auto terminal manager: {e}")

    def _start_queue_worker(self):
        """Start the message queue worker thread"""
        if not self._queue_worker_running:
            self._queue_worker_running = True
            worker_thread = threading.Thread(target=self._queue_worker, daemon=True)
            worker_thread.start()
            logger.info("Message queue worker started")

    def _queue_worker(self):
        """Worker thread that processes queued messages with rate limiting"""
        while self._queue_worker_running:
            try:
                # Get message from queue with timeout
                message_data = self._message_queue.get(timeout=1.0)
                
                if message_data is None:  # Shutdown signal
                    break
                    
                chat_id, text, message_type = message_data
                
                if message_type == 'send':
                    self._send_message_direct(chat_id, text)
                elif message_type == 'update':
                    message_id = message_data[3]
                    self._update_message_direct(chat_id, message_id, text)
                    
                # Longer delay to ensure we don't overwhelm Telegram
                time.sleep(0.5)  # Half second between messages
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in queue worker: {e}")

    async def _error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Global error handler for rate limiting and other errors"""
        try:
            error = context.error
            if isinstance(error, RetryAfter):
                # Rate limit hit - log with descriptive message per user preference
                logger.warning(f"Rate limit exceeded (retry in {error.retry_after} seconds)")
                # Don't try to send a message as that would likely hit the same rate limit
                return
            elif isinstance(error, TelegramError):
                # Other Telegram API errors - log descriptively
                logger.warning(f"Telegram API error ({error.message})")
                return
            else:
                # Other errors
                logger.error(f"Unhandled error in telegram bot: {error}")
        except Exception as e:
            logger.error(f"Error in error handler: {e}")

    async def safe_reply_text(self, update: Update, text: str, **kwargs):
        """Rate-limited reply_text wrapper"""
        try:
            chat_id = update.effective_chat.id
            # For simple text messages without special formatting, use queue
            if not kwargs or (len(kwargs) == 1 and 'disable_web_page_preview' in kwargs):
                self._message_queue.put((chat_id, text, 'send'), block=False)
                return
            # For messages with special parameters (reply_markup, parse_mode, etc.), use direct send with error handling
            await self.safe_reply_text(update, text, **kwargs)
        except RetryAfter as retry_error:
            logger.warning(f"Rate limit exceeded (retry in {retry_error.retry_after} seconds)")
        except TelegramError as tg_error:
            logger.warning(f"Telegram API error ({tg_error.message})")
        except Exception as e:
            logger.warning(f"Failed to send reply for chat {update.effective_chat.id}: {e}")
            # Try queuing as fallback for simple messages
            if not kwargs:
                try:
                    self._message_queue.put((chat_id, text, 'send'), block=False)
                except:
                    pass

    async def safe_send_message(self, chat_id: int, text: str, **kwargs):
        """Rate-limited send_message wrapper"""
        try:
            # Queue the message instead of sending directly
            self._message_queue.put((chat_id, text, 'send'), block=False)
        except Exception as e:
            logger.warning(f"Failed to queue message for chat {chat_id}: {e}")
            # Fallback to direct send (will be handled by global error handler if rate limited)
            try:
                if self.application:
                    await self.application.bot.send_message(chat_id=chat_id, text=text, **kwargs)
            except RetryAfter as retry_error:
                logger.warning(f"Rate limit exceeded (retry in {retry_error.retry_after} seconds)")
            except TelegramError as tg_error:
                logger.warning(f"Telegram API error ({tg_error.message})")

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
            await self.safe_reply_text(update, 
                "❌ Access Denied\n\n"
                "You are not authorized to use this bot. Please contact the administrator to get access."
            )
            return False
        return True
    
    async def _check_admin_authorization(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Check admin authorization and send error message if not admin."""
        chat_id = update.effective_chat.id
        
        if not self._is_user_admin(chat_id):
            await self.safe_reply_text(update, 
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
                await self.safe_reply_text(update, "Please enter your Account (login) number:")
                return
            login = int(args[0])
            password = args[1]
            server = args[2]
            
            # Check if user is already logged in with a different account
            existing_session = self._sessions.get(update.effective_chat.id)
            if existing_session and existing_session._login != login:
                await self.safe_reply_text(update, 
                    f"You are already logged in with account {existing_session._login}. "
                    f"Please logout first before switching to account {login}."
                )
                return
            
            # Check if we have a dedicated terminal for this account or user
            terminal_name = None
            try:
                from auto_terminal_manager import auto_terminal_manager
                # First try to find terminal by account number
                terminal_name = auto_terminal_manager.get_terminal_for_account(login)
                logger.info(f"Terminal lookup by account {login}: {terminal_name}")
                
                # If not found by account, try to find by user ID
                if not terminal_name:
                    bot_user = db_manager.get_bot_user_by_telegram_chat_id(update.effective_chat.id)
                    if bot_user:
                        terminal_name = auto_terminal_manager.get_terminal_for_user(bot_user['bot_user_id'])
                        logger.info(f"Terminal lookup by user ID {bot_user['bot_user_id']}: {terminal_name}")
                        
                        # Debug: Check what's in the database
                        account = db_manager.get_mt_account_by_bot_user_id(bot_user['bot_user_id'])
                        if account:
                            logger.info(f"Database account for user {bot_user['bot_user_id']}: {account}")
                        else:
                            logger.info(f"No database account found for user {bot_user['bot_user_id']}")
                
                # If still not found, use the database terminal name as fallback
                if not terminal_name:
                    if bot_user:
                        account = db_manager.get_mt_account_by_bot_user_id(bot_user['bot_user_id'])
                        if account and account.get('terminal_name'):
                            terminal_name = account['terminal_name']
                            logger.info(f"Using database terminal name as fallback: {terminal_name}")
                        else:
                            # Generate expected terminal name as last resort
                            expected_terminal_name = f"tmn_{update.effective_chat.id}"
                            logger.info(f"No terminal found, expected terminal name: {expected_terminal_name}")
                            terminal_name = expected_terminal_name
            except ImportError:
                logger.warning("Auto Terminal Manager not available")
            except Exception as e:
                logger.error(f"Error during terminal lookup: {e}")
            
            # Ensure terminal is configured and path is correct BEFORE creating connector
            try:
                bot_user = db_manager.get_bot_user_by_telegram_chat_id(update.effective_chat.id)
                if bot_user:
                    # Persist MT account and terminal name first
                    db_manager.add_mt_account(
                        bot_user['bot_user_id'],
                        login,
                        update.effective_chat.id
                    )
                    from auto_terminal_manager import auto_terminal_manager
                    # Create or update terminal configuration
                    auto_terminal_manager.create_terminal_for_user(
                        bot_user['bot_user_id'],
                        login
                    )
                    # If terminal already exists, ensure it uses the latest resolved MT5 path and login
                    try:
                        if terminal_name:
                            from terminal_manager import terminal_manager
                            if terminal_name in terminal_manager.terminals:
                                cfg = terminal_manager.terminals[terminal_name]
                                # Update path and login just in case
                                new_path = auto_terminal_manager._get_mt5_path()
                                if cfg.terminal_path != new_path or cfg.login != login:
                                    cfg.terminal_path = new_path
                                    cfg.login = login
                                    logger.info(f"Updated terminal config for {terminal_name}: path={new_path}, login={login}")
                    except Exception as e:
                        logger.warning(f"Could not ensure terminal config for {terminal_name}: {e}")
                    # Start terminal proactively when we know the name
                    if terminal_name:
                        try:
                            auto_terminal_manager.terminal_manager.start_terminal(terminal_name)
                            logger.info(f"Started terminal {terminal_name} prior to connection")
                        except Exception as e:
                            logger.warning(f"Could not start terminal {terminal_name}: {e}")
            except Exception as e:
                logger.warning(f"Pre-connection terminal setup failed: {e}")

            # Create and connect new session using direct connection
            await self.safe_reply_text(update, f"🔗 Connecting directly to MT5...")
            session = MT5Connector(
                    login=login, 
                    password=password, 
                    server=server,
                direct_connection=True
            )
            logger.info(f"Created MT5Connector with direct_connection=True for account {login}")
            
            if not session.connect():
                # Provide specific MT5 error if available
                try:
                    msg = session.get_last_error_message()
                except Exception:
                    msg = "MT5 login failed. Check credentials/server."
                await self.safe_reply_text(update, f"❌ {msg}")
                return
            
            # Store session (DB already updated before connect)
            self._sessions[update.effective_chat.id] = session
            
            # Create/enable per-user trading session so performance and stats work
            try:
                self.controller.start_trading_for_chat(update.effective_chat.id, session)
            except Exception as e:
                logger.warning(f"Failed to start trading session for chat {update.effective_chat.id}: {e}")
            
            # Get bot user info for keyboard and message
            chat_id = update.effective_chat.id
            
            info = session.get_account_info() or {}
            await self.safe_reply_text(update, 
                f"✅ Logged in to account: {info.get('login', login)}\n"
                f"Balance: {info.get('balance', 0):.2f} {info.get('currency', '')}"
            )
            # Show full keyboard after login
            try:
                count = await self._get_upcoming_count()
                is_admin = self._is_user_admin(chat_id)
                await self.safe_reply_text(update, 
                    "Keyboard updated.", reply_markup=_build_main_reply_keyboard(count, is_admin)
                )
            except Exception:
                pass
        except Exception as e:
            await self.safe_reply_text(update, f"Login error: {e}")

    async def _cmd_logout(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        session = self._sessions.get(chat_id)
        if not session:
            await self.safe_reply_text(update, "No session to logout.")
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
        await self.safe_reply_text(update, f"✅ Logged out of MT5 account {account_login} for this chat.")

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Check user authorization first
        if not await self._check_user_authorization(update, context):
            return
            
        await self.safe_reply_text(update, 
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
            await self.safe_reply_text(update, "Please /login first to stop trading.")
            return
        
        # Send initial message
        await self.safe_reply_text(update, f"🛑 Stopping trade and closing all positions...")
        
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
                    await self.safe_reply_text(update, 
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
            await self.safe_reply_text(update, summary_text, parse_mode='Markdown')
            
            # Keep bot running; only trading is disabled
            try:
                self.controller.unsubscribe_telemetry(chat_id)
            except Exception:
                pass

            # Automatically disable strategy monitoring for this chat
            if hasattr(self.controller, '_strategy_monitors'):
                self.controller._strategy_monitors.discard(chat_id)

            # Stop AutoTrainer if running
            try:
                if self.auto_trainer:
                    self.auto_trainer.stop_auto_training()
            except Exception:
                pass
            
        except Exception as e:
            logger.exception("Error during stop trade")
            await self.safe_reply_text(update, f"❌ Error stopping trade: {e}")
        
        # Show keyboard options
        await self.safe_reply_text(update, "Use ▶️ Start Trade to enable trading again.", reply_markup=_build_show_hide_inline())

    async def _cmd_close_all(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Close all positions without stopping the bot"""
        chat_id = update.effective_chat.id
        session = self._get_session(chat_id)
        
        if not session:
            await self.safe_reply_text(update, "Please /login first.")
            return
        
        # Send initial message
        await self.safe_reply_text(update, f"🔄 Closing all positions...")
        
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
            await self.safe_reply_text(update, summary_text, parse_mode='Markdown')
            
        except Exception as e:
            logger.exception("Error during close all positions")
            await self.safe_reply_text(update, f"❌ Error closing positions: {e}")

    async def _cmd_analyze_now(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Force an immediate analysis pass and send snapshot to Telegram."""
        try:
            # Ensure we use this chat's MT5 session for data access
            chat_id = update.effective_chat.id
            session = self._get_session(chat_id)
            if not session:
                await self.safe_reply_text(update, "Please /login first.")
                return
            # Reconnect if the session is not currently connected
            try:
                if not getattr(session, 'connected', False):
                    ok = session.connect()
                    if not ok:
                        await self.safe_reply_text(update, f"❌ Could not connect to MT5. Please /login again.")
                        return
            except Exception:
                await self.safe_reply_text(update, f"❌ MT5 connection error. Please /login again.")
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
                await self.safe_reply_text(update, snap)
            else:
                await self.safe_reply_text(update, "No snapshot available (no data or error).")
        except Exception as e:
            logger.exception("Error triggering analysis")
            await self.safe_reply_text(update, f"❌ Failed to analyze now: {e}")

    async def _cmd_start_trade(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enable auto trading, using this chat's MT5 session if available."""
        # Check user authorization first
        if not await self._check_user_authorization(update, context):
            return
            
        chat_id = update.effective_chat.id
        session = self._get_session(chat_id)
        if not session:
            await self.safe_reply_text(update, "Please /login first with 🔑 Login or /login.")
            return
        try:
            # Enable trading only for this chat
            try:
                logger.info(f"Controller type: {type(self.controller)}")
                logger.info(f"Controller has start_trading_for_chat: {hasattr(self.controller, 'start_trading_for_chat')}")
                logger.info(f"Controller methods: {[m for m in dir(self.controller) if 'trading' in m.lower()]}")
                
                # Try to use per-user session mode first
                try:
                    logger.info(f"Attempting to use per-user session mode for chat {chat_id}")
                    self.controller.start_trading_for_chat(chat_id, session)
                    logger.info(f"Successfully used per-user session mode for chat {chat_id}")
                except AttributeError:
                    logger.info(f"Per-user session mode not available, using legacy global mode for chat {chat_id}")
                    # Legacy fallback: global
                    self.controller.set_mt5_connector(session)
                    self.controller.enable_trading()
            except Exception as e:
                logger.exception("Error enabling trading for chat")
                await self.safe_reply_text(update, f"❌ Failed to enable trading: {e}")
                return
            
            # Subscribe this chat to telemetry
            try:
                self.controller.subscribe_telemetry(chat_id)
            except Exception:
                pass
            
            # Automatically enable strategy monitoring for this chat
            if not hasattr(self.controller, '_strategy_monitors'):
                self.controller._strategy_monitors = set()
            self.controller._strategy_monitors.add(chat_id)
            
            is_admin = self._is_user_admin(chat_id)
            await self.safe_reply_text(update, 
                "✅ Auto trading enabled with strategy monitoring.....",
                reply_markup=_build_main_reply_keyboard(is_admin=is_admin),
            )

            # Start AutoTrainer for continuous retraining if available
            try:
                if AutoTrainer and hasattr(self.controller, 'ai_strategy') and self.controller.ai_strategy:
                    if self.auto_trainer is None:
                        self.auto_trainer = AutoTrainer(self.controller.ai_strategy, session)
                    # Only start if not already running
                    if not self.auto_trainer.is_running:
                        self.auto_trainer.start_auto_training()
            except Exception:
                logger.exception("Failed to start AutoTrainer after enabling trading")
        except Exception as e:
            logger.exception("Error enabling trading")
            await self.safe_reply_text(update, f"❌ Failed to enable trading: {e}")

    async def _cmd_restart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.controller.stop()
        await asyncio.sleep(0.5)
        self.controller.start()
        await self.safe_reply_text(update, "Bot restarted.")

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
        await self.safe_reply_text(update, "Keyboard shown.", reply_markup=kb)

    async def _cmd_close(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.safe_reply_text(update, 
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

            from datetime import datetime as _dt, timedelta as _td
            d1 = (_dt.utcnow() - _td(days=2)).strftime('%Y-%m-%d')
            d2 = (_dt.utcnow() + _td(days=7)).strftime('%Y-%m-%d')
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
                    return _dt.fromisoformat(dt_str.replace('Z', ''))
                except Exception:
                    return None
            now = _dt.utcnow()
            return sum(1 for e in data if (parse_iso(e.get('Date') or e.get('DateUtc') or '') or now) >= now)
        except Exception:
            return 0

    async def _cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        session = self._get_session(update.effective_chat.id)
        if not session:
            await self.safe_reply_text(update, "Please /login first.")
            return
        info = session.get_account_info()
        if not info:
            await self.safe_reply_text(update, "Unable to fetch account info.")
            return
        msg = (
            f"Balance: {info['balance']:.2f}\n"
            f"Equity: {info['equity']:.2f}\n"
            f"Margin: {info['margin']:.2f}\n"
            f"Free Margin: {info['free_margin']:.2f}"
        )
        await self.safe_reply_text(update, msg)

    async def _cmd_account(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        session = self._get_session(update.effective_chat.id)
        if not session:
            await self.safe_reply_text(update, "Please /login first.")
            return
        info = session.get_account_info()
        if not info:
            await self.safe_reply_text(update, "Unable to fetch account info.")
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
        await self.safe_reply_text(update, msg + msg2)

    async def _cmd_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        session = self._get_session(update.effective_chat.id)
        if not session:
            await self.safe_reply_text(update, "Please /login first.")
            return
        info = session.get_account_info()
        if not info:
            await self.safe_reply_text(update, "Unable to fetch account info.")
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
        await self.safe_reply_text(update, msg)

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        session = self._get_session(update.effective_chat.id)
        if not session:
            await self.safe_reply_text(update, "Please /login first.")
            return
        positions = session.get_positions()
        if not positions:
            await self.safe_reply_text(update, "No open positions.")
            return
        lines: List[str] = []
        for p in positions:
            lines.append(
                f"#{p['ticket']} {p['type'].upper()} {p['symbol']} vol={p['volume']} PnL={p['profit']:.2f}"
            )
        await self.safe_reply_text(update, "\n".join(lines))

    async def _cmd_orders(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        session = self._get_session(update.effective_chat.id)
        if not session:
            await self.safe_reply_text(update, "Please /login first.")
            return
        orders = session.get_orders()
        if not orders:
            await self.safe_reply_text(update, "No pending orders.")
            return
        lines: List[str] = []
        for o in orders:
            lines.append(
                f"#{o['ticket']} {o['symbol']} type={o['type']} vol={o['volume']} price={o['price_open']}"
            )
        await self.safe_reply_text(update, "\n".join(lines))

    async def _cmd_close_all(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        session = self._get_session(update.effective_chat.id)
        if not session:
            await self.safe_reply_text(update, "Please /login first.")
            return
        positions = session.get_positions()
        if not positions:
            await self.safe_reply_text(update, "No open positions to close.")
            return
        closed = 0
        for p in positions:
            try:
                if self.mt5.close_position(p['ticket']):
                    closed += 1
            except Exception:
                logger.exception("Error closing position")
        await self.safe_reply_text(update, f"Closed {closed} positions.")

    async def _cmd_buy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.safe_reply_text(update, "Trading buttons are temporarily disabled.")

    async def _cmd_sell(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.safe_reply_text(update, "Trading buttons are temporarily disabled.")

    async def _place_market(self, update: Update, order_type: str):
        # Best-effort TP/SL using defaults and account-based lot sizing via controller logic
        session = self._get_session(update.effective_chat.id)
        if not session:
            await self.safe_reply_text(update, "Please /login first.")
            return
        prices = session.get_current_price()
        if not prices:
            await self.safe_reply_text(update, "Unable to get current price.")
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
            await self.safe_reply_text(update, "Account info unavailable.")
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
            await self.safe_reply_text(update, 
                f"{order_type.title()} order placed: {sym} vol={volume} | Ticket: {ticket if ticket is not None else 'n/a'}"
            )
        else:
            # Surface last_error info to user for troubleshooting
            try:
                import MetaTrader5 as mt5
                err = mt5.last_error()
                await self.safe_reply_text(update, f"Order failed. MT5: {err}")
            except Exception:
                await self.safe_reply_text(update, "Order failed. Check MT5 and permissions.")

    async def _cmd_set_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Usage: /set_risk 1.5
        try:
            arg = context.args[0]
            value = float(arg)
            if value <= 0 or value > 10:
                raise ValueError
            self.current_risk_percentage = value
            await self.safe_reply_text(update, f"Risk per trade set to {value}%")
        except Exception:
            await self.safe_reply_text(update, "Usage: /set_risk <percent>. Example: /set_risk 1.5")

    async def _cmd_set_tp_sl(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Usage: /set_tp_sl 100 50
        try:
            tp = int(context.args[0])
            sl = int(context.args[1])
            if tp <= 0 or sl <= 0:
                raise ValueError
            self.default_tp_points = tp
            self.default_sl_points = sl
            await self.safe_reply_text(update, 
                f"Defaults set. TP={tp} points, SL={sl} points"
            )
        except Exception:
            await self.safe_reply_text(update, "Usage: /set_tp_sl <tp_points> <sl_points>")

    async def _cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        
        # Get user-specific session stats
        user_session = self.controller._user_sessions.get(chat_id)
        if not user_session:
            await self.safe_reply_text(update, f"❌ No active trading session. Please /login first.")
            return
            
        stats = user_session.get('stats', {})
        if not stats:
            stats = {}
            user_session['stats'] = stats

        # Reconcile stats from MT5 history to capture broker-closed positions (TP/SL) even if missed in-loop
        try:
            session = user_session.get('connector') or self._get_session(chat_id)
            session_start = stats.get('session_start')
            # Normalize session_start to datetime when possible
            from datetime import datetime, timedelta
            start_dt = None
            if session_start and hasattr(session_start, 'strftime'):
                start_dt = session_start
            elif isinstance(session_start, str):
                try:
                    # Best-effort parse common format '%Y-%m-%d %H:%M' used in display
                    start_dt = datetime.strptime(session_start, '%Y-%m-%d %H:%M')
                except Exception:
                    start_dt = None

            deals = []
            if session:
                if start_dt:
                    try:
                        from datetime import datetime as _dt
                        deals = session.get_history_deals(start_dt, _dt.now())
                    except Exception:
                        # Fallback to day if direct range fails
                        deals = session.get_recent_history('day')
                else:
                    deals = session.get_recent_history('day')

            # Compute W/L and PnL from deals
            reconciled_wins = 0
            reconciled_losses = 0
            reconciled_profit = 0.0
            total_deals = 0
            for d in deals or []:
                pnl = float(d.get('profit', 0.0) or 0.0)
                reconciled_profit += pnl
                # Count only closed position deals (all returned here are deals)
                total_deals += 1
                if pnl > 0:
                    reconciled_wins += 1
                elif pnl < 0:
                    reconciled_losses += 1

            # If reconciled totals indicate missing in-memory stats, prefer reconciled values
            if total_deals > 0:
                stats['winning_trades'] = reconciled_wins
                stats['losing_trades'] = reconciled_losses
                # Keep existing total_trades if larger (includes opens), else use deals count
                stats['total_trades'] = max(int(stats.get('total_trades', 0) or 0), total_deals)
                stats['total_profit'] = float(reconciled_profit)
        except Exception:
            # Silent fallback; keep existing stats if reconciliation fails
            pass
            
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
        await self.safe_reply_text(update, msg, parse_mode='Markdown')

    async def _cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Usage: /history [today|day|week|month|last], default today
        period = 'today'
        if context.args:
            arg = context.args[0].lower()
            if arg in ('today', 'day', 'week', 'month', 'last'):
                period = arg
        
        session = self._get_session(update.effective_chat.id)
        if not session:
            await self.safe_reply_text(update, "Please /login first.")
            return
        
        # Get history deals from MT5 only - no fallback to close_reasons
        deals = session.get_recent_history(period)
        
        # Extra safety: strictly filter to today's calendar date when requested
        if period == 'today' and deals:
            try:
                from datetime import datetime as _dt
                today = _dt.now().date()
                logger.info(f"Filtering MT5 deals for today's date: {today}")
                
                filtered_deals = []
                for d in deals:
                    if hasattr(d.get('time'), 'date'):
                        deal_date = d['time'].date()
                        logger.debug(f"MT5 Deal {d.get('ticket', 'unknown')} date: {deal_date}")
                        if deal_date == today:
                            filtered_deals.append(d)
                        else:
                            logger.debug(f"Excluding MT5 deal {d.get('ticket', 'unknown')} - date {deal_date} != today {today}")
                
                logger.info(f"Filtered {len(filtered_deals)} MT5 deals for today out of {len(deals)} total")
                deals = filtered_deals
            except Exception as e:
                logger.exception(f"Error filtering MT5 deals for today: {e}")
                pass
        
        # Only show MT5 deal history - no fallback
        if not deals:
            await self.safe_reply_text(update, "No MT5 trading history found for the selected period.")
            return
        
        # Group by symbol and show totals
        symbol_totals = {}
        lines = []
        # Map position_id -> original side from open deal so close side is correct
        pos_side: Dict = {}
        try:
            for d in deals:
                pid = d.get('position_id')
                if not pid:
                    continue
                if d.get('is_close', False):
                    continue
                t = d.get('type')
                side = None
                try:
                    if t == 0:
                        side = 'BUY'
                    elif t == 1:
                        side = 'SELL'
                except Exception:
                    side = None
                if side:
                    pos_side[pid] = side
        except Exception:
            pass
        
        # Show only close deals (DEAL_ENTRY_OUT) to reflect trade exits
        # Show each close deal individually (match MT5), deriving side from original open when available
        # Order by time (most recent first)
        close_deals_sorted = sorted(
            [x for x in deals if x.get('is_close', False)],
            key=lambda x: x['time'],
            reverse=True  # Most recent first
        )
        
        # Limit to last 20 positions for better readability
        close_deals_sorted = close_deals_sorted[:20]
        
        for d in close_deals_sorted:
            symbol = d['symbol']
            pnl = float(d['profit'] or 0.0)
            if symbol not in symbol_totals:
                symbol_totals[symbol] = {'count': 0, 'total_pnl': 0.0}
            symbol_totals[symbol]['count'] += 1
            symbol_totals[symbol]['total_pnl'] += pnl
            
            dt = d['time'].strftime('%Y-%m-%d %I:%M %p')
            pid = d.get('position_id')
            side = pos_side.get(pid) or ''
            volume = float(d['volume'] or 0.0)
            price = float(d['price'] or 0.0)
            ticket = int(d.get('ticket') or 0)
            
            # Choose emoji based on profit
            if pnl > 0:
                emoji = "💰"
            elif pnl < 0:
                emoji = "📉"
            else:
                emoji = "➖"
            
            # Build MT5-only line (no close_reasons integration)
            line_parts = [
                f"{emoji} **#{ticket}** {symbol}",
                f"📅 {dt}",
                f"📊 {side} {volume:.2f} @ {price:.5f}",
                f"💵 P/L: {pnl:.2f}"
            ]
            
            lines.append("\n".join(line_parts) + "\n")  # Add extra newline after each position
        
        # Add summary at the top for better visibility
        if symbol_totals:
            summary_lines = ["📊 **History Summary**\n"]
            total_trades = sum(totals['count'] for totals in symbol_totals.values())
            total_pnl = sum(totals['total_pnl'] for totals in symbol_totals.values())
            
            summary_lines.append(f"📈 **Total Trades:** {total_trades}")
            summary_lines.append(f"💵 **Total P/L:** {total_pnl:.2f}")
            summary_lines.append("")
            
            for symbol, totals in sorted(symbol_totals.items()):
                summary_lines.append(f"• {symbol}: {totals['count']} trades, P/L: {totals['total_pnl']:.2f}")
            summary_lines.append("")
        
        # Combine and send - add extra spacing between positions
        full_text = "\n".join(summary_lines + lines)
        
        # Telegram message length limit
        if len(full_text) > 4000:
            # Split into chunks
            chunks = []
            current_chunk = []
            current_length = 0
            
            for line in lines:
                line_length = len(line) + 1
                if current_length + line_length > 3500:
                    if current_chunk:
                        chunks.append("\n".join(summary_lines + current_chunk))
                        current_chunk = []
                        current_length = 0
                current_chunk.append(line)
                current_length += line_length
            
            if current_chunk:
                chunks.append("\n".join(summary_lines + current_chunk))
            
            for i, chunk in enumerate(chunks):
                prefix = f"**(Part {i+1}/{len(chunks)})**\n\n" if len(chunks) > 1 else ""
                await self.safe_reply_text(update, prefix + chunk, parse_mode='Markdown')
        else:
            await self.safe_reply_text(update, full_text, parse_mode='Markdown')

    async def _cmd_debug(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Debug command to show session and symbol information"""
        session = self._get_session(update.effective_chat.id)
        if not session:
            await self.safe_reply_text(update, "Please /login first.")
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
        
        await self.safe_reply_text(update, "\n".join(debug_info))

    async def _cmd_terminal_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show terminal status and provide troubleshooting options"""
        chat_id = update.effective_chat.id
        
        try:
            # Check if auto terminal manager is available
            try:
                from auto_terminal_manager import auto_terminal_manager
                terminal_manager = auto_terminal_manager.terminal_manager
                
                status_lines = ["🖥️ **Terminal Status Report**\n"]
                
                # Get user's terminal name
                session = self._get_session(chat_id)
                terminal_name = None
                if session:
                    terminal_name = getattr(session, '_terminal_name', None)
                
                if terminal_name:
                    status_lines.append(f"**Your Terminal:** {terminal_name}")
                    
                    # Check terminal status
                    if terminal_name in terminal_manager.terminal_status:
                        status = terminal_manager.terminal_status[terminal_name]
                        status_emoji = {
                            'running': '🟢',
                            'starting': '🟡', 
                            'stopped': '🔴',
                            'failed': '❌',
                            'crashed': '💥',
                            'configured': '⚙️'
                        }.get(status['status'], '❓')
                        
                        status_lines.extend([
                            f"**Status:** {status_emoji} {status['status'].upper()}",
                            f"**Process ID:** {status.get('process_id', 'None')}",
                            f"**Last Check:** {status.get('last_check', 'Unknown')}",
                            f"**Error Count:** {status.get('error_count', 0)}",
                            f"**Account Connected:** {'✅' if status.get('account_connected', False) else '❌'}"
                        ])
                        
                        # Check if process is actually running
                        if terminal_name in terminal_manager.processes:
                            process = terminal_manager.processes[terminal_name]
                            if process.poll() is None:
                                status_lines.append("**Process Status:** ✅ Running")
                            else:
                                status_lines.append(f"**Process Status:** ❌ Terminated (exit code: {process.returncode})")
                        else:
                            status_lines.append("**Process Status:** ❌ No process found")
                    else:
                        status_lines.append("**Status:** ❌ Terminal not found in manager")
                        
                    # Show troubleshooting options
                    status_lines.extend([
                        "\n**🔧 Troubleshooting Options:**",
                        "• Try `/restart_terminal` to restart your terminal",
                        "• Try `/kill_mt5` to kill all MT5 processes and restart",
                        "• Check if MT5 is running manually on your computer",
                        "• Verify your MT5 path in config.py"
                    ])
                else:
                    status_lines.append("❌ No terminal assigned to your session")
                    status_lines.append("Try logging in again with `/login`")
                
                # Show all terminals for admin users
                if self._is_user_admin(chat_id):
                    status_lines.extend([
                        "\n**📋 All Terminals:**"
                    ])
                    for name, status in terminal_manager.terminal_status.items():
                        emoji = {
                            'running': '🟢',
                            'starting': '🟡', 
                            'stopped': '🔴',
                            'failed': '❌',
                            'crashed': '💥',
                            'configured': '⚙️'
                        }.get(status['status'], '❓')
                        status_lines.append(f"• {name}: {emoji} {status['status']}")
                
                await self.safe_reply_text(update, "\n".join(status_lines), parse_mode='Markdown')
                
            except ImportError:
                await self.safe_reply_text(update, f"❌ Terminal manager not available. Using shared terminal mode.")
            except Exception as e:
                await self.safe_reply_text(update, f"❌ Error checking terminal status: {e}")
                
        except Exception as e:
            logger.exception("Error in terminal status command")
            await self.safe_reply_text(update, f"❌ Error: {e}")

    async def _cmd_restart_terminal(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Restart the user's terminal to fix connection issues"""
        chat_id = update.effective_chat.id
        
        try:
            # Check if auto terminal manager is available
            try:
                from auto_terminal_manager import auto_terminal_manager
                terminal_manager = auto_terminal_manager.terminal_manager
                
                # Get user's terminal name
                session = self._get_session(chat_id)
                terminal_name = None
                if session:
                    terminal_name = getattr(session, '_terminal_name', None)
                
                if not terminal_name:
                    await self.safe_reply_text(update, f"❌ No terminal assigned to your session. Try logging in again with `/login`")
                    return
                
                await self.safe_reply_text(update, f"🔄 Restarting terminal: {terminal_name}")
                
                # Stop the terminal first
                if terminal_name in terminal_manager.processes:
                    try:
                        process = terminal_manager.processes[terminal_name]
                        process.terminate()
                        await self.safe_reply_text(update, f"🛑 Stopping terminal process...")
                        time.sleep(3)  # Wait for process to terminate
                    except Exception as e:
                        await self.safe_reply_text(update, f"⚠️ Warning: Could not stop process gracefully: {e}")
                
                # Update status
                if terminal_name in terminal_manager.terminal_status:
                    terminal_manager.terminal_status[terminal_name]['status'] = 'stopped'
                    terminal_manager.terminal_status[terminal_name]['process_id'] = None
                
                # Remove from processes dict
                if terminal_name in terminal_manager.processes:
                    del terminal_manager.processes[terminal_name]
                
                await self.safe_reply_text(update, f"🔄 Starting terminal...")
                
                # Start the terminal again
                success = terminal_manager.start_terminal(terminal_name)
                
                if success:
                    await self.safe_reply_text(update, f"✅ Terminal {terminal_name} restarted successfully!")
                    await self.safe_reply_text(update, "Try logging in again with `/login` to test the connection.")
                else:
                    await self.safe_reply_text(update, f"❌ Failed to restart terminal {terminal_name}")
                    await self.safe_reply_text(update, "Try `/kill_mt5` to kill all MT5 processes and restart manually.")
                
            except ImportError:
                await self.safe_reply_text(update, f"❌ Terminal manager not available. Using shared terminal mode.")
            except Exception as e:
                await self.safe_reply_text(update, f"❌ Error restarting terminal: {e}")
                
        except Exception as e:
            logger.exception("Error in restart terminal command")
            await self.safe_reply_text(update, f"❌ Error: {e}")

    async def _cmd_test_connection(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Test MT5 connection and data fetching capabilities"""
        session = self._get_session(update.effective_chat.id)
        if not session:
            await self.safe_reply_text(update, f"❌ No active session. Please /login first.")
            return
        
        await self.safe_reply_text(update, "🧪 **Testing MT5 Connection...**\n", parse_mode='Markdown')
        
        try:
            # Test 1: Basic connection
            if not session.connected:
                await self.safe_reply_text(update, f"❌ Not connected to MT5")
                return
            
            await self.safe_reply_text(update, f"✅ Connected to MT5")
            
            # Test 2: Account info
            account_info = session.get_account_info()
            if account_info:
                await self.safe_reply_text(update, f"✅ Account: {account_info['login']} ({account_info['server']})")
            else:
                await self.safe_reply_text(update, f"❌ Failed to get account info")
            
            # Test 3: Symbol availability
            symbols_to_test = ["XAUUSD", "XAUUSD+", "EURUSD", "GBPUSD", "USDJPY"]
            available_symbols = []
            
            for symbol in symbols_to_test:
                symbol_info = session.get_symbol_info(symbol)
                if symbol_info:
                    available_symbols.append(symbol)
            
            if available_symbols:
                await self.safe_reply_text(update, f"✅ Available symbols: {', '.join(available_symbols)}")
            else:
                await self.safe_reply_text(update, f"❌ No symbols available")
            
            # Test 4: Data fetching for different symbols and timeframes
            timeframes_to_test = ["M1", "M5", "M15", "M30", "H1"]
            
            for symbol in available_symbols[:2]:  # Test first 2 available symbols
                await self.safe_reply_text(update, f"\n📊 **Testing {symbol}:**")
                
                for tf in timeframes_to_test:
                    try:
                        rates = session.get_rates(symbol, tf, 10)
                        if rates is not None and len(rates) > 0:
                            await self.safe_reply_text(update, f"✅ {tf}: {len(rates)} bars")
                        else:
                            await self.safe_reply_text(update, f"❌ {tf}: Failed")
                    except Exception as e:
                        await self.safe_reply_text(update, f"❌ {tf}: Error - {str(e)}")
            
            # Test 5: Positions
            positions = session.get_positions()
            if positions is not None:
                await self.safe_reply_text(update, f"\n📈 **Positions:** {len(positions)} open")
            else:
                await self.safe_reply_text(update, "\n📈 **Positions:** Failed to retrieve")
            
            await self.safe_reply_text(update, "\n✅ **Connection test completed!**")
            
        except Exception as e:
            logger.exception("Error in connection test")
            await self.safe_reply_text(update, f"❌ Test failed: {e}")

    async def _cmd_available_symbols(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show available symbols for the current account"""
        session = self._get_session(update.effective_chat.id)
        if not session:
            await self.safe_reply_text(update, f"❌ No active session. Please /login first.")
            return
        
        await self.safe_reply_text(update, "📋 **Available Symbols for Your Account**\n", parse_mode='Markdown')
        
        try:
            if not session.connected:
                await self.safe_reply_text(update, f"❌ Not connected to MT5")
                return
            
            # Get account info
            account_info = session.get_account_info()
            if account_info:
                await self.safe_reply_text(update, f"🏦 **Account:** {account_info['login']} ({account_info['server']})\n")
            
            # Test common symbols and show which ones are available
            common_symbols = [
                "XAUUSD", "XAUUSD+", "GOLD", "XAUUSD.m",
                "EURUSD", "EURUSD+", "EURUSD.m", 
                "GBPUSD", "GBPUSD+", "GBPUSD.m",
                "USDJPY", "USDJPY+", "USDJPY.m",
                "AUDUSD", "AUDUSD+", "AUDUSD.m",
                "USDCAD", "USDCAD+", "USDCAD.m",
                "NZDUSD", "NZDUSD+", "NZDUSD.m",
                "USDCHF", "USDCHF+", "USDCHF.m",
                "EURJPY", "EURJPY+", "EURJPY.m",
                "GBPJPY", "GBPJPY+", "GBPJPY.m",
                "AUDJPY", "AUDJPY+", "AUDJPY.m",
                "XAGUSD", "XAGUSD+", "SILVER", "XAGUSD.m"
            ]
            
            available_symbols = []
            unavailable_symbols = []
            
            for symbol in common_symbols:
                symbol_info = session.get_symbol_info(symbol)
                if symbol_info:
                    available_symbols.append(symbol)
                else:
                    unavailable_symbols.append(symbol)
            
            if available_symbols:
                await self.safe_reply_text(update, f"✅ **Available Symbols:**")
                
                # Group by base symbol
                symbol_groups = {}
                for symbol in available_symbols:
                    base = symbol.replace('+', '').replace('.m', '').replace('#', '').replace('_', '')
                    if base not in symbol_groups:
                        symbol_groups[base] = []
                    symbol_groups[base].append(symbol)
                
                # Show grouped symbols
                for base, variants in symbol_groups.items():
                    if len(variants) > 1:
                        await self.safe_reply_text(update, f"• **{base}:** {', '.join(variants)}")
                    else:
                        await self.safe_reply_text(update, f"• {variants[0]}")
                
                # Show total count
                await self.safe_reply_text(update, f"\n📊 **Total Available:** {len(available_symbols)} symbols")
                
                # Test data fetching for gold symbols
                gold_symbols = [s for s in available_symbols if 'XAU' in s or 'GOLD' in s]
                if gold_symbols:
                    await self.safe_reply_text(update, f"\n🥇 **Testing Gold Symbols:**")
                    for symbol in gold_symbols[:3]:  # Test first 3 gold symbols
                        try:
                            rates = session.get_rates(symbol, "M5", 5)
                            if rates is not None and len(rates) > 0:
                                await self.safe_reply_text(update, f"✅ {symbol}: M5 data available ({len(rates)} bars)")
                            else:
                                await self.safe_reply_text(update, f"⚠️ {symbol}: M5 data not available")
                        except Exception as e:
                            await self.safe_reply_text(update, f"❌ {symbol}: Error - {str(e)}")
            else:
                await self.safe_reply_text(update, f"❌ No common symbols found")
            
            # Show symbol detection example
            if available_symbols:
                test_symbol = "XAUUSD" if "XAUUSD" not in available_symbols else "XAUUSD+"
                detected = session.find_available_symbol(test_symbol)
                if detected:
                    await self.safe_reply_text(update, f"\n🔍 **Symbol Detection Test:**")
                    await self.safe_reply_text(update, f"Looking for: {test_symbol}")
                    await self.safe_reply_text(update, f"Found: {detected}")
            
        except Exception as e:
            logger.exception("Error in available symbols command")
            await self.safe_reply_text(update, f"❌ Error: {e}")

    async def _cmd_debug_timezone(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Debug timezone issues with trade history"""
        session = self._get_session(update.effective_chat.id)
        if not session:
            await self.safe_reply_text(update, f"❌ No active session. Please /login first.")
            return
        
        await self.safe_reply_text(update, "🕐 **Timezone Debug Information**\n", parse_mode='Markdown')
        
        try:
            from datetime import datetime as _dt
            import time
            
            # Show current time information
            now_local = _dt.now()
            now_utc = _dt.utcnow()
            today_local = now_local.date()
            
            await self.safe_reply_text(update, 
                f"**Current Time:**\n"
                f"• Local: {now_local.strftime('%Y-%m-%d %I:%M %p')}\n"
                f"• UTC: {now_utc.strftime('%Y-%m-%d %I:%M %p')}\n"
                f"• Today's date: {today_local}\n"
                f"• Timezone offset: {time.timezone / 3600} hours"
            )
            
            # Get recent deals and show their timestamps
            deals = session.get_recent_history('day')
            if deals:
                await self.safe_reply_text(update, f"\n**Recent Deals (last 24h):** {len(deals)} found")
                
                # Show first few deals with detailed timestamp info
                for i, deal in enumerate(deals[:5]):
                    deal_time = deal.get('time')
                    if deal_time:
                        deal_date = deal_time.date()
                        is_today = deal_date == today_local
                        
                        await self.safe_reply_text(update, 
                            f"**Deal #{deal.get('ticket', 'unknown')}:**\n"
                            f"• Time: {deal_time.strftime('%Y-%m-%d %I:%M %p')}\n"
                            f"• Date: {deal_date}\n"
                            f"• Is Today: {'✅' if is_today else '❌'}\n"
                            f"• Timestamp: {deal_time.timestamp()}"
                        )
                
                # Count deals by date
                date_counts = {}
                for deal in deals:
                    deal_time = deal.get('time')
                    if deal_time:
                        deal_date = deal_time.date()
                        date_counts[deal_date] = date_counts.get(deal_date, 0) + 1
                
                await self.safe_reply_text(update, "\n**Deals by Date:**")
                for date, count in sorted(date_counts.items()):
                    is_today = date == today_local
                    await self.safe_reply_text(update, f"• {date}: {count} deals {'(Today)' if is_today else ''}")
            
            else:
                await self.safe_reply_text(update, f"❌ No deals found in last 24 hours")
            
        except Exception as e:
            logger.exception("Error in timezone debug")
            await self.safe_reply_text(update, f"❌ Error: {e}")

    async def _cmd_force_cleanup_mt5(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Force cleanup all MT5 connections and processes (Admin only)"""
        chat_id = update.effective_chat.id
        
        # Check if user is admin
        if not self._is_admin(chat_id):
            await self.safe_reply_text(update, f"❌ Admin access required")
            return
        
        try:
            await self.safe_reply_text(update, f"🔄 Starting force cleanup of all MT5 connections...\n\n🔄 Closing all open positions...\n🔄 Force closing all MT5 terminals...")
            
            # Import and call the force cleanup function
            from mt5_connector import force_cleanup_all_mt5
            force_cleanup_all_mt5()
            
            await self.safe_reply_text(update, f"✅ Force cleanup completed!\n\n🔄 All open positions closed\n🔄 All MT5 API connections disconnected\n🔄 All MT5 terminal processes force closed\n🔄 All processes cleaned up")
            
        except Exception as e:
            await self.safe_reply_text(update, f"❌ Error during force cleanup: {str(e)}")
            logger.error(f"Error in force cleanup command: {e}")

    async def _cmd_alerts_on(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.alerts_enabled = True
        await self.safe_reply_text(update, "Alerts enabled.")

    async def _cmd_alerts_off(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.alerts_enabled = False
        await self.safe_reply_text(update, "Alerts disabled.")

    async def _cmd_alerts_toggle(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.alerts_enabled = not self.alerts_enabled
        state = "enabled" if self.alerts_enabled else "disabled"
        await self.safe_reply_text(update, f"Alerts {state}.")

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
                    await self.safe_reply_text(update, "NEWS_API_KEY not configured.")
                    return
                url = (
                    f"https://newsapi.org/v2/top-headlines?country={country}"
                    f"&category={category}&pageSize=5&apiKey={NEWS_API_KEY}"
                )
                resp = requests.get(url, timeout=10)
                if resp.status_code != 200:
                    await self.safe_reply_text(update, "Failed to fetch headlines.")
                    return
                data = resp.json()
                articles = data.get('articles', [])[:5]
                if not articles:
                    await self.safe_reply_text(update, "No headlines available.")
                    return
                lines = []
                for a in articles:
                    title = a.get('title', 'Untitled')
                    source = (a.get('source') or {}).get('name', '')
                    lines.append(f"- {title} ({source})")
                await self.safe_reply_text(update, "\n".join(lines))
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
                await self.safe_reply_text(update, "Failed to fetch calendar. Check TE_API_CLIENT.")
                return
            data = resp.json()
            if not data:
                await self.safe_reply_text(update, "No upcoming events.")
                return
            from datetime import datetime as _dt

            def format_dt(dt_str: str) -> str:
                try:
                    # Expecting ISO-like strings from TE
                    dt = _dt.fromisoformat(dt_str.replace('Z', ''))
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
                    return _dt.fromisoformat(dt_str.replace('Z', ''))
                except Exception:
                    return None

            now = _dt.utcnow()
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

            await self.safe_reply_text(update, "\n\n".join(blocks))

            # Update keyboard with badge
            try:
                is_admin = self._is_user_admin(update.effective_chat.id)
                await self.safe_reply_text(update, 
                    "Menu updated.", reply_markup=_build_main_reply_keyboard(len(upcoming), is_admin)
                )
            except Exception:
                pass
        except Exception:
            logger.exception("Error fetching news/calendar")
            await self.safe_reply_text(update, "Error fetching news/calendar.")

    async def _cmd_sessions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show all active MT5 sessions"""
        try:
            if not self._sessions:
                await self.safe_reply_text(update, "No active MT5 sessions.")
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
                    terminal_info = session.get_terminal_info()
                    
                    # Debug terminal info
                    logger.info(f"Session terminal_info for chat {chat_id}: {terminal_info}")
                    logger.info(f"Session _dedicated_terminal: {session._dedicated_terminal}")
                    logger.info(f"Session _terminal_name: {session._terminal_name}")
                    
                    if info:
                        account_display = f"{info['login']} ({info['balance']:.2f} {info['currency']})"
                    else:
                        account_display = f"{session._login} (info unavailable)"
                    
                    # Add terminal type information
                    if terminal_info['type'] == 'dedicated':
                        terminal_display = f" [Terminal: {terminal_info['terminal_name']}]"
                    else:
                        terminal_display = " [Shared Terminal]"
                    
                    account_display += terminal_display
                    
                except Exception:
                    account_display = f"{session._login} (error)"
                
                message_lines.append(f"Chat {chat_id}: {account_display} - {status}")
            
            if current_account:
                message_lines.append(f"\n🌐 Currently connected to: {current_account}")
            
            message = "\n".join(message_lines)
            await self.safe_reply_text(update, message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in sessions command: {e}")
            await self.safe_reply_text(update, f"❌ Error getting sessions: {e}")

    async def _cmd_terminals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show terminal management information (Admin only)"""
        try:
            if not await self._check_admin_authorization(update, context):
                return
            
            # Get all users from database
            users = db_manager.get_all_bot_users()
            if not users:
                await self.safe_reply_text(update, "No users found in database.")
                return
            
            message_lines = ["🖥️ <b>Terminal Management</b> (Admin Only)\n"]
            
            # Show users and their terminal status
            message_lines.append("👥 <b>Users and Terminals:</b>\n")
            
            for user in users:
                telegram_chat_id = user['telegram_chat_id']
                is_admin = user['is_admin']
                role_emoji = "👑" if is_admin else "👤"
                role_text = "Admin" if is_admin else "User"
                
                # Get MT account for this user
                account = db_manager.get_mt_account_by_bot_user_id(user['bot_user_id'])
                
                # Generate expected terminal name: tmn_ + telegram_chat_id
                expected_terminal_name = f"tmn_{telegram_chat_id}"
                
                # Check if terminal exists in terminal manager and get current login info
                terminal_status = "❓"
                current_login = None
                active_session_login = None
                
                # Check for active session first (real-time info)
                if telegram_chat_id in self._sessions:
                    session = self._sessions[telegram_chat_id]
                    if session.is_active_connection():
                        try:
                            info = session.get_account_info()
                            if info:
                                active_session_login = info['login']
                        except:
                            active_session_login = session._login
                
                # Check terminal manager status
                try:
                    from auto_terminal_manager import auto_terminal_manager
                    terminal_info = auto_terminal_manager.terminal_manager.get_terminal_status(expected_terminal_name)
                    if 'status' in terminal_info:
                        status = terminal_info['status']['status']
                        terminal_status = {
                            'running': '🟢',
                            'stopped': '🔴',
                            'starting': '🟡',
                            'failed': '❌',
                            'crashed': '💥',
                            'configured': '⚪'
                        }.get(status, '❓')
                        
                        # Get current login from terminal config if available
                        if 'config' in terminal_info and terminal_info['config']:
                            current_login = terminal_info['config'].login
                    else:
                        # If terminal not found in manager but user is active, show as running
                        if active_session_login:
                            terminal_status = '🟢'
                except:
                    # If terminal manager error but user is active, show as running
                    if active_session_login:
                        terminal_status = '🟢'
                    else:
                        terminal_status = "❓"
                
                # Determine what MT account info to show (prioritize active session)
                if active_session_login:
                    # Show active session login (most current info)
                    message_lines.append(f"{role_emoji} <b>{telegram_chat_id}</b> ({role_text})")
                    message_lines.append(f"   MT Account: {active_session_login} (🟢 Active)")
                    message_lines.append(f"   Terminal: {expected_terminal_name} {terminal_status}")
                elif account:
                    # Show stored MT account from database
                    mt_account = account['mt_account_number']
                    message_lines.append(f"{role_emoji} <b>{telegram_chat_id}</b> ({role_text})")
                    message_lines.append(f"   MT Account: {mt_account}")
                    message_lines.append(f"   Terminal: {expected_terminal_name} {terminal_status}")
                elif current_login and current_login != 0:
                    # Show current login from terminal (user logged in but not stored in DB yet)
                    message_lines.append(f"{role_emoji} <b>{telegram_chat_id}</b> ({role_text})")
                    message_lines.append(f"   MT Account: {current_login} (Terminal)")
                    message_lines.append(f"   Terminal: {expected_terminal_name} {terminal_status}")
                else:
                    # No MT account info available
                    message_lines.append(f"{role_emoji} <b>{telegram_chat_id}</b> ({role_text})")
                    message_lines.append(f"   MT Account: None")
                    message_lines.append(f"   Terminal: {expected_terminal_name} {terminal_status}")
                message_lines.append("")
            
            # Show terminal manager status if available
            try:
                from auto_terminal_manager import auto_terminal_manager
                status = auto_terminal_manager.get_all_terminals_status()
                
                if 'terminals' in status and status['terminals']:
                    message_lines.append("🖥️ <b>Terminal Manager Status:</b>\n")
                    message_lines.append(f"Total Terminals: {status['total_terminals']}")
                    message_lines.append(f"Running Terminals: {status['running_terminals']}\n")
                    
                    for name, info in status['terminals'].items():
                        config = info['config']
                        terminal_status = info['status']
                        
                        status_emoji = {
                            'running': '🟢',
                            'stopped': '🔴',
                            'starting': '🟡',
                            'failed': '❌',
                            'crashed': '💥',
                            'configured': '⚪'
                        }.get(terminal_status['status'], '❓')
                        
                        message_lines.append(f"{status_emoji} <b>{name}</b>")
                        message_lines.append(f"   Account: {config.login}")
                        message_lines.append(f"   Status: {terminal_status['status']}")
                        if terminal_status.get('process_id'):
                            message_lines.append(f"   PID: {terminal_status['process_id']}")
                        message_lines.append("")
            except ImportError:
                message_lines.append("⚠️ Terminal manager not available")
            except Exception as e:
                message_lines.append(f"⚠️ Terminal manager error: {e}")
            
            message = "\n".join(message_lines)
            await self.safe_reply_text(update, message, parse_mode='HTML')
                
        except Exception as e:
            logger.error(f"Error in terminals command: {e}")
            await self.safe_reply_text(update, f"❌ Error getting terminal info: {e}")

    async def _cmd_terminal_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start a specific terminal (Admin only)"""
        try:
            if not await self._check_admin_authorization(update, context):
                return
            
            args = context.args
            if not args:
                await self.safe_reply_text(update, "Usage: /terminal_start <terminal_name>")
                return
            
            terminal_name = args[0]
            
            try:
                from auto_terminal_manager import auto_terminal_manager
                
                if auto_terminal_manager.terminal_manager.start_terminal(terminal_name):
                    await self.safe_reply_text(update, f"✅ Terminal '{terminal_name}' started successfully")
                else:
                    await self.safe_reply_text(update, f"❌ Failed to start terminal '{terminal_name}'")
                    
            except ImportError:
                await self.safe_reply_text(update, f"❌ Auto Terminal Manager not available")
                
        except Exception as e:
            logger.error(f"Error in terminal_start command: {e}")
            await self.safe_reply_text(update, f"❌ Error starting terminal: {e}")

    async def _cmd_terminal_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Stop a specific terminal (Admin only)"""
        try:
            if not await self._check_admin_authorization(update, context):
                return
            
            args = context.args
            if not args:
                await self.safe_reply_text(update, "Usage: /terminal_stop <terminal_name>")
                return
            
            terminal_name = args[0]
            
            try:
                from auto_terminal_manager import auto_terminal_manager
                
                if auto_terminal_manager.terminal_manager.stop_terminal(terminal_name):
                    await self.safe_reply_text(update, f"✅ Terminal '{terminal_name}' stopped successfully")
                else:
                    await self.safe_reply_text(update, f"❌ Failed to stop terminal '{terminal_name}'")
                    
            except ImportError:
                await self.safe_reply_text(update, f"❌ Auto Terminal Manager not available")
                
        except Exception as e:
            logger.error(f"Error in terminal_stop command: {e}")
            await self.safe_reply_text(update, f"❌ Error stopping terminal: {e}")

    async def _cmd_terminal_restart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Restart a specific terminal (Admin only)"""
        try:
            if not await self._check_admin_authorization(update, context):
                return
            
            args = context.args
            if not args:
                await self.safe_reply_text(update, "Usage: /terminal_restart <terminal_name>")
                return
            
            terminal_name = args[0]
            
            try:
                from auto_terminal_manager import auto_terminal_manager
                
                if auto_terminal_manager.terminal_manager.restart_terminal(terminal_name):
                    await self.safe_reply_text(update, f"✅ Terminal '{terminal_name}' restarted successfully")
                else:
                    await self.safe_reply_text(update, f"❌ Failed to restart terminal '{terminal_name}'")
                    
            except ImportError:
                await self.safe_reply_text(update, f"❌ Auto Terminal Manager not available")
                
        except Exception as e:
            logger.error(f"Error in terminal_restart command: {e}")
            await self.safe_reply_text(update, f"❌ Error restarting terminal: {e}")

    async def _cmd_terminals_refresh(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Refresh terminals from database and apply path overrides (Admin only)"""
        try:
            if not await self._check_admin_authorization(update, context):
                return
            from auto_terminal_manager import auto_terminal_manager
            ok = auto_terminal_manager.refresh_terminals()
            if ok:
                await self.safe_reply_text(update, f"✅ Terminals refreshed from database.")
            else:
                await self.safe_reply_text(update, f"❌ Failed to refresh terminals. Check logs.")
        except ImportError:
            await self.safe_reply_text(update, f"❌ Auto Terminal Manager not available")
        except Exception as e:
            logger.error(f"Error in terminals_refresh command: {e}")
            await self.safe_reply_text(update, f"❌ Error refreshing terminals: {e}")

    async def _cmd_switch(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Switch to your MT5 account (force connection)"""
        try:
            session = self._get_session(update.effective_chat.id)
            if not session:
                await self.safe_reply_text(update, "Please /login first.")
                return
            
            # Force connection to this user's account
            if session.ensure_connection():
                info = session.get_account_info()
                if info:
                    await self.safe_reply_text(update, 
                        f"✅ Switched to account: {info['login']}\n"
                        f"Balance: {info['balance']:.2f} {info['currency']}"
                    )
                else:
                    await self.safe_reply_text(update, f"✅ Switched to account: {session._login}")
            else:
                await self.safe_reply_text(update, f"❌ Failed to switch to your account. Please try /login again.")
                
        except Exception as e:
            logger.error(f"Error in switch command: {e}")
            await self.safe_reply_text(update, f"❌ Error switching account: {e}")

    async def _cmd_ai_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show AI strategy status"""
        try:
            # Check if AI is available
            if not hasattr(self.controller, 'ai_strategy') or self.controller.ai_strategy is None:
                await self.safe_reply_text(update, f"❌ AI Strategy not available. Please ensure AI components are properly installed.")
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
            await self.safe_reply_text(update, message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Error in AI status command: {e}")
            await self.safe_reply_text(update, f"❌ Error getting AI status: {e}")

    async def _cmd_ai_train(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Train AI models"""
        try:
            # Check if AI is available
            if not hasattr(self.controller, 'ai_strategy') or self.controller.ai_strategy is None:
                await self.safe_reply_text(update, f"❌ AI Strategy not available. Please ensure AI components are properly installed.")
                return
            
            # Get data periods parameter
            data_periods = 2000
            if context.args and len(context.args) > 0:
                try:
                    data_periods = int(context.args[0])
                    data_periods = max(500, min(data_periods, 10000))  # Limit between 500-10000
                except ValueError:
                    await self.safe_reply_text(update, f"❌ Invalid data periods. Using default 2000.")
            
            await self.safe_reply_text(update, f"🤖 Starting AI training with {data_periods} data points...")
            
            # Get historical data and train
            session = self._get_session(update.effective_chat.id)
            if not session:
                await self.safe_reply_text(update, "Please /login first.")
                return
            
            # Send progress update
            progress_msg = await self.safe_reply_text(update, "📊 Fetching historical data...")
            
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
                await self.safe_reply_text(update, f"❌ Failed to get historical data for training")
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
                    await self.safe_reply_text(update, f"❌ Historical data missing required OHLC columns")
                    return
            else:
                await self.safe_reply_text(update, f"❌ No historical data available for training")
                return
            
            # Update progress
            await progress_msg.edit_text("🧠 Training AI models... (This may take a few minutes)")
            
            # Train models
            result = self.controller.ai_strategy.train_models(combined_data)
            
            if 'error' in result:
                await self.safe_reply_text(update, f"❌ Training failed: {result['error']}")
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
                    from datetime import datetime as _dt
                    timestamp = _dt.now().strftime('%Y%m%d_%H%M%S')
                    safe_tag = sym_tag.replace('/', '').replace(' ', '')
                    csv_path = os.path.join('ai/data', f'train_{safe_tag}_{timestamp}.csv')
                    combined_data.to_csv(csv_path, index=False)
                    await self.safe_reply_text(update, "💾 Training dataset saved successfully")
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
                await self.safe_reply_text(update, message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Error in AI train command: {e}")
            await self.safe_reply_text(update, f"❌ Training error: {e}")

    async def _cmd_ai_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show AI performance report"""
        try:
            # Check if AI is available
            if not hasattr(self.controller, 'ai_strategy') or self.controller.ai_strategy is None:
                await self.safe_reply_text(update, f"❌ AI Strategy not available. Please ensure AI components are properly installed.")
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
            await self.safe_reply_text(update, message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in AI performance command: {e}")
            await self.safe_reply_text(update, f"❌ Error getting performance: {e}")
    
    async def _cmd_close_reasons(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show close reasons statistics"""
        try:
            if not self.controller or not hasattr(self.controller, 'get_close_reasons_stats'):
                await self.safe_reply_text(update, f"❌ Close reasons tracking not available")
                return
            
            stats = self.controller.get_close_reasons_stats()
            
            if stats['total_closes'] == 0:
                await self.safe_reply_text(update, "📊 **Close Reasons Report**\n\nNo positions closed yet.")
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
            await self.safe_reply_text(update, message, parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Error in close reasons command: {e}")
            await self.safe_reply_text(update, f"❌ Error retrieving close reasons data")

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
            await self.safe_reply_text(update, message)
            
        except Exception as e:
            logger.error(f"Error in database stats command: {e}")
            await self.safe_reply_text(update, f"❌ Error getting database stats: {e}")

    async def _cmd_add_user(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Add a new bot user (admin command)"""
        try:
            # Check admin authorization first
            if not await self._check_admin_authorization(update, context):
                return
            if not context.args:
                await self.safe_reply_text(update, "Usage: /add_user <telegram_chat_id> [admin]")
                return
            
            try:
                telegram_chat_id = int(context.args[0])
            except ValueError:
                await self.safe_reply_text(update, f"❌ Invalid telegram_chat_id. Must be a number.")
                return
            
            # Check if admin flag is provided
            is_admin = len(context.args) > 1 and context.args[1].lower() in ['admin', 'true', '1', 'yes']
            
            # Add user to database
            bot_user_id = db_manager.add_bot_user(telegram_chat_id, is_admin)
            
            if bot_user_id:
                admin_status = "Admin" if is_admin else "Regular User"
                await self.safe_reply_text(update, 
                    f"✅ Added bot user successfully!\n"
                    f"Bot User ID: {bot_user_id}\n"
                    f"Telegram Chat ID: {telegram_chat_id}\n"
                    f"Role: {admin_status}"
                )
            else:
                await self.safe_reply_text(update, f"❌ Failed to add bot user. User might already exist.")
                
        except Exception as e:
            logger.error(f"Error in add user command: {e}")
            await self.safe_reply_text(update, f"❌ Error adding user: {e}")

    async def _cmd_create_terminal(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Create terminal for existing user (admin command)"""
        try:
            # Check admin authorization first
            if not await self._check_admin_authorization(update, context):
                return
            if len(context.args) < 2:
                await self.safe_reply_text(update, "Usage: /create_terminal <telegram_chat_id> <mt_account_number>")
                return
            
            try:
                telegram_chat_id = int(context.args[0])
                mt_account_number = int(context.args[1])
            except ValueError:
                await self.safe_reply_text(update, f"❌ Invalid arguments. Both telegram_chat_id and mt_account_number must be numbers.")
                return
            
            # Check if user exists in database
            user = db_manager.get_bot_user_by_telegram_chat_id(telegram_chat_id)
            if not user:
                await self.safe_reply_text(update, f"❌ User with Telegram Chat ID {telegram_chat_id} not found in database!")
                return
            
            # Check if user already has an MT account
            existing_account = db_manager.get_mt_account_by_bot_user_id(user['bot_user_id'])
            if existing_account:
                terminal_name = existing_account.get('terminal_name', 'Not set')
                await self.safe_reply_text(update, 
                    f"⚠️ User already has an MT account:\n"
                    f"Account: {existing_account['mt_account_number']}\n"
                    f"Terminal: {terminal_name}\n\n"
                    f"Use /update_terminal to change the account number."
                )
                return
            
            # Create terminal for the user
            try:
                from auto_terminal_manager import auto_terminal_manager
                
                if auto_terminal_manager.create_terminal_for_user(user['bot_user_id'], mt_account_number):
                    await self.safe_reply_text(update, 
                        f"✅ Terminal created successfully!\n"
                        f"User: {telegram_chat_id}\n"
                        f"Account: {mt_account_number}\n"
                        f"Terminal: user_{mt_account_number}\n\n"
                        f"User can now login with:\n"
                        f"/login {mt_account_number} <password> <server>"
                    )
                else:
                    await self.safe_reply_text(update, f"❌ Failed to create terminal for user.")
                    
            except ImportError:
                await self.safe_reply_text(update, f"❌ Auto Terminal Manager not available.")
                
        except Exception as e:
            logger.error(f"Error in create terminal command: {e}")
            await self.safe_reply_text(update, f"❌ Error creating terminal: {e}")

    async def _cmd_delete_user(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Delete a specific user (admin command)"""
        try:
            # Check admin authorization first
            if not await self._check_admin_authorization(update, context):
                return
            
            if not context.args:
                await self.safe_reply_text(update, "Usage: /delete_user <telegram_chat_id>")
                return
            
            try:
                telegram_chat_id = int(context.args[0])
            except ValueError:
                await self.safe_reply_text(update, f"❌ Invalid telegram_chat_id. Must be a number.")
                return
            
            # Check if user exists
            user = db_manager.get_bot_user_by_telegram_chat_id(telegram_chat_id)
            if not user:
                await self.safe_reply_text(update, f"❌ User with Telegram Chat ID {telegram_chat_id} not found in database!")
                return
            
            # Check if user is admin
            if user['is_admin']:
                await self.safe_reply_text(update, 
                    f"⚠️ WARNING: User {telegram_chat_id} is an ADMIN!\n"
                    f"❌ Admin users cannot be deleted for security reasons.\n"
                    f"💡 Only regular users can be deleted."
                )
                return
            
            # Get user's MT account info
            account = db_manager.get_mt_account_by_bot_user_id(user['bot_user_id'])
            
            # Show user info and ask for confirmation
            user_info = (
                f"📋 User to be deleted:\n"
                f"• Bot User ID: {user['bot_user_id']}\n"
                f"• Telegram Chat ID: {user['telegram_chat_id']}\n"
                f"• Role: Regular User\n"
            )
            
            if account:
                user_info += f"• MT Account: {account['mt_account_number']}\n"
                terminal_name = account.get('terminal_name', 'Not set')
                user_info += f"• Terminal: {terminal_name}\n"
            else:
                user_info += f"• MT Account: None\n"
            
            user_info += f"\n⚠️ Type 'DELETE USER {telegram_chat_id}' to confirm deletion:"
            
            await self.safe_reply_text(update, user_info)
            
            # Store the expected confirmation in context
            context.user_data['expected_confirmation'] = f"DELETE USER {telegram_chat_id}"
            context.user_data['user_to_delete'] = user
                
        except Exception as e:
            logger.error(f"Error in delete user command: {e}")
            await self.safe_reply_text(update, f"❌ Error: {e}")

    async def _cmd_list_users(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List all bot users (admin command)"""
        try:
            # Check admin authorization first
            if not await self._check_admin_authorization(update, context):
                return
            users = db_manager.get_all_bot_users()
            
            if not users:
                await self.safe_reply_text(update, "📋 Bot Users\n\nNo users found.")
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
                await self.safe_reply_text(update, first_part)
                
                # Send remaining parts
                remaining_lines = message_lines[10:]
                for i in range(0, len(remaining_lines), 10):
                    chunk = "\n".join(remaining_lines[i:i+10])
                    await self.safe_reply_text(update, chunk)
            else:
                await self.safe_reply_text(update, message)
                
        except Exception as e:
            logger.error(f"Error in list users command: {e}")
            await self.safe_reply_text(update, f"❌ Error listing users: {e}")

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
            await self.safe_reply_text(update, message)
            
        except Exception as e:
            logger.error(f"Error in admin panel command: {e}")
            await self.safe_reply_text(update, f"❌ Error showing admin panel: {e}")

    async def _cmd_add_user_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Interactive add user via button (admin command)"""
        try:
            # Check admin authorization first
            if not await self._check_admin_authorization(update, context):
                return
            
            await self.safe_reply_text(update, 
                "👑 Add New User\n\n"
                "Please send the Telegram Chat ID of the user you want to add.\n\n"
                "Format: Just send the number (e.g., 123456789)\n\n"
                "To add as admin: Send 'admin' after the chat ID (e.g., 123456789 admin)\n\n"
                "A terminal will be automatically created with name: tmn_[chat_id]"
            )
            
            # Set state for interactive user addition
            chat_id = update.effective_chat.id
            self._login_states[chat_id] = {"stage": "add_user"}
            
        except Exception as e:
            logger.error(f"Error in add user button command: {e}")
            await self.safe_reply_text(update, f"❌ Error: {e}")

    async def _cmd_delete_user_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Interactive delete user via button (admin command)"""
        try:
            # Check admin authorization first
            if not await self._check_admin_authorization(update, context):
                return
            
            # Get all users to show available options
            users = db_manager.get_all_bot_users()
            if not users:
                await self.safe_reply_text(update, "No users found in database.")
                return
            
            # Separate admin and regular users
            admin_users = [user for user in users if user['is_admin']]
            regular_users = [user for user in users if not user['is_admin']]
            
            if not regular_users:
                await self.safe_reply_text(update, f"✅ No regular users to delete. All users are admins.")
                return
            
            # Show available users for deletion
            user_list = "🗑️ Delete User\n\n"
            user_list += f"📋 Available users for deletion ({len(regular_users)}):\n\n"
            
            for i, user in enumerate(regular_users[:10], 1):  # Show first 10 users
                account = db_manager.get_mt_account_by_bot_user_id(user['bot_user_id'])
                if account:
                    user_list += f"{i}. User {user['telegram_chat_id']} (MT: {account['mt_account_number']})\n"
                else:
                    user_list += f"{i}. User {user['telegram_chat_id']} (No MT account)\n"
            
            if len(regular_users) > 10:
                user_list += f"... and {len(regular_users) - 10} more users\n"
            
            user_list += f"\n⚠️ Admin users ({len(admin_users)}) cannot be deleted for security reasons.\n\n"
            user_list += "Please send the Telegram Chat ID of the user you want to delete.\n"
            user_list += "Format: Just send the number (e.g., 987654321)"
            
            await self.safe_reply_text(update, user_list)
            
            # Set state for interactive user deletion
            chat_id = update.effective_chat.id
            self._login_states[chat_id] = {"stage": "delete_user"}
            
        except Exception as e:
            logger.error(f"Error in delete user button command: {e}")
            await self.safe_reply_text(update, f"❌ Error: {e}")

    async def _on_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text:
            return
        text = update.message.text.strip().lower()
        
        # Handle delete confirmation FIRST (before stage handling)
        if context.user_data.get('expected_confirmation'):
            expected = context.user_data['expected_confirmation']
            if update.message.text.strip() == expected:
                # Confirmation received, proceed with deletion
                
                # Check if it's a single user deletion
                if 'user_to_delete' in context.user_data:
                    # Single user deletion
                    user = context.user_data['user_to_delete']
                    try:
                        # Remove MT account first (if exists)
                        account = db_manager.get_mt_account_by_bot_user_id(user['bot_user_id'])
                        if account:
                            db_manager.remove_mt_account(user['bot_user_id'])

                        # Remove user
                        if db_manager.remove_bot_user(user['telegram_chat_id']):
                            await self.safe_reply_text(update, 
                                f"✅ User deletion complete!\n"
                                f"🗑️ Deleted user: {user['telegram_chat_id']}\n"
                                f"✅ User and associated data removed"
                            )
                        else:
                            await self.safe_reply_text(update, f"❌ Failed to delete user.")
                    except Exception as e:
                        logger.error(f"Error deleting user {user['telegram_chat_id']}: {e}")
                        await self.safe_reply_text(update, f"❌ Error deleting user: {e}")
                
                # Clear context and login state
                context.user_data.pop('expected_confirmation', None)
                context.user_data.pop('user_to_delete', None)
                self._login_states.pop(chat_id, None)
                
            else:
                await self.safe_reply_text(update, f"❌ Confirmation text doesn't match. Deletion cancelled.")
                context.user_data.pop('expected_confirmation', None)
                context.user_data.pop('user_to_delete', None)
                self._login_states.pop(chat_id, None)
            return
        
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
                        await self.safe_reply_text(update, f"❌ Please provide a Telegram Chat ID.")
                        return
                    
                    telegram_chat_id = int(parts[0])
                    is_admin = len(parts) > 1 and parts[1].lower() in ['admin', 'true', '1', 'yes']
                    
                    # Add user to database
                    bot_user_id = db_manager.add_bot_user(telegram_chat_id, is_admin)
                    
                    if bot_user_id:
                        admin_status = "Admin" if is_admin else "Regular User"
                        
                        # Generate terminal name: tmn_ + telegram_chat_id
                        terminal_name = f"tmn_{telegram_chat_id}"
                        
                        # Create terminal for this user (without MT account)
                        try:
                            from auto_terminal_manager import auto_terminal_manager
                            
                            # Create terminal configuration
                            from terminal_manager import TerminalConfig
                            config = TerminalConfig(
                                name=terminal_name,
                                terminal_path=auto_terminal_manager._get_mt5_path(),
                                login=0,  # Will be set when user logs in
                                password="",  # Will be provided during login
                                server="",    # Will be provided during login
                                symbol="EURUSD",  # Use global config
                                timeframe="M15",  # Use global config
                                auto_start=False  # Don't auto-start without credentials
                            )
                            
                            # Add to terminal manager
                            if auto_terminal_manager.terminal_manager.add_terminal(config):
                                await self.safe_reply_text(update, 
                                    f"✅ User setup complete!\n"
                                    f"Bot User ID: {bot_user_id}\n"
                                    f"Telegram Chat ID: {telegram_chat_id}\n"
                                    f"Terminal: {terminal_name}\n"
                                    f"Role: {admin_status}\n\n"
                                    f"🎉 User can now login and use their dedicated terminal!"
                                )
                            else:
                                await self.safe_reply_text(update, 
                                    f"⚠️ User added but terminal creation failed.\n"
                                    f"Bot User ID: {bot_user_id}\n"
                                    f"Telegram Chat ID: {telegram_chat_id}\n"
                                    f"Terminal: {terminal_name}\n"
                                    f"Role: {admin_status}\n\n"
                                    f"User can still login, but will use shared terminal."
                                )
                        except ImportError:
                            await self.safe_reply_text(update, 
                                f"⚠️ User added but terminal manager not available.\n"
                                f"Bot User ID: {bot_user_id}\n"
                                f"Telegram Chat ID: {telegram_chat_id}\n"
                                f"Terminal: {terminal_name}\n"
                                f"Role: {admin_status}\n\n"
                                f"User can still login, but will use shared terminal."
                            )
                        except Exception as e:
                            await self.safe_reply_text(update, 
                                f"⚠️ User added but terminal creation failed: {e}\n"
                                f"Bot User ID: {bot_user_id}\n"
                                f"Telegram Chat ID: {telegram_chat_id}\n"
                                f"Terminal: {terminal_name}\n"
                                f"Role: {admin_status}\n\n"
                                f"User can still login, but will use shared terminal."
                            )
                    else:
                        await self.safe_reply_text(update, f"❌ Failed to add bot user. User might already exist.")
                    
                except ValueError:
                    await self.safe_reply_text(update, f"❌ Invalid Telegram Chat ID. Must be a number.")
                except Exception as e:
                    await self.safe_reply_text(update, f"❌ Error adding user: {e}")
                finally:
                    self._login_states.pop(chat_id, None)
                return
            elif stage == "delete_user":
                # Handle interactive user deletion
                try:
                    telegram_chat_id = int(update.message.text.strip())
                    
                    # Check if user exists
                    user = db_manager.get_bot_user_by_telegram_chat_id(telegram_chat_id)
                    if not user:
                        await self.safe_reply_text(update, f"❌ User with Telegram Chat ID {telegram_chat_id} not found in database!")
                        self._login_states.pop(chat_id, None)
                        return
                    
                    # Check if user is admin
                    if user['is_admin']:
                        await self.safe_reply_text(update, 
                            f"⚠️ WARNING: User {telegram_chat_id} is an ADMIN!\n"
                            f"❌ Admin users cannot be deleted for security reasons.\n"
                            f"💡 Only regular users can be deleted."
                        )
                        self._login_states.pop(chat_id, None)
                        return
                    
                    # Get user's MT account info
                    account = db_manager.get_mt_account_by_bot_user_id(user['bot_user_id'])
                    
                    # Show user info and ask for confirmation
                    user_info = (
                        f"📋 User to be deleted:\n"
                        f"• Bot User ID: {user['bot_user_id']}\n"
                        f"• Telegram Chat ID: {user['telegram_chat_id']}\n"
                        f"• Role: Regular User\n"
                    )
                    
                    if account:
                        user_info += f"• MT Account: {account['mt_account_number']}\n"
                        terminal_name = account.get('terminal_name', 'Not set')
                        user_info += f"• Terminal: {terminal_name}\n"
                    else:
                        user_info += f"• MT Account: None\n"
                    
                    user_info += f"\n⚠️ Type 'DELETE USER {telegram_chat_id}' to confirm deletion:"
                    
                    await self.safe_reply_text(update, user_info)
                    
                    # Store the expected confirmation in context
                    context.user_data['expected_confirmation'] = f"DELETE USER {telegram_chat_id}"
                    context.user_data['user_to_delete'] = user
                    
                except ValueError:
                    await self.safe_reply_text(update, f"❌ Invalid Telegram Chat ID. Must be a number.")
                    self._login_states.pop(chat_id, None)
                except Exception as e:
                    await self.safe_reply_text(update, f"❌ Error: {e}")
                    self._login_states.pop(chat_id, None)
                return
            elif stage == "account":
                try:
                    state["login"] = int(update.message.text.strip())
                    state["stage"] = "password"
                    await self.safe_reply_text(update, "Enter Password:")
                except Exception:
                    await self.safe_reply_text(update, "Invalid account. Enter numeric account:")
                return
            if stage == "password":
                state["password"] = update.message.text.strip()
                state["stage"] = "server"
                await self.safe_reply_text(update, "Enter Server (e.g., VantageInternational-Demo):")
                return
            if stage == "server":
                state["server"] = update.message.text.strip()
                # Attempt login
                try:
                    # Check if user is already logged in with a different account
                    existing_session = self._sessions.get(chat_id)
                    if existing_session and existing_session._login != state["login"]:
                        await self.safe_reply_text(update, 
                            f"You are already logged in with account {existing_session._login}. "
                            f"Please logout first before switching to account {state['login']}."
                        )
                        self._login_states.pop(chat_id, None)
                        return
                    
                    # Skip terminal management for direct connections
                    terminal_name = None
                    logger.info(f"[Interactive] Using direct connection for account {state['login']}")

                    # Ensure terminal exists and path is correct before connecting
                    try:
                        bot_user = db_manager.get_bot_user_by_telegram_chat_id(chat_id)
                        if bot_user:
                            db_manager.add_mt_account(bot_user['bot_user_id'], state["login"], chat_id)
                            from auto_terminal_manager import auto_terminal_manager
                            auto_terminal_manager.create_terminal_for_user(bot_user['bot_user_id'], state["login"])
                            # If terminal already exists, ensure it uses the latest resolved MT5 path and login
                            try:
                                if terminal_name:
                                    from terminal_manager import terminal_manager
                                    if terminal_name in terminal_manager.terminals:
                                        cfg = terminal_manager.terminals[terminal_name]
                                        new_path = auto_terminal_manager._get_mt5_path()
                                        if cfg.terminal_path != new_path or cfg.login != state["login"]:
                                            cfg.terminal_path = new_path
                                            cfg.login = state["login"]
                                            logger.info(f"[Interactive] Updated terminal config for {terminal_name}: path={new_path}, login={state['login']}")
                            except Exception as e:
                                logger.warning(f"[Interactive] Could not ensure terminal config for {terminal_name}: {e}")
                            if terminal_name:
                                try:
                                    auto_terminal_manager.terminal_manager.start_terminal(terminal_name)
                                    logger.info(f"[Interactive] Started terminal {terminal_name} prior to connection")
                                except Exception as e:
                                    logger.warning(f"[Interactive] Could not start terminal {terminal_name}: {e}")
                    except Exception as e:
                        logger.warning(f"[Interactive] Pre-connection terminal setup failed: {e}")

                    # Create connector with dedicated terminal when available
                    await self.safe_reply_text(update, f"🔗 Connecting directly to MT5...")
                    session = MT5Connector(
                        login=state["login"],
                        password=state["password"],
                        server=state["server"],
                        direct_connection=True
                    )

                    if not session.connect():
                        try:
                            msg = session.get_last_error_message()
                        except Exception:
                            msg = "Login failed. Check credentials/server and try /login again."
                        await self.safe_reply_text(update, f"❌ {msg}")
                    else:
                        self._sessions[chat_id] = session
                        # Create/enable per-user trading session so performance and stats work
                        try:
                            self.controller.start_trading_for_chat(chat_id, session)
                        except Exception as e:
                            logger.warning(f"[Interactive] Failed to start trading session for chat {chat_id}: {e}")
                        
                        info = session.get_account_info() or {}
                        await self.safe_reply_text(update,
                            f"✅ Logged in to account: {info.get('login', state['login'])}\n"
                            f"Balance: {info.get('balance', 0):.2f} {info.get('currency', '')}"
                        )
                        try:
                            count = await self._get_upcoming_count()
                            is_admin = self._is_user_admin(chat_id)
                            await self.safe_reply_text(update,
                                "Keyboard updated.", reply_markup=_build_main_reply_keyboard(count, is_admin)
                            )
                        except Exception:
                            pass
                except Exception as e:
                    await self.safe_reply_text(update, f"Login error: {e}")
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
        elif text in ("terminal status", "🖥️ terminal status"):
            await self._cmd_terminal_status(update, context)
        elif text in ("restart terminal", "🔄 restart terminal"):
            await self._cmd_restart_terminal(update, context)
        elif text in ("test connection", "🧪 test connection"):
            await self._cmd_test_connection(update, context)
        elif text in ("available symbols", "📋 available symbols"):
            await self._cmd_available_symbols(update, context)
        elif text in ("debug timezone", "🕐 debug timezone"):
            await self._cmd_debug_timezone(update, context)
        elif text in ("login", "🔑 login"):
            # kick off interactive login
            self._login_states[chat_id] = {"stage": "account"}
            await self.safe_reply_text(update, "Please enter your Account (login) number:")
        elif text in ("logout", "🚪 logout"):
            await self._cmd_logout(update, context)
        # Admin-only button handlers
        elif text in ("admin panel", "👑 admin panel"):
            await self._cmd_admin_panel(update, context)
        elif text in ("add user", "➕ add user"):
            await self._cmd_add_user_button(update, context)
        elif text in ("list users", "📋 list users"):
            await self._cmd_list_users(update, context)
        elif text in ("db stats", "📊 db stats"):
            await self._cmd_db_stats(update, context)
        elif text in ("delete user", "🗑️ delete user"):
            await self._cmd_delete_user_button(update, context)
        elif text in ("terminals", "🖥️ terminals"):
            await self._cmd_terminals(update, context)
        elif text in ("sessions", "🔄 sessions"):
            await self._cmd_sessions(update, context)
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
        # Terminal management commands (admin only)
        application.add_handler(CommandHandler("terminals", self._cmd_terminals))
        application.add_handler(CommandHandler("terminal_start", self._cmd_terminal_start))
        application.add_handler(CommandHandler("terminal_stop", self._cmd_terminal_stop))
        application.add_handler(CommandHandler("terminal_restart", self._cmd_terminal_restart))
        application.add_handler(CommandHandler("force_cleanup_mt5", self._cmd_force_cleanup_mt5))
        # Admin commands
        application.add_handler(CommandHandler("db_stats", self._cmd_db_stats))
        application.add_handler(CommandHandler("add_user", self._cmd_add_user))
        application.add_handler(CommandHandler("create_terminal", self._cmd_create_terminal))
        application.add_handler(CommandHandler("delete_user", self._cmd_delete_user))
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
            # Configure rate limiter to prevent hitting Telegram's limits
            # Very conservative settings to avoid any rate limit issues
            rate_limiter = AIORateLimiter(
                overall_max_rate=15,  # 15 messages per second overall (very conservative)
                overall_time_period=1.0,
                group_max_rate=10,    # 10 messages per minute for groups (very conservative)
                group_time_period=60.0,
                max_retries=5         # Retry up to 5 times on rate limit
            )
            
            application = (
                ApplicationBuilder()
                .token(TELEGRAM_BOT_TOKEN)
                .rate_limiter(rate_limiter)
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
            # Terminal management commands (admin only)
            application.add_handler(CommandHandler("terminals", self._cmd_terminals))
            application.add_handler(CommandHandler("terminal_start", self._cmd_terminal_start))
            application.add_handler(CommandHandler("terminal_stop", self._cmd_terminal_stop))
            application.add_handler(CommandHandler("terminal_restart", self._cmd_terminal_restart))
            application.add_handler(CommandHandler("terminals_refresh", self._cmd_terminals_refresh))
            # Admin commands
            application.add_handler(CommandHandler("db_stats", self._cmd_db_stats))
            application.add_handler(CommandHandler("add_user", self._cmd_add_user))
            application.add_handler(CommandHandler("create_terminal", self._cmd_create_terminal))
            application.add_handler(CommandHandler("delete_user", self._cmd_delete_user))
            application.add_handler(CommandHandler("list_users", self._cmd_list_users))
            application.add_handler(CallbackQueryHandler(self._on_inline_toggle))
            # Text handler for reply keyboard buttons
            application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text))

            # Add global error handler for rate limiting
            application.add_error_handler(self._error_handler)
            
            self.application = application
            application.run_polling(allowed_updates=None, stop_signals=None)
        except Exception:
            logger.exception("Telegram bot polling crashed")

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        # Start message queue worker
        self._start_queue_worker()
        self._thread = threading.Thread(target=self._run_blocking, name="TelegramBot", daemon=True)
        self._thread.start()

    def stop(self):
        try:
            # Stop message queue worker
            self._queue_worker_running = False
            try:
                self._message_queue.put(None, timeout=1)  # Shutdown signal
            except:
                pass
                
            if self.application:
                # Graceful stop; run_polling will exit. Ensure coroutine is awaited from thread loop.
                if self._loop and self._loop.is_running():
                    try:
                        fut = asyncio.run_coroutine_threadsafe(self.application.stop(), self._loop)
                        fut.result(timeout=5)
                    except Exception as e:
                        # Treat 'Application is not running' as already-stopped; ignore
                        try:
                            msg = str(e).lower()
                        except Exception:
                            msg = ""
                        if "not running" in msg or "application is not running" in msg:
                            logger.debug("Telegram application already stopped; ignoring stop error")
                        else:
                            logger.warning(f"Telegram stop encountered error: {e}")
                else:
                    # Fallback: call stop synchronously if loop not available
                    try:
                        self.application.stop_running()
                    except Exception:
                        pass
        except Exception:
            logger.exception("Error stopping Telegram application")
        
        # Cleanup auto terminal manager
        try:
            from auto_terminal_manager import auto_terminal_manager
            auto_terminal_manager.cleanup()
            logger.info("Auto Terminal Manager cleaned up")
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"Error cleaning up auto terminal manager: {e}")

    # External notification helpers
    def notify(self, chat_id: int, text: str):
        """Queue a message for sending (non-blocking)"""
        try:
            self._message_queue.put((chat_id, text, 'send'), block=False)
            return True
        except Exception as e:
            logger.warning(f"Failed to queue message for chat {chat_id}: {e}")
            # Fallback to direct sending
            return self._send_message_direct(chat_id, text)

    def _send_message_direct(self, chat_id: int, text: str):
        """Send message directly (used by queue worker)"""
        try:
            if not self.application or not self._loop:
                return None
            fut = asyncio.run_coroutine_threadsafe(
                self.application.bot.send_message(chat_id=chat_id, text=text, disable_web_page_preview=True),
                self._loop,
            )
            try:
                # Allow more time for Telegram API under load
                result = fut.result(timeout=15)
                return result.message_id if result else None
            except RetryAfter as e:
                # Rate limit hit - log with descriptive message per user preference
                logger.warning(f"Rate limit exceeded (retry in {e.retry_after} seconds)")
                try:
                    fut.cancel()
                except Exception:
                    pass
                return None
            except TelegramError as e:
                # Other Telegram API errors - log descriptively
                logger.warning(f"Telegram API error ({e.message})")
                try:
                    fut.cancel()
                except Exception:
                    pass
                return None
            except Exception as e:
                # On timeout or cancellation, attempt to cancel and log at warning level
                try:
                    fut.cancel()
                except Exception:
                    pass
                return None
        except Exception:
            logger.exception("Failed to send Telegram notification")
            return None
    
    def update_message(self, chat_id: int, message_id: int, text: str):
        """Queue a message update (non-blocking)"""
        try:
            self._message_queue.put((chat_id, text, 'update', message_id), block=False)
            return True
        except Exception as e:
            logger.warning(f"Failed to queue message update for chat {chat_id}: {e}")
            # Fallback to direct update
            return self._update_message_direct(chat_id, message_id, text)

    def _update_message_direct(self, chat_id: int, message_id: int, text: str):
        """Update an existing message directly (used by queue worker)"""
        try:
            if not self.application or not self._loop:
                logger.warning(f"Cannot update message - application or loop not available for chat {chat_id}")
                return False
            fut = asyncio.run_coroutine_threadsafe(
                self.application.bot.edit_message_text(
                    chat_id=chat_id, 
                    message_id=message_id, 
                    text=text, 
                    disable_web_page_preview=True
                ),
                self._loop,
            )
            try:
                result = fut.result(timeout=15)
                success = result is not None
                logger.debug(f"Message update result for chat {chat_id}, message_id {message_id}: {success}")
                return success
            except RetryAfter as e:
                # Rate limit hit - log with descriptive message per user preference
                logger.warning(f"Rate limit exceeded (retry in {e.retry_after} seconds)")
                try:
                    fut.cancel()
                except Exception:
                    pass
                return False
            except TelegramError as e:
                # Treat "message is not modified" as a successful no-op
                try:
                    msg = str(e).lower()
                except Exception:
                    msg = ""
                if "message is not modified" in msg or "not modified" in msg:
                    logger.debug(f"Message not modified for chat {chat_id}, message_id {message_id}; skipping update")
                    return True
                # Other Telegram API errors - log descriptively
                logger.warning(f"Telegram API error ({e.message})")
                try:
                    fut.cancel()
                except Exception:
                    pass
                return False
            except Exception as e:
                logger.warning(f"Message update failed for chat {chat_id}, message_id {message_id}: {e}")
                # On timeout or cancellation, attempt to cancel and log at warning level
                try:
                    fut.cancel()
                except Exception:
                    pass
                # Fallback: send a new message if the original cannot be edited (400 Bad Request)
                try:
                    if any(x in msg for x in [
                        "message to edit not found",
                        "message can't be edited",
                        "message identifier is not specified",
                        "bad request",
                        "chat not found",
                    ]):
                        logger.info(f"Falling back to sending a new message for chat {chat_id}")
                        self.send_message(chat_id, text)
                        return True
                except Exception:
                    pass
                return False
        except Exception as e:
            logger.error(f"Exception in update_message for chat {chat_id}, message_id {message_id}: {e}")
            return False

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
                # Fallback on 400 Bad Request or non-editable cases: send a new message
                try:
                    msg = str(e).lower()
                except Exception:
                    msg = ""
                logger.warning(f"Telegram edit_message failed for chat {chat_id}: {e}")
                try:
                    if any(x in msg for x in [
                        "message to edit not found",
                        "message can't be edited",
                        "message identifier is not specified",
                        "bad request",
                        "chat not found",
                    ]):
                        logger.info(f"Falling back to sending a new message for chat {chat_id}")
                        self.send_message(chat_id, text)
                        return None
                except Exception:
                    pass
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


