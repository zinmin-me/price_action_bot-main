"""
Price Action Trading Bot - Main Entry Point
Orchestrates all trading strategies and manages the trading session
"""

import time
import logging
import schedule
import sys
import os
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import bot components
from config import *

# Setup logging first
from utils import LoggingUtils
LoggingUtils.setup_logging(LOG_FILE, LOG_LEVEL)
logger = logging.getLogger(__name__)

# Try to import MT5 connector with graceful fallback
try:
    from mt5_connector import MT5Connector
    MT5_AVAILABLE = True
except ImportError as e:
    logger.warning(f"MT5 connector not available: {e}")
    MT5_AVAILABLE = False
    MT5Connector = None

from utils import DataValidation, RiskManagement

# Try to import strategies with graceful fallback
try:
    from strategies.trend_following import TrendFollowingStrategy
    from strategies.support_resistance import SupportResistanceStrategy
    from strategies.breakout import BreakoutStrategy
    from strategies.reversal_patterns import ReversalPatternsStrategy
    STRATEGIES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Strategies not available: {e}")
    STRATEGIES_AVAILABLE = False
    TrendFollowingStrategy = None
    SupportResistanceStrategy = None
    BreakoutStrategy = None
    ReversalPatternsStrategy = None

# Try to import telegram bot with graceful fallback
try:
    from telegram_bot import TelegramBot
    TELEGRAM_BOT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Telegram bot not available: {e}")
    TELEGRAM_BOT_AVAILABLE = False
    TelegramBot = None

# Import pandas with graceful fallback
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Pandas not available: {e}")
    PANDAS_AVAILABLE = False
    pd = None

# AI components
try:
    from ai import AIStrategy, AutoTrainer, AITelegramBot
    AI_AVAILABLE = True
except ImportError as e:
    logger.warning(f"AI components not available: {e}")
    AI_AVAILABLE = False
    AIStrategy = None
    AutoTrainer = None
    AITelegramBot = None

# Import Telegram bot components for buttons
try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.ext import CallbackQueryHandler, ContextTypes, CommandHandler, ApplicationBuilder, MessageHandler, filters
    TELEGRAM_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Telegram components not available: {e}")
    TELEGRAM_COMPONENTS_AVAILABLE = False
    InlineKeyboardButton = None
    InlineKeyboardMarkup = None
    Update = None
    CallbackQueryHandler = None
    ContextTypes = None
    CommandHandler = None
    ApplicationBuilder = None
    MessageHandler = None
    filters = None

if TELEGRAM_BOT_AVAILABLE and TELEGRAM_COMPONENTS_AVAILABLE:
    class AITelegramBotWrapper(TelegramBot):
        """
        AI-Enhanced Telegram Bot
        
        This is the AI-enhanced Telegram bot with advanced machine learning capabilities.
        """
    
        def __init__(self, mt5_connector, controller):
            """Initialize AI-enhanced Telegram bot."""
            super().__init__(mt5_connector, controller)
            self.ai_strategy = None
            self.auto_trainer = None
            self.ai_telegram_bot = None
            
            # Initialize AI components if available
            if AI_AVAILABLE:
                try:
                    self.ai_strategy = AIStrategy(mt5_connector)
                    self.auto_trainer = AutoTrainer(self.ai_strategy, mt5_connector)
                    self.ai_telegram_bot = AITelegramBot(self, self.ai_strategy, self.auto_trainer)
                    logger.info("AI components initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize AI components: {e}")
                    self.ai_strategy = None
                    self.auto_trainer = None
                    self.ai_telegram_bot = None
            
            logger.info("AI-Enhanced Telegram Bot initialized")
    
        def setup_commands(self):
            """Setup Telegram bot commands including AI commands."""
            logger.info("Setting up AI-enhanced Telegram bot commands")
            
            # Setup base commands (this will be done in the parent class)
            # We'll add AI commands in the _run_blocking method
    
        def start(self):
            """Start the AI-enhanced Telegram bot."""
            # Start auto-trainer if available
            if self.auto_trainer:
                self.auto_trainer.start_auto_training()
                logger.info("AI auto-trainer started")
            
            # Start parent bot
            super().start()
        
        def stop(self):
            """Stop the AI-enhanced Telegram bot."""
            # Stop auto-trainer if available
            if self.auto_trainer:
                self.auto_trainer.stop_auto_training()
                logger.info("AI auto-trainer stopped")
            
            # Stop parent bot
            super().stop()
        
        def _run_blocking(self):
            """Override _run_blocking to add AI command handlers"""
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
                
                # Add AI command handlers if AI components are available
                if self.ai_telegram_bot:
                    self.ai_telegram_bot.setup_ai_commands(application)
                
                application.add_handler(CallbackQueryHandler(self._on_inline_toggle))
                # Text handler for reply keyboard buttons
                application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text))

                self.application = application
                application.run_polling(allowed_updates=None, stop_signals=None)
            except Exception:
                logger.exception("Telegram bot polling crashed")
    
    # Use AI-enhanced bot as the main bot
    StandardTelegramBot = AITelegramBotWrapper
    
else:
    # Create a dummy StandardTelegramBot class when Telegram is not available
    class StandardTelegramBot:
        def __init__(self, mt5_connector, controller):
            logger.warning("StandardTelegramBot not available - Telegram features disabled")
        
        def start(self):
            logger.warning("Telegram bot not available")
        
        def stop(self):
            pass


class PriceActionTradingBot:
    """Main trading bot class that orchestrates all strategies"""
    
    def __init__(self):
        if MT5_AVAILABLE:
            self.mt5 = MT5Connector()
        else:
            self.mt5 = None
            logger.warning("MT5 not available - bot will run in demo mode")
        
        self.strategies = []
        self.daily_trades = 0
        self.last_trade_time = None
        self.session_start_time = datetime.now()
        self.telegram_bot = None
        # Global trading flag retained for backward compatibility (unused in per-user mode)
        self.trading_enabled = False
        # Telemetry subscribers (chat IDs)
        self._telemetry_chat_ids = set()
        # Per-user trading sessions: chat_id -> state
        # state = {
        #   'connector': MT5Connector,
        #   'strategies': List[strategy],
        #   'trading_enabled': bool,
        #   'daily_trades': int,
        #   'last_trade_time': Optional[datetime],
        #   'stats': dict,
        # }
        self._user_sessions: Dict[int, dict] = {}
        self._init_progress: Dict[int, dict] = {}  # chat_id -> {message_id, start_time, countdown}
        
        # Initialize strategies
        self._initialize_strategies()
        
        # Initialize AI strategy if available
        self.ai_strategy = None
        if AI_AVAILABLE and self.mt5:
            try:
                self.ai_strategy = AIStrategy(self.mt5)
                self.strategies.append(self.ai_strategy)
                logger.info("AI Strategy initialized")
            except Exception as e:
                logger.error(f"Failed to initialize AI strategy: {e}")
        
        # Trading statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'daily_trades': 0,
            'session_start': self.session_start_time
        }
        
        # Close reasons tracking
        self.close_reasons = []
        # Track previous positions to detect auto-closed positions
        self._previous_positions: Dict[int, Dict] = {}  # chat_id -> {ticket: position_data}
        # Track processed close logs to trigger training on new logs
        self._processed_close_logs: set = set()
        
    def _initialize_strategies(self):
        """Initialize all trading strategies"""
        if not MT5_AVAILABLE or self.mt5 is None or not STRATEGIES_AVAILABLE:
            logger.warning("Cannot initialize strategies - MT5 or strategies not available")
            return
        
        if TREND_FOLLOWING_ENABLED and TrendFollowingStrategy:
            self.strategies.append(TrendFollowingStrategy(self.mt5))
            logger.info("Trend Following strategy initialized")
        
        if SUPPORT_RESISTANCE_ENABLED and SupportResistanceStrategy:
            self.strategies.append(SupportResistanceStrategy(self.mt5))
            logger.info("Support & Resistance strategy initialized")
        
        if BREAKOUT_ENABLED and BreakoutStrategy:
            self.strategies.append(BreakoutStrategy(self.mt5))
            logger.info("Breakout strategy initialized")
        
        if REVERSAL_PATTERNS_ENABLED and ReversalPatternsStrategy:
            self.strategies.append(ReversalPatternsStrategy(self.mt5))
            logger.info("Reversal Patterns strategy initialized")
        
        logger.info(f"Initialized {len(self.strategies)} trading strategies")
    
    def start(self):
        """Start the trading bot"""
        logger.info("Starting Price Action Trading Bot...")
        
        # Connect to MT5 only if AUTO_LOGIN enabled and MT5 is available
        if AUTO_LOGIN and MT5_AVAILABLE and self.mt5:
            if not self.mt5.connect():
                logger.error("Failed to connect to MT5. Exiting...")
                return False
        elif not MT5_AVAILABLE:
            logger.warning("MT5 not available - running in demo mode")
        
        # Display account information
        if AUTO_LOGIN and MT5_AVAILABLE and self.mt5:
            account_info = self.mt5.get_account_info()
            if account_info:
                logger.info(f"Account: {account_info['login']}")
                logger.info(f"Balance: {account_info['balance']}")
                logger.info(f"Equity: {account_info['equity']}")
                logger.info(f"Currency: {account_info['currency']}")
        
        # Check trading hours
        if AUTO_LOGIN and MT5_AVAILABLE and self.mt5:
            if not self.mt5.get_trading_hours():
                logger.warning("Outside trading hours. Bot will run but may not trade.")
        
        # Schedule main trading loop
        schedule.every(1).minutes.do(self.trading_loop)
        schedule.every().hour.do(self.hourly_maintenance)
        schedule.every().day.at("00:00").do(self.daily_reset)
        
        logger.info("Trading bot started successfully!")
        logger.info(f"Trading symbol: {SYMBOL}")
        logger.info(f"Timeframe: {TIMEFRAME}")
        logger.info(f"Risk per trade: {RISK_PERCENTAGE}%")
        if not AUTO_LOGIN:
            logger.info("MT5 auto login disabled. Use Telegram /login to connect.")

        # Start Telegram bot if enabled
        if TELEGRAM_ENABLED and TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_AVAILABLE:
            try:
                if not self.telegram_bot:
                    self.telegram_bot = StandardTelegramBot(self.mt5, self)
                self.telegram_bot.start()
                logger.info("Standard Telegram bot started")
            except Exception as e:
                logger.error(f"Failed to start Telegram bot: {e}")
        elif TELEGRAM_ENABLED and not TELEGRAM_BOT_AVAILABLE:
            logger.warning("Telegram bot not available - skipping Telegram features")
        
        # Main loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.stop()
        
        return True
    
    def trading_loop(self):
        """Main trading loop executed every minute. Iterates per active user session."""
        try:
            # If no per-user sessions, fall back to legacy single-session behavior when enabled
            if not self._user_sessions:
                if not self._should_trade():
                    return
                self._trading_loop_for_connector(self.mt5, self.strategies, self)
                return

            # Iterate over active user sessions
            for chat_id, state in list(self._user_sessions.items()):
                try:
                    if not state.get('trading_enabled'):
                        continue
                    connector = state.get('connector')
                    if connector is None:
                        continue
                    strategies = state.get('strategies') or []
                    # Run a trading pass for this user/session
                    self._trading_loop_for_connector(connector, strategies, state)
                except Exception as e:
                    logger.error(f"Error in trading loop for chat {chat_id}: {e}")

        except Exception as e:
            logger.error(f"Error in trading loop: {e}")

    def _check_for_auto_closed_positions(self, connector: 'MT5Connector', state_or_self):
        """Check for positions that were automatically closed by broker (SL/TP hits)"""
        try:
            # Get current chat_id for session tracking
            chat_id = None
            if isinstance(state_or_self, dict):
                # Find chat_id for this connector
                for cid, session_state in self._user_sessions.items():
                    if session_state.get('connector') is connector:
                        chat_id = cid
                        break
            
            if chat_id is None:
                return
            
            # Get current positions
            current_positions = {}
            try:
                positions = connector.get_positions()
                for pos in positions:
                    current_positions[pos['ticket']] = pos
            except Exception:
                return
            
            # Get previous positions for this chat
            previous_positions = self._previous_positions.get(chat_id, {})
            
            # Find positions that were closed automatically
            for ticket, prev_pos in previous_positions.items():
                if ticket not in current_positions:
                    # Position was closed - determine if it was auto-closed
                    profit = prev_pos.get('profit', 0.0)
                    symbol = prev_pos.get('symbol', 'Unknown')
                    pos_type = prev_pos.get('type', 'unknown')
                    volume = prev_pos.get('volume', 0)
                    
                    # Determine close reason based on profit and SL/TP levels
                    close_reason = "Unknown reason"
                    if profit > 0:
                        close_reason = "Take profit hit (broker)"
                    elif profit < 0:
                        close_reason = "Stop loss hit (broker)"
                    else:
                        close_reason = "Position closed (broker)"
                    
                    # Update session statistics
                    if isinstance(state_or_self, dict):
                        session_stats = state_or_self.get('stats', {})
                        session_stats['total_profit'] = session_stats.get('total_profit', 0.0) + profit
                        if profit > 0:
                            session_stats['winning_trades'] = session_stats.get('winning_trades', 0) + 1
                        else:
                            session_stats['losing_trades'] = session_stats.get('losing_trades', 0) + 1
                        state_or_self['stats'] = session_stats
                        
                        # Update loss cooldown if it was a loss
                        if profit < 0:
                            state_or_self['last_loss_time'] = datetime.now()
                    
                    # Send notification
                    if TELEGRAM_ENABLED and self.telegram_bot:
                        try:
                            # Create notification message
                            if "take profit" in close_reason.lower():
                                emoji = "🎯"
                                title = "Take Profit Hit!"
                            elif "stop loss" in close_reason.lower():
                                emoji = "🛑"
                                title = "Stop Loss Hit!"
                            else:
                                emoji = "💰" if profit > 0 else "📉"
                                title = "Position Closed"
                            
                            message_lines = [
                                f"{emoji} **{title}**",
                                f"",
                                f"**Ticket:** #{ticket}",
                                f"**Symbol:** {symbol}",
                                f"**Type:** {pos_type.upper()}",
                                f"**Volume:** {volume}",
                                f"**P/L:** {profit:.2f}",
                                f"**Reason:** {close_reason}",
                                f"**Strategy:** Auto-detected"
                            ]
                            
                            self._notify_subscribers("\n".join(message_lines))
                            logger.info(f"Auto-closed position detected: #{ticket} {symbol} P/L: {profit:.2f} ({close_reason})")
                        except Exception:
                            logger.exception("Failed to notify auto-closed position")
            
            # Update previous positions for next check
            self._previous_positions[chat_id] = current_positions
            
        except Exception as e:
            logger.exception("Error in _check_for_auto_closed_positions")

    def generate_analysis_snapshot(self) -> str:
        """Generate a one-off analysis snapshot for all symbols without placing trades."""
        try:
            from config import SYMBOLS
            lines = []
            now_str = datetime.now().strftime('%H:%M:%S')
            for sym in SYMBOLS:
                # Switch symbol and load data
                try:
                    self.mt5.change_symbol(sym)
                except Exception:
                    pass
                df = self.mt5.get_rates(sym, TIMEFRAME, 200)
                if df is None or not DataValidation.validate_ohlc_data(df):
                    lines.append(f"{sym}: no data")
                    continue
                df = DataValidation.clean_data(df)
                spread = self.mt5.get_spread(sym)
                # Run strategies
                sym_lines = [f"🧠 {sym} {TIMEFRAME} | {now_str} | spread {spread}"]
                best = None
                for strategy in self.strategies:
                    if not strategy.enabled:
                        continue
                    try:
                        analysis = strategy.analyze(df)
                        sig = analysis.get('signal', 'no_signal')
                        conf = analysis.get('confidence', '-')
                        pat = analysis.get('pattern_type', '')
                        sym_lines.append(f"• {strategy.name}: {sig.upper()} (conf {conf}%) {pat}")
                        if sig in ['buy','sell']:
                            score = analysis.get('confidence', 0) or 0
                            if best is None or score > best[0]:
                                best = (score, analysis, strategy)
                    except Exception as e:
                        logger.error(f"Snapshot analysis error in {strategy.name}: {e}")
                if best:
                    score, ana, strat = best
                    sym_lines.append(f"👉 Best: {strat.name} {ana['signal'].upper()} (conf {score}%)")
                lines.append("\n".join(sym_lines))
            return "\n\n".join(lines)
        except Exception as e:
            logger.error(f"Error generating analysis snapshot: {e}")
            return ""
    
    def _should_trade(self) -> bool:
        """Check if bot should trade based on various conditions"""
        # Respect trading toggle
        if not self.trading_enabled:
            return False
        # Weekday filter
        try:
            from config import TRADE_WEEKDAYS_ONLY
        except Exception:
            TRADE_WEEKDAYS_ONLY = True
        if TRADE_WEEKDAYS_ONLY:
            # Monday=0 ... Sunday=6
            if datetime.now().weekday() > 4:
                return False
        # Check trading hours (optional bypass)
        if not TRADE_ANYTIME:
            if not self.mt5.get_trading_hours():
                return False
        
        # Check daily trade limit (disabled when MAX_DAILY_TRADES <= 0)
        if MAX_DAILY_TRADES and MAX_DAILY_TRADES > 0:
            if self.daily_trades >= MAX_DAILY_TRADES:
                logger.info("Daily trade limit reached")
                return False
        
        # Check if enough time has passed since last trade
        if self.last_trade_time:
            time_since_last = datetime.now() - self.last_trade_time
            if time_since_last < timedelta(minutes=5):  # Minimum 5 minutes between trades
                return False
        
        return True

    # --- Telemetry helpers ---
    def subscribe_telemetry(self, chat_id: int):
        self._telemetry_chat_ids.add(chat_id)
        logger.info(f"Subscribed chat {chat_id} to telemetry")

    def unsubscribe_telemetry(self, chat_id: int):
        self._telemetry_chat_ids.discard(chat_id)
        logger.info(f"Unsubscribed chat {chat_id} from telemetry")

    def _notify_subscribers(self, text: str):
        try:
            for cid in list(self._telemetry_chat_ids):
                self.telegram_bot.notify(cid, text)
        except Exception:
            logger.exception("Failed notifying telemetry subscribers")

    # --- Trading control API (called by Telegram bot) ---
    def enable_trading(self):
        """Enable auto trading."""
        self.trading_enabled = True
        logger.info("Auto trading enabled")

    def disable_trading(self):
        """Disable auto trading."""
        self.trading_enabled = False
        logger.info("Auto trading disabled")

    def set_mt5_connector(self, connector: MT5Connector):
        """Swap the MT5 connector (e.g., to a per-user session)."""
        if connector is None:
            return
        self.mt5 = connector
        # Reinitialize strategies to use the new connector
        self.strategies = []
        self._initialize_strategies()
        # Ensure AI strategy is also attached for snapshots/analysis
        if AI_AVAILABLE and self.mt5:
            try:
                ai_strat = AIStrategy(self.mt5)
                self.strategies.append(ai_strat)
                logger.info("AI Strategy reinitialized for current MT5 session")
            except Exception as e:
                logger.warning(f"Failed to reinitialize AI Strategy: {e}")
        logger.info("MT5 connector set from Telegram session; strategies reinitialized")
    
    # --- Per-user trading session management ---
    def _build_strategies_for_connector(self, connector: 'MT5Connector', chat_id: Optional[int] = None) -> List:
        strategies: List = []
        
        # Progress tracking disabled - no longer show initialization messages
        
        if STRATEGIES_AVAILABLE:
            try:
                strategies.append(TrendFollowingStrategy(connector))
            except Exception:
                logger.exception("Failed to init TrendFollowingStrategy")
            try:
                strategies.append(SupportResistanceStrategy(connector))
            except Exception:
                logger.exception("Failed to init SupportResistanceStrategy")
            try:
                strategies.append(BreakoutStrategy(connector))
            except Exception:
                logger.exception("Failed to init BreakoutStrategy")
            try:
                strategies.append(ReversalPatternsStrategy(connector))
            except Exception:
                logger.exception("Failed to init ReversalPatternsStrategy")
        # Optionally include AI strategy if available
        if AI_AVAILABLE:
            try:
                strategies.append(AIStrategy(connector))
            except Exception:
                logger.exception("Failed to init AI Strategy for session")
        
        # Progress tracking disabled - no longer needed
        
        return strategies

    def _start_init_progress(self, chat_id: int):
        """Start initialization progress tracking for a chat"""
        # Disabled: No longer show initialization progress messages
        pass

    def _update_init_progress(self, chat_id: int):
        """Update initialization progress for a chat"""
        # Disabled: No longer show initialization progress messages
        pass

    def _clear_init_progress(self, chat_id: int):
        """Clear initialization progress for a chat"""
        try:
            if chat_id in self._init_progress:
                progress = self._init_progress[chat_id]
                # Delete the progress message
                if self.telegram_bot and hasattr(self.telegram_bot, 'delete_message'):
                    self.telegram_bot.delete_message(chat_id, progress['message_id'])
                del self._init_progress[chat_id]
        except Exception as e:
            logger.warning(f"Failed to clear init progress for chat {chat_id}: {e}")

    def start_trading_for_chat(self, chat_id: int, connector: 'MT5Connector'):
        """Enable trading for a specific Telegram chat using its connector."""
        if chat_id not in self._user_sessions:
            self._user_sessions[chat_id] = {
                'connector': connector,
                'strategies': self._build_strategies_for_connector(connector, chat_id),
                'trading_enabled': True,
                'daily_trades': 0,
                'last_trade_time': None,
                'stats': {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_profit': 0.0,
                    'daily_trades': 0,
                    'session_start': datetime.now(),
                },
            }
            
            # Initialize previous positions tracking for this session
            self._previous_positions[chat_id] = {}
        else:
            state = self._user_sessions[chat_id]
            state['connector'] = connector
            if not state.get('strategies'):
                state['strategies'] = self._build_strategies_for_connector(connector, chat_id)
            state['trading_enabled'] = True
            # Reset cooldown timers when restarting trading
            state['last_trade_time'] = None
            state['last_loss_time'] = None
        logger.info(f"Enabled trading for chat {chat_id}")

    def stop_trading_for_chat(self, chat_id: int):
        """Disable trading for a specific Telegram chat."""
        state = self._user_sessions.get(chat_id)
        if state:
            state['trading_enabled'] = False
            logger.info(f"Disabled trading for chat {chat_id}")

    def close_all_positions_for_chat(self, chat_id: int) -> dict:
        """Close all positions for a specific Telegram chat/account."""
        state = self._user_sessions.get(chat_id)
        if not state or not state.get('connector'):
            return {'message': 'No active session for this chat'}
        try:
            close_results = state['connector'].close_all_positions()
            
            # Update session statistics for manually closed positions
            if 'details' in close_results:
                for detail in close_results['details']:
                    if detail.get('status') == 'closed':
                        profit = detail.get('profit', 0.0)
                        session_stats = state.get('stats', {})
                        session_stats['total_profit'] = session_stats.get('total_profit', 0.0) + profit
                        if profit > 0:
                            session_stats['winning_trades'] = session_stats.get('winning_trades', 0) + 1
                        else:
                            session_stats['losing_trades'] = session_stats.get('losing_trades', 0) + 1
                        state['stats'] = session_stats
                        logger.info(f"Updated session stats for manual close: {detail['symbol']} P/L: {profit:.2f}")
            
            return close_results
        except Exception as e:
            logger.error(f"Error closing positions for chat {chat_id}: {e}")
            return {'message': f'Error: {e}'}

    # --- Internal helpers for per-user loop ---
    def _should_trade_for_state(self, state: dict, connector: 'MT5Connector') -> bool:
        if not state.get('trading_enabled'):
            return False
        # Weekday filter
        try:
            from config import TRADE_WEEKDAYS_ONLY
        except Exception:
            TRADE_WEEKDAYS_ONLY = True
        if TRADE_WEEKDAYS_ONLY:
            if datetime.now().weekday() > 4:
                return False
        if not TRADE_ANYTIME:
            try:
                if not connector.get_trading_hours():
                    return False
            except Exception:
                return False
        # Cooldown after loss
        try:
            if LOSS_COOLDOWN_MINUTES > 0 and state.get('last_loss_time'):
                if datetime.now() - state['last_loss_time'] < timedelta(minutes=LOSS_COOLDOWN_MINUTES):
                    return False
        except Exception:
            pass
        # Daily drawdown stop (per-session)
        try:
            stats = state.get('stats', {}) if isinstance(state.get('stats'), dict) else {}
            day_pnl = float(stats.get('day_profit', 0.0))
            if DAILY_MAX_DRAWDOWN_PERCENT > 0:
                ai = connector.get_account_info() or {}
                bal = float(ai.get('balance', 0.0) or 0.0)
                if bal > 0 and day_pnl < 0:
                    if (abs(day_pnl) / bal) * 100 >= DAILY_MAX_DRAWDOWN_PERCENT:
                        return False
        except Exception:
            pass
        # Check per-session daily trade limit (disabled when MAX_DAILY_TRADES <= 0)
        if MAX_DAILY_TRADES and MAX_DAILY_TRADES > 0:
            if state.get('daily_trades', 0) >= MAX_DAILY_TRADES:
                return False
        last_trade_time = state.get('last_trade_time')
        if last_trade_time:
            time_since_last = datetime.now() - last_trade_time
            if time_since_last < timedelta(minutes=5):
                return False
        return True

    def _trading_loop_for_connector(self, connector: 'MT5Connector', strategies: List, state_or_self):
        """Run one trading pass for a given connector and strategy set.
        If state_or_self is a session dict, apply per-session gates and telemetry; otherwise legacy global behavior.
        """
        try:
            # Per-state gate
            if isinstance(state_or_self, dict):
                state = state_or_self
                if not self._should_trade_for_state(state, connector):
                    return
                
                # Progress tracking disabled - no longer show initialization messages
            else:
                # Legacy, use existing guard
                if not self._should_trade():
                    return
            
            # Check for automatically closed positions (broker SL/TP hits)
            try:
                self._check_for_auto_closed_positions(connector, state_or_self)
            except Exception:
                logger.exception("Error checking for auto-closed positions")

            from config import SYMBOLS
            # Choose per-session symbol variant if available
            try:
                base_symbols = SYMBOLS if isinstance(SYMBOLS, list) else [SYMBOLS]
                symbols_to_use = connector.detect_symbol_variant(base_symbols)
            except Exception:
                symbols_to_use = SYMBOLS if isinstance(SYMBOLS, list) else [SYMBOLS]
            for sym in symbols_to_use:
                try:
                    connector.change_symbol(sym)
                except Exception:
                    pass
                df = connector.get_rates(sym, TIMEFRAME, 200)
                if df is None or not DataValidation.validate_ohlc_data(df):
                    continue
                # Manage existing positions for this symbol before taking new signals
                try:
                    existing_positions = connector.get_positions(sym)
                    if existing_positions:
                        self._manage_existing_positions(existing_positions, DataValidation.clean_data(df.copy()), mt5_conn=connector, strategies=strategies)
                except Exception:
                    logger.exception("Failed managing existing positions for symbol")

                analyses = []
                # Analyze with each strategy
                for strategy in strategies:
                    try:
                        analysis = strategy.analyze(df)
                        if analysis and analysis.get('signal') in ('buy', 'sell'):
                            # Enforce per-strategy minimum confidence (non-AI)
                            conf_val = float(analysis.get('confidence', 0) or 0)
                            if strategy.__class__.__name__ == 'AIStrategy':
                                analyses.append((analysis, strategy))
                            else:
                                strat_min_conf = float(os.getenv('SIG_STRAT_MIN_CONF', '60'))
                                if conf_val >= strat_min_conf:
                                    analyses.append((analysis, strategy))
                    except Exception:
                        logger.exception("Strategy analysis failed")

                # Apply quality filters and select best candidate for execution
                if analyses:
                    try:
                        # Volatility filter (ATR-based)
                        try:
                            from utils import TechnicalIndicators
                            atr_series = TechnicalIndicators.atr(df['high'], df['low'], df['close'], int(os.getenv('SIG_MIN_ATR_PERIOD', '14')))
                            atr = float(atr_series.iloc[-1]) if atr_series is not None else None
                        except Exception:
                            atr = None
                        
                        if os.getenv('SIG_VOL_FILTER_ENABLED', 'True').lower() == 'true' and atr is not None:
                            close_px = float(df['close'].iloc[-1])
                            atr_pct = (atr / close_px) * 100 if close_px else 0.0
                            min_atr_pct = float(os.getenv('SIG_MIN_ATR_PCT', '0.03'))  # e.g., 0.03% min
                            max_atr_pct = float(os.getenv('SIG_MAX_ATR_PCT', '1.5'))   # optional cap, 1.5%
                            if atr_pct < min_atr_pct or atr_pct > max_atr_pct:
                                # Skip trading this symbol due to low/high volatility
                                logger.info(f"Symbol {sym}: Volatility filter failed (ATR: {atr_pct:.3f}%, min: {min_atr_pct}%, max: {max_atr_pct}%)")
                                continue

                        # Strategy consensus gating
                        want_consensus = os.getenv('SIG_CONSENSUS_ENABLED', 'True').lower() == 'true'
                        min_consensus = int(os.getenv('SIG_CONSENSUS_MIN', '2'))
                        consensus_signal = None
                        if want_consensus and min_consensus > 1:
                            dir_counts = {'buy': 0, 'sell': 0}
                            # Only include non-AI strategies that passed min-conf
                            for a, strat in analyses:
                                if strat.__class__.__name__ != 'AIStrategy':
                                    sig = a.get('signal')
                                    if sig in dir_counts:
                                        dir_counts[sig] += 1
                            # Determine consensus direction
                            for d in ('buy', 'sell'):
                                if dir_counts[d] >= min_consensus:
                                    consensus_signal = d
                                    break
                            if consensus_signal is None:
                                # Not enough agreement
                                logger.info(f"Symbol {sym}: No consensus reached (buy:{dir_counts['buy']}, sell:{dir_counts['sell']}, need:{min_consensus})")
                                continue

                        # Optional: require no high-confidence contradiction
                        if os.getenv('SIG_REQUIRE_NO_CONTRADICTION', 'False').lower() == 'true':
                            contra_conf = float(os.getenv('SIG_CONTRA_CONF', '65'))
                            opposite = 'sell' if consensus_signal == 'buy' else 'buy'
                            contradicts = [a for a, strat in analyses if strat.__class__.__name__ != 'AIStrategy' and a.get('signal') == opposite and float(a.get('confidence', 0) or 0) >= contra_conf]
                            if contradicts:
                                continue

                        # AI alignment gating
                        ai_align = os.getenv('SIG_AI_ALIGN_ENABLED', 'True').lower() == 'true'
                        ai_min_conf = float(os.getenv('SIG_AI_MIN_CONF', '70'))
                        ai_item = None
                        for a, strat in analyses:
                            if strat.__class__.__name__ == 'AIStrategy':
                                ai_item = (a, strat)
                                break
                        
                        if ai_align and ai_item is not None:
                            ai_analysis, _ai_strat = ai_item
                            ai_conf = float(ai_analysis.get('confidence', 0) or 0)
                            if ai_conf >= ai_min_conf:
                                # Require AI direction to match consensus (if present) or to be used as consensus
                                ai_sig = ai_analysis.get('signal')
                                if consensus_signal is not None and ai_sig != consensus_signal:
                                    logger.info(f"Symbol {sym}: AI alignment failed (AI: {ai_sig}, Consensus: {consensus_signal})")
                                    continue
                                if consensus_signal is None:
                                    consensus_signal = ai_sig

                        # Filter candidates to those matching consensus if defined
                        filtered = []
                        if consensus_signal is not None:
                            for a, strat in analyses:
                                if a.get('signal') == consensus_signal:
                                    filtered.append((a, strat))
                        else:
                            filtered = analyses

                        if not filtered:
                            logger.info(f"Symbol {sym}: No candidates match consensus signal ({consensus_signal})")
                            continue

                        def _confidence(item):
                            a, _ = item
                            return a.get('confidence', 0) or 0
                        best_analysis, best_strategy = max(filtered, key=_confidence)
                        
                        # Additional check: ensure no existing positions on this symbol
                        try:
                            current_positions = connector.get_positions(sym)
                            if current_positions:
                                logger.info(f"Skipping trade - {len(current_positions)} existing position(s) on {sym}")
                                continue
                        except Exception as e:
                            logger.warning(f"Error checking positions for {sym}: {e}")
                        
                        # Check maximum open positions limit across all symbols
                        try:
                            all_positions = connector.get_positions()  # Get all positions
                            if len(all_positions) >= MAX_OPEN_POSITIONS:
                                logger.info(f"Skipping trade - maximum open positions limit reached ({len(all_positions)}/{MAX_OPEN_POSITIONS})")
                                continue
                        except Exception as e:
                            logger.warning(f"Error checking total positions: {e}")
                        
                        # Execute trade using connector directly
                        self._execute_trade_for_session(connector, state_or_self, best_analysis, best_strategy)
                    except Exception:
                        logger.exception("Failed to select/execute best analysis")

                # Telemetry intentionally omitted
        except Exception as e:
            logger.error(f"Error in trading pass: {e}")

    def _execute_trade_for_session(self, connector: 'MT5Connector', state_or_self, analysis: Dict, strategy):
        try:
            # Check if we already have a position from this strategy to prevent duplicates
            try:
                current_symbol = connector.get_symbol()
                positions = connector.get_positions(current_symbol)
                for position in positions:
                    if position['magic'] == getattr(strategy, 'magic_number', 0):
                        logger.info(f"Skipping trade - position already exists for strategy {strategy.name} (magic: {getattr(strategy, 'magic_number', 0)})")
                        return
            except Exception as e:
                logger.warning(f"Error checking existing positions: {e}")
            
            # Check maximum open positions limit across all symbols
            try:
                all_positions = connector.get_positions()  # Get all positions
                if len(all_positions) >= MAX_OPEN_POSITIONS:
                    logger.info(f"Skipping trade - maximum open positions limit reached ({len(all_positions)}/{MAX_OPEN_POSITIONS})")
                    return
            except Exception as e:
                logger.warning(f"Error checking total positions: {e}")
            
            account_info = connector.get_account_info()
            if not account_info:
                logger.error("Failed to get account info")
                return
            risk_amount = account_info['balance'] * (RISK_PERCENTAGE / 100)
            entry_price = analysis['entry_price']
            stop_loss = analysis['stop_loss']
            point_value = 0.0
            try:
                point_value = connector.symbol_info.point if connector.symbol_info else 0.0
            except Exception:
                point_value = 0.0
            if not point_value:
                # attempt to refresh symbol info
                try:
                    connector.change_symbol(connector.get_symbol())
                    point_value = connector.symbol_info.point if connector.symbol_info else 0.0
                except Exception:
                    point_value = 0.0
            if not point_value:
                logger.error("Symbol point value unavailable; skipping trade execution")
                return
            stop_loss_points = abs(entry_price - stop_loss) / point_value
            lot_size = connector.calculate_lot_size(risk_amount, int(stop_loss_points))
            order_type = analysis['signal']
            # Guard: spread and slippage
            try:
                spread_pts = connector.get_spread(connector.get_symbol())
                if spread_pts is not None and spread_pts > MAX_SPREAD_POINTS:
                    logger.info(f"Skip trade due to spread {spread_pts} > {MAX_SPREAD_POINTS}")
                    return
            except Exception:
                pass

            result = connector.place_market_order(
                order_type=order_type,
                volume=lot_size,
                sl=stop_loss,
                tp=analysis['take_profit'],
                comment=f"{strategy.name} - {analysis.get('pattern_type', 'Signal')}",
                magic=getattr(strategy, 'magic_number', 0)
            )
            if result:
                # Progress tracking disabled - no longer needed
                
                # Update per-session or global counters
                now = datetime.now()
                if isinstance(state_or_self, dict):
                    state = state_or_self
                    state['daily_trades'] = int(state.get('daily_trades', 0)) + 1
                    state['last_trade_time'] = now
                    stats = state.get('stats', {})
                    stats['total_trades'] = int(stats.get('total_trades', 0)) + 1
                    stats['daily_trades'] = int(stats.get('daily_trades', 0)) + 1
                    state['stats'] = stats
                    # Reset loss cooldown on new trade open
                    state['last_loss_time'] = state.get('last_loss_time')
                else:
                    self.daily_trades += 1
                    self.last_trade_time = now
                    self.stats['total_trades'] += 1
                    self.stats['daily_trades'] += 1
                if TELEGRAM_ENABLED and self.telegram_bot:
                    try:
                        sym = connector.get_symbol()
                    except Exception:
                        sym = SYMBOL
                    ticket = result.get('order') if isinstance(result, dict) else None
                    msg = (
                        f"✅ Trade Executed\n"
                        f"{order_type.upper()} {sym} vol={lot_size}\n"
                        f"Entry={entry_price} SL={stop_loss} TP={analysis['take_profit']}\n"
                        f"Strategy: {strategy.name} | Conf: {analysis.get('confidence', 0)}%\n"
                        f"Ticket: {ticket if ticket is not None else 'n/a'}"
                    )
                    # Send to all telemetry subscribers
                    self._notify_subscribers(msg)
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def _should_take_signal(self, analysis: Dict, strategy, connector: 'MT5Connector' = None) -> bool:
        """Check if we should take a trading signal"""
        # Check signal confidence
        confidence = analysis.get('confidence', 0)
        if confidence < 60:  # Minimum confidence threshold
            return False
        
        # Use provided connector or fall back to self.mt5
        mt5_connector = connector or self.mt5
        if not mt5_connector:
            return False
        
        # Check if we already have a position from this strategy
        try:
            current_symbol = mt5_connector.get_symbol()
            positions = mt5_connector.get_positions(current_symbol)
            for position in positions:
                if position['magic'] == getattr(strategy, 'magic_number', 0):
                    return False
        except Exception as e:
            logger.warning(f"Error checking positions in _should_take_signal: {e}")
            return False
        
        return True
    
    def _execute_trade(self, analysis: Dict, strategy):
        """Execute a trade based on strategy analysis"""
        try:
            # Calculate position size
            account_info = self.mt5.get_account_info()
            if not account_info:
                logger.error("Failed to get account info")
                return
            
            # Calculate risk amount
            risk_amount = account_info['balance'] * (RISK_PERCENTAGE / 100)
            
            # Calculate stop loss distance
            entry_price = analysis['entry_price']
            stop_loss = analysis['stop_loss']
            point_value = 0.0
            try:
                point_value = self.mt5.symbol_info.point if self.mt5.symbol_info else 0.0
            except Exception:
                point_value = 0.0
            if not point_value:
                try:
                    self.mt5.change_symbol(self.mt5.get_symbol())
                    point_value = self.mt5.symbol_info.point if self.mt5.symbol_info else 0.0
                except Exception:
                    point_value = 0.0
            if not point_value:
                logger.error("Symbol point value unavailable; skipping trade execution")
                return
            stop_loss_points = abs(entry_price - stop_loss) / point_value
            
            # Calculate lot size
            lot_size = self.mt5.calculate_lot_size(risk_amount, int(stop_loss_points))
            
            # Place order
            order_type = analysis['signal']
            result = self.mt5.place_market_order(
                order_type=order_type,
                volume=lot_size,
                sl=stop_loss,
                tp=analysis['take_profit'],
                comment=f"{strategy.name} - {analysis.get('pattern_type', 'Signal')}",
                magic=strategy.magic_number
            )
            
            if result:
                self.daily_trades += 1
                self.last_trade_time = datetime.now()
                self.stats['total_trades'] += 1
                self.stats['daily_trades'] += 1
                
                logger.info(f"Trade executed: {order_type} {lot_size} {SYMBOL}")
                logger.info(f"Entry: {entry_price}, SL: {stop_loss}, TP: {analysis['take_profit']}")
                logger.info(f"Strategy: {strategy.name}, Confidence: {analysis['confidence']}%")
                
                # Send notification if enabled
                if TELEGRAM_ENABLED and self.telegram_bot:
                    sym = ''
                    try:
                        sym = self.mt5.get_symbol()  # prefer connector's active symbol
                    except Exception:
                        sym = SYMBOL
                    ticket = result.get('order') if isinstance(result, dict) else None
                    msg = (
                        f"✅ Trade Executed\n"
                        f"{order_type.upper()} {sym} vol={lot_size}\n"
                        f"Entry={entry_price} SL={stop_loss} TP={analysis['take_profit']}\n"
                        f"Strategy: {strategy.name} | Conf: {analysis['confidence']}%\n"
                        f"Ticket: {ticket if ticket is not None else 'n/a'}"
                    )
                    # Send to telemetry subscribers only
                    self._notify_subscribers(msg)
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def _manage_existing_positions(self, positions: List[Dict], df: pd.DataFrame, mt5_conn=None, strategies: List = None):
        """Manage existing positions. Optionally operate on a provided MT5 connector and strategy set."""
        active_mt5 = mt5_conn or self.mt5
        strategy_list = strategies or self.strategies
        for position in positions:
            try:
                # Find the strategy that opened this position
                strategy = None
                for s in strategy_list:
                    if s.magic_number == position['magic']:
                        strategy = s
                        break
                
                if not strategy:
                    continue
                
                # Trailing stop management before exit checks
                try:
                    if TRAILING_STOP_ENABLED and active_mt5 and df is not None and len(df) >= 20:
                        self._apply_trailing_stop(position, df)
                except Exception:
                    logger.exception("Trailing stop update failed")

                # Check if position should be exited
                exit_analysis = strategy.should_exit_position(position, df)
                
                if exit_analysis.get('exit', False):
                    # Get close reason
                    close_reason = exit_analysis.get('reason', 'Unknown reason')
                    
                    # Close position with reason
                    if active_mt5.close_position(position['ticket'], close_reason):
                        profit = position['profit']
                        
                        # Update global stats
                        self.stats['total_profit'] += profit
                        if profit > 0:
                            self.stats['winning_trades'] += 1
                        else:
                            self.stats['losing_trades'] += 1
                        
                        # Update per-user session stats
                        try:
                            # Find which user session this connector belongs to
                            for chat_id, session_state in self._user_sessions.items():
                                if session_state.get('connector') is active_mt5:
                                    session_stats = session_state.get('stats', {})
                                    session_stats['total_profit'] = session_stats.get('total_profit', 0.0) + profit
                                    if profit > 0:
                                        session_stats['winning_trades'] = session_stats.get('winning_trades', 0) + 1
                                    else:
                                        session_stats['losing_trades'] = session_stats.get('losing_trades', 0) + 1
                                    session_state['stats'] = session_stats
                                    break
                        except Exception:
                            logger.exception("Failed to update per-user session stats")
                        
                        logger.info(f"Position closed: {position['ticket']}, Profit: {profit:.2f}")
                        logger.info(f"Close reason: {close_reason}")
                        
                        # Track close reason
                        self.close_reasons.append({
                            'ticket': position['ticket'],
                            'symbol': position['symbol'],
                            'type': position['type'],
                            'profit': profit,
                            'reason': close_reason,
                            'timestamp': datetime.now(),
                            'strategy': strategy.__class__.__name__
                        })
                        # Update cooldown after loss
                        try:
                            if profit < 0:
                                # Map back to session to set last_loss_time
                                for chat_id, st in self._user_sessions.items():
                                    if st.get('connector') is active_mt5:
                                        st['last_loss_time'] = datetime.now()
                                        break
                        except Exception:
                            pass
                        
                        # Send notification if enabled
                        if TELEGRAM_ENABLED and self.telegram_bot:
                            try:
                                # Create more descriptive messages for TP/SL hits
                                if "profit target reached" in close_reason.lower() or "take profit" in close_reason.lower():
                                    emoji = "🎯"
                                    title = "Take Profit Hit!"
                                elif "stop loss" in close_reason.lower() or "stop loss triggered" in close_reason.lower():
                                    emoji = "🛑"
                                    title = "Stop Loss Hit!"
                                else:
                                    emoji = "💰" if profit > 0 else "📉"
                                    title = "Position Closed"
                                
                                # Build detailed notification message
                                symbol = position.get('symbol', 'Unknown')
                                volume = position.get('volume', 0)
                                position_type = position.get('type', '').upper()
                                
                                message_lines = [
                                    f"{emoji} **{title}**",
                                    f"",
                                    f"**Ticket:** #{position['ticket']}",
                                    f"**Symbol:** {symbol}",
                                    f"**Type:** {position_type}",
                                    f"**Volume:** {volume}",
                                    f"**P/L:** {profit:.2f}",
                                    f"**Reason:** {close_reason}",
                                    f"**Strategy:** {strategy.__class__.__name__}"
                                ]
                                
                                self._notify_subscribers("\n".join(message_lines))
                            except Exception:
                                logger.exception("Failed to notify position close")
                
            except Exception as e:
                logger.error(f"Error managing position {position['ticket']}: {e}")

    def _apply_trailing_stop(self, position: Dict, df: 'pd.DataFrame'):
        """Compute and update trailing stop for a single position."""
        symbol = position['symbol']
        ticket = position['ticket']
        current_price = position['price_current']
        price_open = position['price_open']
        sl = position.get('sl')
        tp = position.get('tp')
        # Determine trail distance
        if TRAILING_STOP_TYPE == 'atr':
            try:
                from utils import TechnicalIndicators
                atr_series = TechnicalIndicators.atr(df['high'], df['low'], df['close'], TRAILING_STOP_ATR_PERIOD)
                trail_dist = float(atr_series.iloc[-1]) * TRAILING_STOP_ATR_MULTIPLIER
            except Exception:
                return
        else:
            # points-based
            point = 0.0
            try:
                point = self.mt5.symbol_info.point if getattr(self.mt5, 'symbol_info', None) else 0.0
            except Exception:
                point = 0.0
            if not point:
                return
            trail_dist = TRAILING_STOP_POINTS * point

        new_sl = None
        new_tp = tp
        if position['type'] == 'buy':
            # Only trail if in profit
            if current_price <= price_open:
                return
            candidate = current_price - trail_dist
            if sl is None or sl == 0 or candidate > sl:
                new_sl = candidate
            # Break-even move
            try:
                if BREEAKVEN_ENABLED := BREAKEVEN_ENABLED:
                    r = (price_open - sl) if sl else trail_dist
                    if r and (current_price - price_open) >= (BREAKEVEN_TRIGGER_R_MULT * r):
                        be_price = price_open + (BREAKEVEN_OFFSET_POINTS * (self.mt5.symbol_info.point if getattr(self.mt5, 'symbol_info', None) else 0.0))
                        if new_sl is None or be_price > new_sl:
                            new_sl = be_price
            except Exception:
                pass
        else:  # sell
            if current_price >= price_open:
                return
            candidate = current_price + trail_dist
            if sl is None or sl == 0 or candidate < sl:
                new_sl = candidate
            try:
                if BREEAKVEN_ENABLED := BREAKEVEN_ENABLED:
                    r = (sl - price_open) if sl else trail_dist
                    if r and (price_open - current_price) >= (BREAKEVEN_TRIGGER_R_MULT * r):
                        be_price = price_open - (BREAKEVEN_OFFSET_POINTS * (self.mt5.symbol_info.point if getattr(self.mt5, 'symbol_info', None) else 0.0))
                        if new_sl is None or be_price < new_sl:
                            new_sl = be_price
            except Exception:
                pass

        # Partial take profit trigger
        if PARTIAL_TP_ENABLED:
            try:
                # Estimate initial risk in price terms
                init_r = None
                if position['type'] == 'buy' and sl:
                    init_r = price_open - sl
                elif position['type'] == 'sell' and sl:
                    init_r = sl - price_open
                if init_r and init_r > 0:
                    if position['type'] == 'buy' and (current_price - price_open) >= PARTIAL_TP_TRIGGER_R_MULT * init_r:
                        self.mt5.close_partial_position(ticket, PARTIAL_TP_CLOSE_FRACTION, reason="Partial TP trigger")
                    elif position['type'] == 'sell' and (price_open - current_price) >= PARTIAL_TP_TRIGGER_R_MULT * init_r:
                        self.mt5.close_partial_position(ticket, PARTIAL_TP_CLOSE_FRACTION, reason="Partial TP trigger")
            except Exception:
                logger.exception("Partial TP handling failed")

        if new_sl is not None:
            try:
                ok = self.mt5.modify_position_sl_tp(ticket, new_sl, new_tp)
                if ok:
                    logger.info(f"Trailing SL updated for {ticket}: {sl} -> {new_sl}")
            except Exception:
                logger.exception("Failed to modify SL for trailing stop")

        # Optional: immediate exit on trailing breach for scalping
        try:
            if os.getenv('SCALPER_FORCE_TRAIL_EXIT', 'True').lower() == 'true':
                if position['type'] == 'buy' and sl:
                    if current_price <= sl:
                        self.mt5.close_position(ticket, reason="Trailing stop breached (scalp)")
                elif position['type'] == 'sell' and sl:
                    if current_price >= sl:
                        self.mt5.close_position(ticket, reason="Trailing stop breached (scalp)")
        except Exception:
            logger.exception("Force trailing exit check failed")
    
    def hourly_maintenance(self):
        """Hourly maintenance tasks"""
        try:
            # Update statistics
            self._update_statistics()
            
            # Log current status
            logger.info("=== Hourly Status ===")
            logger.info(f"Daily trades: {self.daily_trades}/{MAX_DAILY_TRADES}")
            logger.info(f"Open positions: {len(self.mt5.get_positions(SYMBOL))}")
            logger.info(f"Total profit: {self.stats['total_profit']:.2f}")
            
            # Check account status
            account_info = self.mt5.get_account_info()
            if account_info:
                logger.info(f"Account equity: {account_info['equity']}")
                logger.info(f"Free margin: {account_info['free_margin']}")
            
            # Detect close log updates and trigger auto-training
            try:
                import os
                os.makedirs('logs', exist_ok=True)
                close_log_path = os.path.join('logs', 'close_log.csv')
                if os.path.exists(close_log_path):
                    # Check if close_log.csv has been modified since last check
                    mtime = os.path.getmtime(close_log_path)
                    key = f"close_log_{int(mtime)}"
                    if key not in self._processed_close_logs:
                        logger.info("Detected close_log.csv update. Triggering auto-training.")
                        self._processed_close_logs.add(key)
                        # Trigger auto-trainer if available
                        try:
                            if getattr(self, 'ai_strategy', None) is not None:
                                # Prefer auto_trainer from telegram bot if present
                                auto_trainer = None
                                try:
                                    auto_trainer = getattr(self.telegram_bot, 'auto_trainer', None)
                                except Exception:
                                    auto_trainer = None
                                if auto_trainer:
                                    auto_trainer.start_auto_training()
                        except Exception:
                            logger.exception("Failed to trigger auto-training on close log update")
                # Also trigger if trade_log.csv or loss_log.csv updated
                trade_log_path = os.path.join('logs', 'trade_log.csv')
                loss_log_path = os.path.join('logs', 'loss_log.csv')
                try:
                    for path, label in [(trade_log_path, 'trade_log'), (loss_log_path, 'loss_log')]:
                        if os.path.exists(path):
                            mtime = os.path.getmtime(path)
                            key = f"{label}_{int(mtime)}"
                            if key not in self._processed_close_logs:
                                logger.info(f"Detected update to {os.path.basename(path)}. Triggering auto-training.")
                                self._processed_close_logs.add(key)
                                if getattr(self, 'ai_strategy', None) is not None:
                                    auto_trainer = None
                                    try:
                                        auto_trainer = getattr(self.telegram_bot, 'auto_trainer', None)
                                    except Exception:
                                        auto_trainer = None
                                    if auto_trainer:
                                        auto_trainer.start_auto_training()
                except Exception:
                    logger.exception("Error checking trade_log.csv for updates")
            except Exception:
                logger.exception("Error scanning logs for auto-training")
            
        except Exception as e:
            logger.error(f"Error in hourly maintenance: {e}")
    
    def daily_reset(self):
        """Daily reset tasks"""
        try:
            logger.info("=== Daily Reset ===")
            self.daily_trades = 0
            self.stats['daily_trades'] = 0
            
            # Log daily statistics
            win_rate = (self.stats['winning_trades'] / max(self.stats['total_trades'], 1)) * 100
            logger.info(f"Daily Statistics:")
            logger.info(f"Total trades: {self.stats['total_trades']}")
            logger.info(f"Win rate: {win_rate:.1f}%")
            logger.info(f"Total profit: {self.stats['total_profit']:.2f}")
            
        except Exception as e:
            logger.error(f"Error in daily reset: {e}")
    
    def _update_statistics(self):
        """Update trading statistics"""
        try:
            positions = self.mt5.get_positions(SYMBOL)
            total_profit = sum(pos['profit'] for pos in positions)
            
            # Update session profit
            self.stats['session_profit'] = total_profit
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
    
    def get_close_reasons_stats(self) -> Dict:
        """Get close reasons statistics"""
        if not self.close_reasons:
            return {'total_closes': 0, 'reasons': {}}
        
        # Count reasons
        reason_counts = {}
        strategy_counts = {}
        profit_by_reason = {}
        
        for close in self.close_reasons:
            reason = close['reason']
            strategy = close['strategy']
            profit = close['profit']
            
            # Count reasons
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
            # Count by strategy
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            # Profit by reason
            if reason not in profit_by_reason:
                profit_by_reason[reason] = {'total': 0, 'count': 0}
            profit_by_reason[reason]['total'] += profit
            profit_by_reason[reason]['count'] += 1
        
        # Calculate average profit by reason
        avg_profit_by_reason = {}
        for reason, data in profit_by_reason.items():
            avg_profit_by_reason[reason] = data['total'] / data['count']
        
        return {
            'total_closes': len(self.close_reasons),
            'reasons': reason_counts,
            'strategies': strategy_counts,
            'avg_profit_by_reason': avg_profit_by_reason,
            'recent_closes': self.close_reasons[-5:] if len(self.close_reasons) > 5 else self.close_reasons
        }
    
    def _send_telegram_notification(self, message: str):
        """Send Telegram notification"""
        if not TELEGRAM_ENABLED or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return
        
        try:
            import requests
            
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = {
                'chat_id': TELEGRAM_CHAT_ID,
                'text': f"🤖 Price Action Bot\n{message}",
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, data=data, timeout=10)
            if response.status_code == 200:
                logger.info("Telegram notification sent")
            else:
                logger.warning(f"Failed to send Telegram notification: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
    
    def stop(self):
        """Stop the trading bot"""
        logger.info("Stopping trading bot...")
        # Stop Telegram bot first
        try:
            if self.telegram_bot:
                self.telegram_bot.stop()
                logger.info("Telegram bot stopped")
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {e}")
        
        # Disconnect from MT5
        if self.mt5:
            self.mt5.disconnect()
        
        # Log final statistics
        logger.info("=== Final Statistics ===")
        logger.info(f"Session duration: {datetime.now() - self.session_start_time}")
        logger.info(f"Total trades: {self.stats['total_trades']}")
        logger.info(f"Total profit: {self.stats['total_profit']:.2f}")
        
        if self.stats['total_trades'] > 0:
            win_rate = (self.stats['winning_trades'] / self.stats['total_trades']) * 100
            logger.info(f"Win rate: {win_rate:.1f}%")
        
        logger.info("Trading bot stopped")

def main():
    """Main entry point"""
    print("=" * 60)
    print("🤖 Price Action Trading Bot")
    print("=" * 60)
    print(f"Symbol: {SYMBOL}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Risk per trade: {RISK_PERCENTAGE}%")
    print(f"Demo mode: {DEMO_MODE}")
    print("=" * 60)
    
    # Create and start bot
    bot = PriceActionTradingBot()
    
    try:
        bot.start()
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Fatal error: {e}")
    finally:
        bot.stop()

if __name__ == "__main__":
    main()
