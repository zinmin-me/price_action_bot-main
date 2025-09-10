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
    from telegram.ext import CallbackQueryHandler, ContextTypes, CommandHandler
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
        # Trading enable/disable flag (disabled until user starts trading)
        self.trading_enabled = False
        # Telemetry subscribers (chat IDs)
        self._telemetry_chat_ids = set()
        
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
        """Main trading loop executed every minute"""
        try:
            # Check if we should trade
            if not self._should_trade():
                return
            
            # Iterate through configured symbols
            from config import SYMBOLS
            for sym in SYMBOLS:
                # Switch working symbol on connector
                try:
                    self.mt5.change_symbol(sym)
                except Exception:
                    pass
                # Get market data for this symbol
                df = self.mt5.get_rates(sym, TIMEFRAME, 200)
                if df is None or not DataValidation.validate_ohlc_data(df):
                    logger.warning("Invalid market data received")
                    continue
                
                # Clean data
                df = DataValidation.clean_data(df)
                
                # Check spread
                spread = self.mt5.get_spread(sym)
                if spread and spread > MAX_SPREAD:
                    logger.warning(f"Spread too high: {spread} points")
                    continue
                
                # Check existing positions for this symbol
                positions = self.mt5.get_positions(sym)
                self._manage_existing_positions(positions, df)
                
                # Check if we can open new positions
                if len(positions) >= MAX_OPEN_POSITIONS:
                    continue
                
                # Analyze with each strategy
                analysis_lines = [
                    f"🧠 Analysis Snapshot | {sym} {TIMEFRAME} | {datetime.now().strftime('%H:%M:%S')}"
                ]
                trade_candidates = []  # collect potential trades to avoid conflicting entries
                for strategy in self.strategies:
                    if not strategy.enabled:
                        continue
                    
                    try:
                        # Get strategy analysis
                        analysis = strategy.analyze(df)
                        # Append concise analysis line
                        sig = analysis.get('signal', 'no_signal')
                        conf = analysis.get('confidence', '-')
                        pat = analysis.get('pattern_type', '')
                        analysis_lines.append(
                            f"• {strategy.name}: {sig.upper()} (conf {conf}%) {pat}"
                        )
                        
                        if analysis['signal'] in ['buy', 'sell']:
                            # Check if we should take this signal
                            if self._should_take_signal(analysis, strategy):
                                trade_candidates.append((analysis, strategy))
                        
                    except Exception as e:
                        logger.error(f"Error in {strategy.name} analysis: {e}")

                # Resolve candidates: only execute at most one trade to avoid conflicting buy/sell
                if trade_candidates:
                    # If any open positions exist, skip opening new ones to prevent conflict/hedge
                    existing_positions = self.mt5.get_positions(sym)
                    if not existing_positions:
                        # choose the highest-confidence candidate
                        def _confidence(item):
                            a, _ = item
                            return a.get('confidence', 0) or 0
                        best_analysis, best_strategy = max(trade_candidates, key=_confidence)
                        self._execute_trade(best_analysis, best_strategy)

                # Send snapshot to subscribers
                if self._telemetry_chat_ids and len(analysis_lines) > 1 and TELEGRAM_ENABLED and self.telegram_bot:
                    # Disabled per user request to not show analysis
                    pass
            
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")

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
        # Check trading hours (optional bypass)
        if not TRADE_ANYTIME:
            if not self.mt5.get_trading_hours():
                return False
        
        # Check daily trade limit
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
        logger.info("MT5 connector set from Telegram session; strategies reinitialized")
    
    def _should_take_signal(self, analysis: Dict, strategy) -> bool:
        """Check if we should take a trading signal"""
        # Check signal confidence
        confidence = analysis.get('confidence', 0)
        if confidence < 60:  # Minimum confidence threshold
            return False
        
        # Check if we already have a position from this strategy
        positions = self.mt5.get_positions(SYMBOL)
        for position in positions:
            if position['magic'] == strategy.magic_number:
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
            stop_loss_points = abs(entry_price - stop_loss) / self.mt5.symbol_info.point
            
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
    
    def _manage_existing_positions(self, positions: List[Dict], df: pd.DataFrame):
        """Manage existing positions"""
        for position in positions:
            try:
                # Find the strategy that opened this position
                strategy = None
                for s in self.strategies:
                    if s.magic_number == position['magic']:
                        strategy = s
                        break
                
                if not strategy:
                    continue
                
                # Check if position should be exited
                exit_analysis = strategy.should_exit_position(position, df)
                
                if exit_analysis.get('exit', False):
                    # Get close reason
                    close_reason = exit_analysis.get('reason', 'Unknown reason')
                    
                    # Close position with reason
                    if self.mt5.close_position(position['ticket'], close_reason):
                        profit = position['profit']
                        self.stats['total_profit'] += profit
                        
                        if profit > 0:
                            self.stats['winning_trades'] += 1
                        else:
                            self.stats['losing_trades'] += 1
                        
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
                        
                        # Send notification if enabled
                        if TELEGRAM_ENABLED and self.telegram_bot:
                            try:
                                profit_emoji = "💰" if profit > 0 else "📉"
                                self._notify_subscribers(
                                    f"{profit_emoji} Position closed #{position['ticket']}\n"
                                    f"P/L: {profit:.2f}\n"
                                    f"Reason: {close_reason}"
                                )
                            except Exception:
                                logger.exception("Failed to notify position close")
                
            except Exception as e:
                logger.error(f"Error managing position {position['ticket']}: {e}")
    
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
