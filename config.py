"""
Configuration file for Price Action Trading Bot
Contains MT5 credentials, risk settings, and trading parameters
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MT5 Connection Settings
MT5_LOGIN = int(os.getenv('MT5_LOGIN', '0'))
MT5_PASSWORD = os.getenv('MT5_PASSWORD', '')
MT5_SERVER = os.getenv('MT5_SERVER', '')
MT5_PATH = os.getenv('MT5_PATH', 'C:\\Program Files\\MetaTrader 5\\terminal64.exe')

# Trading Settings
SYMBOL = os.getenv('SYMBOL', 'EURUSD')
# Optional multi-symbol list: comma or space separated
_SYMBOLS_RAW = os.getenv('SYMBOLS', '')
SYMBOLS = [s.strip() for s in _SYMBOLS_RAW.replace(',', ' ').split() if s.strip()] or [SYMBOL]
TIMEFRAME = os.getenv('TIMEFRAME', 'M15')  # M1, M5, M15, M30, H1, H4, D1
LOT_SIZE = float(os.getenv('LOT_SIZE', '0.01'))

# Risk Management
RISK_PERCENTAGE = float(os.getenv('RISK_PERCENTAGE', '2.0'))  # Risk per trade as % of account
MAX_SPREAD = int(os.getenv('MAX_SPREAD', '20'))  # Maximum spread in points
MAX_DAILY_TRADES = int(os.getenv('MAX_DAILY_TRADES', '0'))
MAX_OPEN_POSITIONS = int(os.getenv('MAX_OPEN_POSITIONS', '5'))
MAX_SPREAD_POINTS = int(os.getenv('MAX_SPREAD_POINTS', '25'))
MAX_SLIPPAGE_POINTS = int(os.getenv('MAX_SLIPPAGE_POINTS', '50'))

# Stop Loss and Take Profit (in points)
DEFAULT_SL_POINTS = int(os.getenv('DEFAULT_SL_POINTS', '50'))
DEFAULT_TP_POINTS = int(os.getenv('DEFAULT_TP_POINTS', '100'))

# SL/TP ratio: supports formats like "1:2" or a plain number like "2.0"
_SL_TO_TP_RATIO_RAW = os.getenv('SL_TO_TP_RATIO', '1:2')
try:
    if isinstance(_SL_TO_TP_RATIO_RAW, str) and ':' in _SL_TO_TP_RATIO_RAW:
        left, right = _SL_TO_TP_RATIO_RAW.split(':', 1)
        left_val = float(left.strip())
        right_val = float(right.strip())
        SL_TO_TP_RATIO = right_val / max(left_val, 1e-9)
    else:
        SL_TO_TP_RATIO = float(_SL_TO_TP_RATIO_RAW)
except Exception:
    # Fallback to 2.0 if parsing fails
    SL_TO_TP_RATIO = 2.0

# Strategy Settings
TREND_FOLLOWING_ENABLED = os.getenv('TREND_FOLLOWING_ENABLED', 'True').lower() == 'true'
SUPPORT_RESISTANCE_ENABLED = os.getenv('SUPPORT_RESISTANCE_ENABLED', 'True').lower() == 'true'
BREAKOUT_ENABLED = os.getenv('BREAKOUT_ENABLED', 'True').lower() == 'true'
REVERSAL_PATTERNS_ENABLED = os.getenv('REVERSAL_PATTERNS_ENABLED', 'True').lower() == 'true'

# Technical Indicators Settings
RSI_PERIOD = int(os.getenv('RSI_PERIOD', '14'))
RSI_OVERBOUGHT = int(os.getenv('RSI_OVERBOUGHT', '70'))
RSI_OVERSOLD = int(os.getenv('RSI_OVERSOLD', '30'))

MA_FAST_PERIOD = int(os.getenv('MA_FAST_PERIOD', '20'))
MA_SLOW_PERIOD = int(os.getenv('MA_SLOW_PERIOD', '50'))

# Additional Indicator Filters
INDICATORS_ENABLE_MACD = os.getenv('INDICATORS_ENABLE_MACD', 'True').lower() == 'true'
INDICATORS_ENABLE_STOCH = os.getenv('INDICATORS_ENABLE_STOCH', 'False').lower() == 'true'
INDICATORS_ENABLE_BB = os.getenv('INDICATORS_ENABLE_BB', 'False').lower() == 'true'

MACD_FAST = int(os.getenv('MACD_FAST', '12'))
MACD_SLOW = int(os.getenv('MACD_SLOW', '26'))
MACD_SIGNAL = int(os.getenv('MACD_SIGNAL', '9'))

STOCH_K = int(os.getenv('STOCH_K', '14'))
STOCH_D = int(os.getenv('STOCH_D', '3'))
STOCH_SMOOTH = int(os.getenv('STOCH_SMOOTH', '3'))
STOCH_OVERBOUGHT = int(os.getenv('STOCH_OVERBOUGHT', '80'))
STOCH_OVERSOLD = int(os.getenv('STOCH_OVERSOLD', '20'))

BB_PERIOD = int(os.getenv('BB_PERIOD', '20'))
BB_STD = float(os.getenv('BB_STD', '2.0'))

# Scalper-oriented additional indicators
EMA_FAST_SHORT = int(os.getenv('EMA_FAST_SHORT', '9'))
EMA_SLOW_SHORT = int(os.getenv('EMA_SLOW_SHORT', '20'))
EMA_MEDIUM = int(os.getenv('EMA_MEDIUM', '50'))
EMA_LONG = int(os.getenv('EMA_LONG', '200'))

VWAP_ENABLED = os.getenv('VWAP_ENABLED', 'False').lower() == 'true'
RSI_FAST_PERIOD = int(os.getenv('RSI_FAST_PERIOD', '7'))
OBV_ENABLED = os.getenv('OBV_ENABLED', 'False').lower() == 'true'

# Trailing Stop Settings
TRAILING_STOP_ENABLED = os.getenv('TRAILING_STOP_ENABLED', 'True').lower() == 'true'
# Type: 'points' or 'atr'
TRAILING_STOP_TYPE = os.getenv('TRAILING_STOP_TYPE', 'points').lower()
TRAILING_STOP_POINTS = int(os.getenv('TRAILING_STOP_POINTS', '150'))
TRAILING_STOP_ATR_PERIOD = int(os.getenv('TRAILING_STOP_ATR_PERIOD', '14'))
TRAILING_STOP_ATR_MULTIPLIER = float(os.getenv('TRAILING_STOP_ATR_MULTIPLIER', '1.5'))

# Cooldown & Risk Stops
LOSS_COOLDOWN_MINUTES = int(os.getenv('LOSS_COOLDOWN_MINUTES', '10'))
DAILY_MAX_DRAWDOWN_PERCENT = float(os.getenv('DAILY_MAX_DRAWDOWN_PERCENT', '5.0'))

# Break-even & Partial Take Profit
BREAKEVEN_ENABLED = os.getenv('BREAKEVEN_ENABLED', 'True').lower() == 'true'
BREAKEVEN_TRIGGER_R_MULT = float(os.getenv('BREAKEVEN_TRIGGER_R_MULT', '1.0'))
BREAKEVEN_OFFSET_POINTS = int(os.getenv('BREAKEVEN_OFFSET_POINTS', '10'))

PARTIAL_TP_ENABLED = os.getenv('PARTIAL_TP_ENABLED', 'False').lower() == 'true'
PARTIAL_TP_TRIGGER_R_MULT = float(os.getenv('PARTIAL_TP_TRIGGER_R_MULT', '1.5'))
PARTIAL_TP_CLOSE_FRACTION = float(os.getenv('PARTIAL_TP_CLOSE_FRACTION', '0.5'))

# Higher timeframe confirmation
HTF_CONFIRM_ENABLED = os.getenv('HTF_CONFIRM_ENABLED', 'False').lower() == 'true'
HTF_TIMEFRAME = os.getenv('HTF_TIMEFRAME', 'H1')

# Support/Resistance Settings
SR_LOOKBACK_PERIODS = int(os.getenv('SR_LOOKBACK_PERIODS', '100'))
SR_TOUCH_TOLERANCE = int(os.getenv('SR_TOUCH_TOLERANCE', '10'))  # Points

# Breakout Settings
BREAKOUT_CONFIRMATION_BARS = int(os.getenv('BREAKOUT_CONFIRMATION_BARS', '2'))
BREAKOUT_MIN_VOLUME = float(os.getenv('BREAKOUT_MIN_VOLUME', '1.5'))

# Telegram Notifications (Optional)
TELEGRAM_ENABLED = os.getenv('TELEGRAM_ENABLED', 'False').lower() == 'true'
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Economic Calendar (Optional)
TE_API_CLIENT = os.getenv('TE_API_CLIENT', 'guest:guest')  # TradingEconomics demo
TE_COUNTRY = os.getenv('TE_COUNTRY', 'United States')
TE_IMPORTANCE = os.getenv('TE_IMPORTANCE', 'high')  # low|medium|high|all

# News Headlines (Optional)
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
NEWS_COUNTRY = os.getenv('NEWS_COUNTRY', 'us')
NEWS_CATEGORY = os.getenv('NEWS_CATEGORY', 'business')

# Session/News/Holiday Filters
SESSION_FILTER_ENABLED = os.getenv('SESSION_FILTER_ENABLED', 'False').lower() == 'true'
# Comma-separated windows in HH:MM-HH:MM (local server time)
SESSION_WINDOWS = os.getenv('SESSION_WINDOWS', '08:00-12:00,13:00-17:00')
HOLIDAY_FILTER_ENABLED = os.getenv('HOLIDAY_FILTER_ENABLED', 'False').lower() == 'true'
# Comma-separated YYYY-MM-DD dates
HOLIDAYS = [d.strip() for d in os.getenv('HOLIDAYS', '').split(',') if d.strip()]
NEWS_FILTER_ENABLED = os.getenv('NEWS_FILTER_ENABLED', 'False').lower() == 'true'

# Logging Settings
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'trading_bot.log')

# Trading Hours (24-hour format)
TRADING_START_HOUR = int(os.getenv('TRADING_START_HOUR', '0'))
TRADING_END_HOUR = int(os.getenv('TRADING_END_HOUR', '23'))
# Bypass trading hours check (trade anytime)
TRADE_ANYTIME = os.getenv('TRADE_ANYTIME', 'True').lower() == 'true'
# Trade only on weekdays (Mon-Fri). Set to False to allow weekends
TRADE_WEEKDAYS_ONLY = os.getenv('TRADE_WEEKDAYS_ONLY', 'False').lower() == 'true'

# Demo/Live Mode
DEMO_MODE = os.getenv('DEMO_MODE', 'True').lower() == 'true'

# Auto login default MT5 session on startup
AUTO_LOGIN = os.getenv('AUTO_LOGIN', 'False').lower() == 'true'

# Magic Numbers for different strategies
MAGIC_TREND_FOLLOWING = 1001
MAGIC_SUPPORT_RESISTANCE = 1002
MAGIC_BREAKOUT = 1003
MAGIC_REVERSAL_PATTERNS = 1004

# Timeframe mapping
TIMEFRAME_MAPPING = {
    'M1': 1,
    'M5': 5,
    'M15': 15,
    'M30': 30,
    'H1': 16385,
    'H4': 16388,
    'D1': 16408
}

# AI Monitoring & Auto-Training
AI_RETRAIN_INTERVAL_HOURS = int(os.getenv('AI_RETRAIN_INTERVAL_HOURS', '24'))
AI_MIN_TRAIN_SAMPLES = int(os.getenv('AI_MIN_TRAIN_SAMPLES', '1000'))
AI_PERF_THRESHOLD = float(os.getenv('AI_PERF_THRESHOLD', '0.55'))
AI_MIN_PREDICTIONS_FOR_PERF = int(os.getenv('AI_MIN_PREDICTIONS_FOR_PERF', '50'))
AI_DRIFT_CHECK_ENABLED = os.getenv('AI_DRIFT_CHECK_ENABLED', 'True').lower() == 'true'
AI_DRIFT_WINDOW = int(os.getenv('AI_DRIFT_WINDOW', '200'))
AI_DRIFT_LOSS_RATE_ALERT = float(os.getenv('AI_DRIFT_LOSS_RATE_ALERT', '0.6'))
AI_AUTORETRAIN_ON_DRIFT = os.getenv('AI_AUTORETRAIN_ON_DRIFT', 'True').lower() == 'true'

# Signal Filtering Settings (Override environment variables)
SIG_CONSENSUS_ENABLED = os.getenv('SIG_CONSENSUS_ENABLED', 'False').lower() == 'true'
SIG_AI_ALIGN_ENABLED = os.getenv('SIG_AI_ALIGN_ENABLED', 'False').lower() == 'true'
SIG_STRAT_MIN_CONF = int(os.getenv('SIG_STRAT_MIN_CONF', '45'))
SIG_VOL_FILTER_ENABLED = os.getenv('SIG_VOL_FILTER_ENABLED', 'True').lower() == 'true'
SIG_MIN_ATR_PCT = float(os.getenv('SIG_MIN_ATR_PCT', '0.02'))
SIG_MAX_ATR_PCT = float(os.getenv('SIG_MAX_ATR_PCT', '2.0'))

# Trading Limits
MAX_OPEN_POSITIONS = int(os.getenv('MAX_OPEN_POSITIONS', '8'))
LOSS_COOLDOWN_MINUTES = int(os.getenv('LOSS_COOLDOWN_MINUTES', '5'))