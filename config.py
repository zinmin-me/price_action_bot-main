"""
Configuration file for Price Action Trading Bot
Contains MT5 credentials, risk settings, and trading parameters
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MT5 Connection Settings
MT5_LOGIN = int(os.getenv('MT5_LOGIN'))
MT5_PASSWORD = os.getenv('MT5_PASSWORD')
MT5_SERVER = os.getenv('MT5_SERVER')
MT5_PATH = os.getenv('MT5_PATH')

# Trading Settings
SYMBOL = os.getenv('SYMBOL')
# Optional multi-symbol list: comma or space separated
_SYMBOLS_RAW = os.getenv('SYMBOLS')
SYMBOLS = [s.strip() for s in _SYMBOLS_RAW.replace(',', ' ').split() if s.strip()] or [SYMBOL] if SYMBOL else []
TIMEFRAME = os.getenv('TIMEFRAME')
LOT_SIZE = float(os.getenv('LOT_SIZE'))

# Risk Management
RISK_PERCENTAGE = float(os.getenv('RISK_PERCENTAGE'))
MAX_SPREAD = int(os.getenv('MAX_SPREAD'))
MAX_DAILY_TRADES = int(os.getenv('MAX_DAILY_TRADES'))
MAX_OPEN_POSITIONS = int(os.getenv('MAX_OPEN_POSITIONS'))
MAX_SPREAD_POINTS = int(os.getenv('MAX_SPREAD_POINTS'))
MAX_SLIPPAGE_POINTS = int(os.getenv('MAX_SLIPPAGE_POINTS'))

# Dynamic Position Sizing
DYNAMIC_POSITION_SIZING = os.getenv('DYNAMIC_POSITION_SIZING').lower() == 'true'
MIN_POSITION_SIZE = float(os.getenv('MIN_POSITION_SIZE'))
MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE'))
VOLATILITY_ADJUSTMENT = os.getenv('VOLATILITY_ADJUSTMENT').lower() == 'true'

# Stop Loss and Take Profit (in points)
DEFAULT_SL_POINTS = int(os.getenv('DEFAULT_SL_POINTS'))
DEFAULT_TP_POINTS = int(os.getenv('DEFAULT_TP_POINTS'))

# SL/TP ratio: supports formats like "1:2" or a plain number like "2.0"
_SL_TO_TP_RATIO_RAW = os.getenv('SL_TO_TP_RATIO')
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
TREND_FOLLOWING_ENABLED = os.getenv('TREND_FOLLOWING_ENABLED').lower() == 'true'
SUPPORT_RESISTANCE_ENABLED = os.getenv('SUPPORT_RESISTANCE_ENABLED').lower() == 'true'
BREAKOUT_ENABLED = os.getenv('BREAKOUT_ENABLED').lower() == 'true'
REVERSAL_PATTERNS_ENABLED = os.getenv('REVERSAL_PATTERNS_ENABLED').lower() == 'true'

# Technical Indicators Settings
RSI_PERIOD = int(os.getenv('RSI_PERIOD'))
RSI_OVERBOUGHT = int(os.getenv('RSI_OVERBOUGHT'))
RSI_OVERSOLD = int(os.getenv('RSI_OVERSOLD'))

MA_FAST_PERIOD = int(os.getenv('MA_FAST_PERIOD'))
MA_SLOW_PERIOD = int(os.getenv('MA_SLOW_PERIOD'))

# Additional Indicator Filters
INDICATORS_ENABLE_MACD = os.getenv('INDICATORS_ENABLE_MACD').lower() == 'true'
INDICATORS_ENABLE_STOCH = os.getenv('INDICATORS_ENABLE_STOCH').lower() == 'true'
INDICATORS_ENABLE_BB = os.getenv('INDICATORS_ENABLE_BB').lower() == 'true'

MACD_FAST = int(os.getenv('MACD_FAST'))
MACD_SLOW = int(os.getenv('MACD_SLOW'))
MACD_SIGNAL = int(os.getenv('MACD_SIGNAL'))

STOCH_K = int(os.getenv('STOCH_K'))
STOCH_D = int(os.getenv('STOCH_D'))
STOCH_SMOOTH = int(os.getenv('STOCH_SMOOTH'))
STOCH_OVERBOUGHT = int(os.getenv('STOCH_OVERBOUGHT'))
STOCH_OVERSOLD = int(os.getenv('STOCH_OVERSOLD'))

BB_PERIOD = int(os.getenv('BB_PERIOD'))
BB_STD = float(os.getenv('BB_STD'))

# Scalper-oriented additional indicators
EMA_FAST_SHORT = int(os.getenv('EMA_FAST_SHORT'))
EMA_SLOW_SHORT = int(os.getenv('EMA_SLOW_SHORT'))
EMA_MEDIUM = int(os.getenv('EMA_MEDIUM'))
EMA_LONG = int(os.getenv('EMA_LONG'))

VWAP_ENABLED = os.getenv('VWAP_ENABLED').lower() == 'true'
RSI_FAST_PERIOD = int(os.getenv('RSI_FAST_PERIOD'))
OBV_ENABLED = os.getenv('OBV_ENABLED').lower() == 'true'

# Trailing Stop Settings
TRAILING_STOP_ENABLED = os.getenv('TRAILING_STOP_ENABLED').lower() == 'true'
# Type: 'points' or 'atr'
TRAILING_STOP_TYPE = os.getenv('TRAILING_STOP_TYPE').lower()
TRAILING_STOP_POINTS = int(os.getenv('TRAILING_STOP_POINTS'))
TRAILING_STOP_ATR_PERIOD = int(os.getenv('TRAILING_STOP_ATR_PERIOD'))
TRAILING_STOP_ATR_MULTIPLIER = float(os.getenv('TRAILING_STOP_ATR_MULTIPLIER'))

# Cooldown & Risk Stops
LOSS_COOLDOWN_MINUTES = int(os.getenv('LOSS_COOLDOWN_MINUTES'))
DAILY_MAX_DRAWDOWN_PERCENT = float(os.getenv('DAILY_MAX_DRAWDOWN_PERCENT'))

# Break-even & Partial Take Profit
BREAKEVEN_ENABLED = os.getenv('BREAKEVEN_ENABLED').lower() == 'true'
BREAKEVEN_TRIGGER_R_MULT = float(os.getenv('BREAKEVEN_TRIGGER_R_MULT'))
BREAKEVEN_OFFSET_POINTS = int(os.getenv('BREAKEVEN_OFFSET_POINTS'))

PARTIAL_TP_ENABLED = os.getenv('PARTIAL_TP_ENABLED').lower() == 'true'
PARTIAL_TP_TRIGGER_R_MULT = float(os.getenv('PARTIAL_TP_TRIGGER_R_MULT'))
PARTIAL_TP_CLOSE_FRACTION = float(os.getenv('PARTIAL_TP_CLOSE_FRACTION'))

# Higher timeframe confirmation
HTF_CONFIRM_ENABLED = os.getenv('HTF_CONFIRM_ENABLED').lower() == 'true'
HTF_TIMEFRAME = os.getenv('HTF_TIMEFRAME')

# Support/Resistance Settings
SR_LOOKBACK_PERIODS = int(os.getenv('SR_LOOKBACK_PERIODS'))
SR_TOUCH_TOLERANCE = int(os.getenv('SR_TOUCH_TOLERANCE'))  # Points

# Breakout Settings
BREAKOUT_CONFIRMATION_BARS = int(os.getenv('BREAKOUT_CONFIRMATION_BARS'))
BREAKOUT_MIN_VOLUME = float(os.getenv('BREAKOUT_MIN_VOLUME'))

# Telegram Notifications (Optional)
TELEGRAM_ENABLED = os.getenv('TELEGRAM_ENABLED').lower() == 'true'
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Economic Calendar (Optional)
TE_API_CLIENT = os.getenv('TE_API_CLIENT')  # TradingEconomics demo
TE_COUNTRY = os.getenv('TE_COUNTRY')
TE_IMPORTANCE = os.getenv('TE_IMPORTANCE')  # low|medium|high|all

# News Headlines (Optional)
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
NEWS_COUNTRY = os.getenv('NEWS_COUNTRY')
NEWS_CATEGORY = os.getenv('NEWS_CATEGORY')

# Session/News/Holiday Filters
SESSION_FILTER_ENABLED = os.getenv('SESSION_FILTER_ENABLED').lower() == 'true'
# Comma-separated windows in HH:MM-HH:MM (local server time)
SESSION_WINDOWS = os.getenv('SESSION_WINDOWS')
HOLIDAY_FILTER_ENABLED = os.getenv('HOLIDAY_FILTER_ENABLED').lower() == 'true'
# Comma-separated YYYY-MM-DD dates
HOLIDAYS = [d.strip() for d in os.getenv('HOLIDAYS').split(',') if d.strip()]
NEWS_FILTER_ENABLED = os.getenv('NEWS_FILTER_ENABLED').lower() == 'true'

# Logging Settings
LOG_LEVEL = os.getenv('LOG_LEVEL')
LOG_FILE = os.getenv('LOG_FILE')

# Trading Hours (24-hour format)
TRADING_START_HOUR = int(os.getenv('TRADING_START_HOUR'))
TRADING_END_HOUR = int(os.getenv('TRADING_END_HOUR'))
# Bypass trading hours check (trade anytime)
TRADE_ANYTIME = os.getenv('TRADE_ANYTIME').lower() == 'true'
# Trade only on weekdays (Mon-Fri). Set to False to allow weekends
TRADE_WEEKDAYS_ONLY = os.getenv('TRADE_WEEKDAYS_ONLY').lower() == 'true'

# Demo/Live Mode
DEMO_MODE = os.getenv('DEMO_MODE').lower() == 'true'

# Auto login default MT5 session on startup
AUTO_LOGIN = os.getenv('AUTO_LOGIN').lower() == 'true'

# Magic Numbers for different strategies (these are static identifiers, not configurable via .env)
MAGIC_TREND_FOLLOWING = 1001
MAGIC_SUPPORT_RESISTANCE = 1002
MAGIC_BREAKOUT = 1003
MAGIC_REVERSAL_PATTERNS = 1004

# Timeframe mapping (this is a static mapping, not configurable via .env)
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
AI_RETRAIN_INTERVAL_HOURS = int(os.getenv('AI_RETRAIN_INTERVAL_HOURS'))
AI_MIN_TRAIN_SAMPLES = int(os.getenv('AI_MIN_TRAIN_SAMPLES'))
AI_PERF_THRESHOLD = float(os.getenv('AI_PERF_THRESHOLD'))
AI_MIN_PREDICTIONS_FOR_PERF = int(os.getenv('AI_MIN_PREDICTIONS_FOR_PERF'))
AI_DRIFT_CHECK_ENABLED = os.getenv('AI_DRIFT_CHECK_ENABLED').lower() == 'true'
AI_DRIFT_WINDOW = int(os.getenv('AI_DRIFT_WINDOW'))
AI_DRIFT_LOSS_RATE_ALERT = float(os.getenv('AI_DRIFT_LOSS_RATE_ALERT'))
AI_AUTORETRAIN_ON_DRIFT = os.getenv('AI_AUTORETRAIN_ON_DRIFT').lower() == 'true'

# Signal Filtering Settings (Override environment variables)
SIG_CONSENSUS_ENABLED = os.getenv('SIG_CONSENSUS_ENABLED').lower() == 'true'
SIG_AI_ALIGN_ENABLED = os.getenv('SIG_AI_ALIGN_ENABLED').lower() == 'true'
SIG_STRAT_MIN_CONF = int(os.getenv('SIG_STRAT_MIN_CONF'))
SIG_VOL_FILTER_ENABLED = os.getenv('SIG_VOL_FILTER_ENABLED').lower() == 'true'
SIG_MIN_ATR_PCT = float(os.getenv('SIG_MIN_ATR_PCT'))
SIG_MAX_ATR_PCT = float(os.getenv('SIG_MAX_ATR_PCT'))

# Note: MAX_OPEN_POSITIONS and LOSS_COOLDOWN_MINUTES are defined above in Risk Management section