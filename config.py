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
MAX_DAILY_TRADES = int(os.getenv('MAX_DAILY_TRADES', '10'))
MAX_OPEN_POSITIONS = int(os.getenv('MAX_OPEN_POSITIONS', '3'))

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

# Logging Settings
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'trading_bot.log')

# Trading Hours (24-hour format)
TRADING_START_HOUR = int(os.getenv('TRADING_START_HOUR', '8'))
TRADING_END_HOUR = int(os.getenv('TRADING_END_HOUR', '18'))
# Bypass trading hours check (trade anytime)
TRADE_ANYTIME = os.getenv('TRADE_ANYTIME', 'True').lower() == 'true'

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
