"""
Utility functions for Price Action Trading Bot
Includes technical indicators, helper functions, and logging utilities
"""

import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError as e:
    import logging
    logging.warning(f"Pandas not available: {e}")
    PANDAS_AVAILABLE = False
    pd = None
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
try:
    import ta
    TA_AVAILABLE = True
except ImportError as e:
    import logging
    logging.warning(f"TA library not available: {e}")
    TA_AVAILABLE = False
    ta = None
from config import *

# Configure logging (minimal default; real config is done via LoggingUtils.setup_logging)
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Technical analysis indicators for price action trading"""
    
    @staticmethod
    def sma(data, period: int):
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data, period: int):
        """Exponential Moving Average"""
        return data.ewm(span=period).mean()
    
    @staticmethod
    def rsi(data, period: int = RSI_PERIOD):
        """Relative Strength Index"""
        return ta.momentum.RSIIndicator(data, window=period).rsi()
    
    @staticmethod
    def macd(data, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD indicator"""
        macd_indicator = ta.trend.MACD(data, window_fast=fast, window_slow=slow, window_sign=signal)
        return macd_indicator.macd(), macd_indicator.macd_signal(), macd_indicator.macd_diff()
    
    @staticmethod
    def bollinger_bands(data, period: int = 20, std_dev: float = 2):
        """Bollinger Bands"""
        bb_indicator = ta.volatility.BollingerBands(data, window=period, window_dev=std_dev)
        return bb_indicator.bollinger_hband(), bb_indicator.bollinger_lband(), bb_indicator.bollinger_mavg()
    
    @staticmethod
    def atr(high, low, close, period: int = 14):
        """Average True Range"""
        return ta.volatility.AverageTrueRange(high, low, close, window=period).average_true_range()
    
    @staticmethod
    def stochastic(high, low, close, 
                   k_period: int = 14, d_period: int = 3):
        """Stochastic Oscillator"""
        stoch_indicator = ta.momentum.StochasticOscillator(high, low, close, window=k_period, smooth_window=d_period)
        return stoch_indicator.stoch(), stoch_indicator.stoch_signal()

class PriceActionPatterns:
    """Price action pattern recognition"""
    
    @staticmethod
    def is_hammer(candle) -> bool:
        """
        Identify hammer candlestick pattern
        
        Args:
            candle: Single candle data (open, high, low, close)
            
        Returns:
            bool: True if hammer pattern detected
        """
        body_size = abs(candle['close'] - candle['open'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        total_range = candle['high'] - candle['low']
        
        # Hammer criteria: small body, long lower shadow, small upper shadow
        return (body_size < total_range * 0.3 and 
                lower_shadow > body_size * 2 and 
                upper_shadow < body_size * 0.5)
    
    @staticmethod
    def is_shooting_star(candle) -> bool:
        """
        Identify shooting star candlestick pattern
        
        Args:
            candle: Single candle data (open, high, low, close)
            
        Returns:
            bool: True if shooting star pattern detected
        """
        body_size = abs(candle['close'] - candle['open'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        total_range = candle['high'] - candle['low']
        
        # Shooting star criteria: small body, long upper shadow, small lower shadow
        return (body_size < total_range * 0.3 and 
                upper_shadow > body_size * 2 and 
                lower_shadow < body_size * 0.5)
    
    @staticmethod
    def is_engulfing(candles, index: int):
        """
        Identify engulfing pattern
        
        Args:
            candles: DataFrame with OHLC data
            index: Current candle index
            
        Returns:
            Tuple[bool, str]: (is_engulfing, pattern_type)
        """
        if index < 1:
            return False, ""
        
        current = candles.iloc[index]
        previous = candles.iloc[index - 1]
        
        current_body = abs(current['close'] - current['open'])
        previous_body = abs(previous['close'] - previous['open'])
        
        # Bullish engulfing
        if (previous['close'] < previous['open'] and  # Previous bearish
            current['close'] > current['open'] and    # Current bullish
            current['open'] < previous['close'] and   # Current opens below previous close
            current['close'] > previous['open']):     # Current closes above previous open
            return True, "bullish_engulfing"
        
        # Bearish engulfing
        if (previous['close'] > previous['open'] and  # Previous bullish
            current['close'] < current['open'] and    # Current bearish
            current['open'] > previous['close'] and   # Current opens above previous close
            current['close'] < previous['open']):     # Current closes below previous open
            return True, "bearish_engulfing"
        
        return False, ""
    
    @staticmethod
    def is_pin_bar(candle, atr: float):
        """
        Identify pin bar pattern
        
        Args:
            candle: Single candle data
            atr: Average True Range for context
            
        Returns:
            Tuple[bool, str]: (is_pin_bar, direction)
        """
        body_size = abs(candle['close'] - candle['open'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        total_range = candle['high'] - candle['low']
        
        # Pin bar criteria: small body, long shadow (at least 2/3 of total range)
        if body_size < total_range * 0.3:
            if lower_shadow > total_range * 0.6:
                return True, "bullish_pin"
            elif upper_shadow > total_range * 0.6:
                return True, "bearish_pin"
        
        return False, ""
    
    @staticmethod
    def is_doji(candle) -> bool:
        """
        Identify doji pattern
        
        Args:
            candle: Single candle data
            
        Returns:
            bool: True if doji pattern detected
        """
        body_size = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        
        # Doji criteria: very small body (less than 10% of total range)
        return body_size < total_range * 0.1

class SupportResistance:
    """Support and Resistance level identification"""
    
    @staticmethod
    def find_levels(df, lookback: int = SR_LOOKBACK_PERIODS, 
                   min_touches: int = 2) -> Tuple[List[float], List[float]]:
        """
        Find support and resistance levels
        
        Args:
            df: OHLC DataFrame
            lookback: Number of periods to look back
            min_touches: Minimum number of touches for a level
            
        Returns:
            Tuple[List[float], List[float]]: (support_levels, resistance_levels)
        """
        if len(df) < lookback:
            return [], []
        
        # Get recent data
        recent_data = df.tail(lookback)
        
        # Find local highs and lows
        highs = recent_data['high'].rolling(window=5, center=True).max() == recent_data['high']
        lows = recent_data['low'].rolling(window=5, center=True).min() == recent_data['low']
        
        resistance_candidates = recent_data[highs]['high'].tolist()
        support_candidates = recent_data[lows]['low'].tolist()
        
        # Group similar levels
        resistance_levels = SupportResistance._group_levels(resistance_candidates, min_touches)
        support_levels = SupportResistance._group_levels(support_candidates, min_touches)
        
        return support_levels, resistance_levels
    
    @staticmethod
    def _group_levels(levels: List[float], min_touches: int, tolerance: float = None) -> List[float]:
        """
        Group similar price levels
        
        Args:
            levels: List of price levels
            min_touches: Minimum touches required
            tolerance: Price tolerance for grouping
            
        Returns:
            List[float]: Grouped levels
        """
        if not levels:
            return []
        
        if tolerance is None:
            tolerance = SR_TOUCH_TOLERANCE * 0.0001  # Convert points to price
        
        grouped = []
        levels = sorted(levels)
        
        i = 0
        while i < len(levels):
            current_level = levels[i]
            group = [current_level]
            
            # Find all levels within tolerance
            j = i + 1
            while j < len(levels) and abs(levels[j] - current_level) <= tolerance:
                group.append(levels[j])
                j += 1
            
            # If group has enough touches, add to result
            if len(group) >= min_touches:
                grouped.append(sum(group) / len(group))  # Average of group
            
            i = j
        
        return grouped
    
    @staticmethod
    def is_near_level(price: float, levels: List[float], tolerance: float = None) -> Tuple[bool, float]:
        """
        Check if price is near a support/resistance level
        
        Args:
            price: Current price
            levels: List of levels
            tolerance: Price tolerance
            
        Returns:
            Tuple[bool, float]: (is_near, level_price)
        """
        if tolerance is None:
            tolerance = SR_TOUCH_TOLERANCE * 0.0001
        
        for level in levels:
            if abs(price - level) <= tolerance:
                return True, level
        
        return False, 0.0

class TrendAnalysis:
    """Trend analysis utilities"""
    
    @staticmethod
    def identify_trend(df, fast_period: int = MA_FAST_PERIOD, 
                      slow_period: int = MA_SLOW_PERIOD) -> str:
        """
        Identify current trend using moving averages
        
        Args:
            df: OHLC DataFrame
            fast_period: Fast MA period
            slow_period: Slow MA period
            
        Returns:
            str: 'uptrend', 'downtrend', or 'sideways'
        """
        if len(df) < slow_period:
            return 'sideways'
        
        # Calculate moving averages
        fast_ma = TechnicalIndicators.sma(df['close'], fast_period)
        slow_ma = TechnicalIndicators.sma(df['close'], slow_period)
        
        current_fast = fast_ma.iloc[-1]
        current_slow = slow_ma.iloc[-1]
        previous_fast = fast_ma.iloc[-2]
        previous_slow = slow_ma.iloc[-2]
        
        # Determine trend
        if current_fast > current_slow and previous_fast > previous_slow:
            return 'uptrend'
        elif current_fast < current_slow and previous_fast < previous_slow:
            return 'downtrend'
        else:
            return 'sideways'
    
    @staticmethod
    def find_higher_highs_lower_lows(df, lookback: int = 20):
        """
        Find higher highs and lower lows pattern
        
        Args:
            df: OHLC DataFrame
            lookback: Number of periods to analyze
            
        Returns:
            Tuple[bool, bool]: (higher_highs, lower_lows)
        """
        if len(df) < lookback:
            return False, False
        
        recent_data = df.tail(lookback)
        
        # Find local highs and lows
        highs = recent_data['high'].rolling(window=3, center=True).max() == recent_data['high']
        lows = recent_data['low'].rolling(window=3, center=True).min() == recent_data['low']
        
        high_prices = recent_data[highs]['high'].tolist()
        low_prices = recent_data[lows]['low'].tolist()
        
        # Check for higher highs
        higher_highs = len(high_prices) >= 2 and high_prices[-1] > high_prices[-2]
        
        # Check for lower lows
        lower_lows = len(low_prices) >= 2 and low_prices[-1] < low_prices[-2]
        
        return higher_highs, lower_lows

class RiskManagement:
    """Risk management utilities"""
    
    @staticmethod
    def calculate_position_size(account_balance: float, risk_percentage: float, 
                              stop_loss_points: int, point_value: float) -> float:
        """
        Calculate position size based on risk management
        
        Args:
            account_balance: Account balance
            risk_percentage: Risk percentage per trade
            stop_loss_points: Stop loss in points
            point_value: Point value
            
        Returns:
            float: Calculated position size
        """
        risk_amount = account_balance * (risk_percentage / 100)
        position_size = risk_amount / (stop_loss_points * point_value)
        return max(0.01, min(position_size, 1.0))  # Limit between 0.01 and 1.0
    
    @staticmethod
    def calculate_stop_loss(entry_price: float, direction: str, atr: float, 
                          atr_multiplier: float = 2.0) -> float:
        """
        Calculate stop loss using ATR
        
        Args:
            entry_price: Entry price
            direction: 'buy' or 'sell'
            atr: Average True Range
            atr_multiplier: ATR multiplier
            
        Returns:
            float: Stop loss price
        """
        if direction.lower() == 'buy':
            return entry_price - (atr * atr_multiplier)
        else:
            return entry_price + (atr * atr_multiplier)
    
    @staticmethod
    def calculate_take_profit(entry_price: float, stop_loss: float, 
                            direction: str, risk_reward_ratio: float = 2.0) -> float:
        """
        Calculate take profit based on risk-reward ratio
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            direction: 'buy' or 'sell'
            risk_reward_ratio: Risk to reward ratio
            
        Returns:
            float: Take profit price
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward_ratio
        
        if direction.lower() == 'buy':
            return entry_price + reward
        else:
            return entry_price - reward

class LoggingUtils:
    """Logging utilities for trading bot"""
    
    @staticmethod
    def setup_logging(log_file: str = LOG_FILE, level: str = LOG_LEVEL):
        """
        Setup logging configuration
        
        Args:
            log_file: Log file path
            level: Logging level
        """
        import os
        # Ensure log directory exists
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
        except Exception:
            pass
        # Force reconfigure root logger with both file and console handlers
        logging.basicConfig(
            level=getattr(logging, level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True,
        )
    
    @staticmethod
    def log_trade(trade_data: Dict):
        """
        Log trade information
        
        Args:
            trade_data: Dictionary containing trade information
        """
        logger.info(f"Trade executed: {trade_data}")
    
    @staticmethod
    def log_strategy_signal(strategy: str, signal: str, symbol: str, price: float):
        """
        Log strategy signal
        
        Args:
            strategy: Strategy name
            signal: Signal type
            symbol: Trading symbol
            price: Current price
        """
        logger.info(f"{strategy} signal: {signal} for {symbol} at {price}")

class DataValidation:
    """Data validation utilities"""
    
    @staticmethod
    def validate_ohlc_data(df) -> bool:
        """
        Validate OHLC data integrity
        
        Args:
            df: OHLC DataFrame
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        if df.empty:
            return False
        
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            return False
        
        # Check for missing values
        if df[required_columns].isnull().any().any():
            return False
        
        # Check OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        if invalid_ohlc.any():
            return False
        
        return True
    
    @staticmethod
    def clean_data(df):
        """
        Clean and prepare data for analysis
        
        Args:
            df: Raw OHLC DataFrame
            
        Returns:
            DataFrame: Cleaned DataFrame
        """
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Sort by time
        df = df.sort_index()
        
        # Remove outliers (prices that are too far from previous close)
        if len(df) > 1:
            price_changes = df['close'].pct_change().abs()
            outlier_threshold = price_changes.quantile(0.99)  # Remove top 1% outliers
            df = df[price_changes <= outlier_threshold]
        
        return df
