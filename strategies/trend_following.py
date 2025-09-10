"""
Trend Following Strategy with Pullbacks
Implements higher high/lower low trend following with pullback entries
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
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

from config import *
from utils import TechnicalIndicators, TrendAnalysis, RiskManagement, LoggingUtils

logger = logging.getLogger(__name__)

class TrendFollowingStrategy:
    """Trend following strategy with pullback entries"""
    
    def __init__(self, mt5_connector):
        self.mt5 = mt5_connector
        self.name = "Trend Following"
        self.magic_number = MAGIC_TREND_FOLLOWING
        self.enabled = TREND_FOLLOWING_ENABLED
        
        # Strategy parameters
        self.fast_ma_period = MA_FAST_PERIOD
        self.slow_ma_period = MA_SLOW_PERIOD
        self.rsi_period = RSI_PERIOD
        self.rsi_oversold = RSI_OVERSOLD
        self.rsi_overbought = RSI_OVERBOUGHT
        self.pullback_threshold = 0.3  # 30% pullback from recent high/low
        self.min_trend_bars = 5  # Minimum bars for trend confirmation
        
    def analyze(self, df) -> Dict:
        """
        Analyze market for trend following opportunities
        
        Args:
            df: OHLC DataFrame
            
        Returns:
            Dict: Analysis results with signals
        """
        # Require enough bars to compute all indicators and swing structure safely
        min_required = max(self.slow_ma_period + 10, 50)
        if len(df) < min_required:
            return {'signal': 'no_signal', 'reason': f'Insufficient data (<{min_required} bars)'}
        
        try:
            # Calculate indicators
            fast_ma = TechnicalIndicators.sma(df['close'], self.fast_ma_period)
            slow_ma = TechnicalIndicators.sma(df['close'], self.slow_ma_period)
            rsi = TechnicalIndicators.rsi(df['close'], self.rsi_period)
            # Guard against NaNs or insufficient computed points
            if PANDAS_AVAILABLE:
                if (
                    len(fast_ma) == 0 or len(slow_ma) == 0 or len(rsi) == 0 or
                    pd.isna(fast_ma.iloc[-1]) or pd.isna(slow_ma.iloc[-1]) or pd.isna(rsi.iloc[-1])
                ):
                    return {'signal': 'no_signal', 'reason': 'Indicators not ready'}
            else:
                if len(fast_ma) == 0 or len(slow_ma) == 0 or len(rsi) == 0:
                    return {'signal': 'no_signal', 'reason': 'Indicators not ready'}
            
            # Get current values
            current_price = df['close'].iloc[-1]
            current_fast_ma = fast_ma.iloc[-1]
            current_slow_ma = slow_ma.iloc[-1]
            current_rsi = rsi.iloc[-1]
            
            # Identify trend
            try:
                trend = TrendAnalysis.identify_trend(df, self.fast_ma_period, self.slow_ma_period)
            except Exception:
                return {'signal': 'no_signal', 'reason': 'Trend calc failed'}
            
            # Check for pullback conditions
            pullback_signal = self._check_pullback(df, trend, current_price, current_fast_ma, current_slow_ma)
            
            # Check for higher highs/lower lows (guard against short series)
            try:
                higher_highs, lower_lows = TrendAnalysis.find_higher_highs_lower_lows(df)
            except Exception:
                higher_highs, lower_lows = False, False
            
            # Generate signals
            signal = self._generate_signal(
                trend, pullback_signal, higher_highs, lower_lows,
                current_rsi, current_price, current_fast_ma, current_slow_ma,
                df
            )
            
            return {
                'signal': signal['type'],
                'direction': signal.get('direction'),
                'entry_price': signal.get('entry_price'),
                'stop_loss': signal.get('stop_loss'),
                'take_profit': signal.get('take_profit'),
                'confidence': signal.get('confidence'),
                'trend': trend,
                'rsi': current_rsi,
                'fast_ma': current_fast_ma,
                'slow_ma': current_slow_ma,
                'higher_highs': higher_highs,
                'lower_lows': lower_lows,
                'pullback_signal': pullback_signal
            }
            
        except Exception as e:
            logger.error(f"Error in trend following analysis: {e}")
            return {'signal': 'error', 'reason': str(e)}
    
    def _check_pullback(self, df, trend: str, current_price: float, 
                       fast_ma: float, slow_ma: float) -> Dict:
        """
        Check for pullback conditions
        
        Args:
            df: OHLC DataFrame
            trend: Current trend direction
            current_price: Current price
            fast_ma: Fast moving average
            slow_ma: Slow moving average
            
        Returns:
            Dict: Pullback analysis results
        """
        if trend == 'uptrend':
            # Look for pullback to fast MA in uptrend
            recent_high = df['high'].tail(20).max()
            pullback_depth = (recent_high - current_price) / (recent_high - slow_ma)
            
            if (current_price <= fast_ma * 1.001 and  # Near fast MA
                pullback_depth >= self.pullback_threshold and  # Significant pullback
                current_price > slow_ma):  # Still above slow MA
                return {
                    'type': 'bullish_pullback',
                    'depth': pullback_depth,
                    'ma_distance': (current_price - fast_ma) / fast_ma
                }
        
        elif trend == 'downtrend':
            # Look for pullback to fast MA in downtrend
            recent_low = df['low'].tail(20).min()
            pullback_depth = (current_price - recent_low) / (slow_ma - recent_low)
            
            if (current_price >= fast_ma * 0.999 and  # Near fast MA
                pullback_depth >= self.pullback_threshold and  # Significant pullback
                current_price < slow_ma):  # Still below slow MA
                return {
                    'type': 'bearish_pullback',
                    'depth': pullback_depth,
                    'ma_distance': (fast_ma - current_price) / fast_ma
                }
        
        return {'type': 'no_pullback'}
    
    def _generate_signal(self, trend: str, pullback_signal: Dict, higher_highs: bool,
                        lower_lows: bool, rsi: float, current_price: float,
                        fast_ma: float, slow_ma: float, df) -> Dict:
        """
        Generate trading signal based on analysis
        
        Args:
            trend: Current trend
            pullback_signal: Pullback analysis
            higher_highs: Higher highs detected
            lower_lows: Lower lows detected
            rsi: Current RSI value
            current_price: Current price
            fast_ma: Fast MA value
            slow_ma: Slow MA value
            df: OHLC DataFrame used for analysis
            
        Returns:
            Dict: Trading signal
        """
        # Uptrend signals
        if (trend == 'uptrend' and 
            pullback_signal['type'] == 'bullish_pullback' and
            higher_highs and
            rsi < self.rsi_overbought and
            rsi > 30):  # Not oversold
            
            # Calculate stop loss and take profit
            atr_period = 14
            if PANDAS_AVAILABLE and len(df) >= atr_period:
                try:
                    atr_series = TechnicalIndicators.atr(
                        df['high'].tail(atr_period * 3),
                        df['low'].tail(atr_period * 3),
                        df['close'].tail(atr_period * 3),
                        period=atr_period,
                    )
                    atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else current_price * 0.01
                except Exception:
                    atr = current_price * 0.01
            else:
                atr = current_price * 0.01  # Fallback when insufficient history
            
            stop_loss = RiskManagement.calculate_stop_loss(current_price, 'buy', atr, 1.5)
            take_profit = RiskManagement.calculate_take_profit(current_price, stop_loss, 'buy', 2.0)
            
            return {
                'type': 'buy',
                'direction': 'long',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': self._calculate_confidence(trend, pullback_signal, higher_highs, rsi)
            }
        
        # Downtrend signals
        elif (trend == 'downtrend' and 
              pullback_signal['type'] == 'bearish_pullback' and
              lower_lows and
              rsi > self.rsi_oversold and
              rsi < 70):  # Not overbought
            
            # Calculate stop loss and take profit
            atr_period = 14
            if PANDAS_AVAILABLE and len(df) >= atr_period:
                try:
                    atr_series = TechnicalIndicators.atr(
                        df['high'].tail(atr_period * 3),
                        df['low'].tail(atr_period * 3),
                        df['close'].tail(atr_period * 3),
                        period=atr_period,
                    )
                    atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else current_price * 0.01
                except Exception:
                    atr = current_price * 0.01
            else:
                atr = current_price * 0.01  # Fallback when insufficient history
            
            stop_loss = RiskManagement.calculate_stop_loss(current_price, 'sell', atr, 1.5)
            take_profit = RiskManagement.calculate_take_profit(current_price, stop_loss, 'sell', 2.0)
            
            return {
                'type': 'sell',
                'direction': 'short',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': self._calculate_confidence(trend, pullback_signal, lower_lows, rsi)
            }
        
        return {'type': 'no_signal'}
    
    def _calculate_confidence(self, trend: str, pullback_signal: Dict, 
                            trend_confirmation: bool, rsi: float) -> float:
        """
        Calculate signal confidence score
        
        Args:
            trend: Current trend
            pullback_signal: Pullback analysis
            trend_confirmation: Trend confirmation (higher highs/lower lows)
            rsi: Current RSI value
            
        Returns:
            float: Confidence score (0-100)
        """
        confidence = 0
        
        # Trend strength (30 points)
        if trend in ['uptrend', 'downtrend']:
            confidence += 30
        
        # Pullback quality (25 points)
        if pullback_signal['type'] != 'no_pullback':
            confidence += 25
            if pullback_signal.get('depth', 0) > 0.5:  # Deep pullback
                confidence += 10
        
        # Trend confirmation (20 points)
        if trend_confirmation:
            confidence += 20
        
        # RSI alignment (15 points)
        if trend == 'uptrend' and 30 < rsi < 70:
            confidence += 15
        elif trend == 'downtrend' and 30 < rsi < 70:
            confidence += 15
        
        # MA alignment (10 points)
        confidence += 10
        
        return min(confidence, 100)
    
    def should_exit_position(self, position: Dict, df) -> Dict:
        """
        Check if position should be exited
        
        Args:
            position: Current position data
            df: OHLC DataFrame
            
        Returns:
            Dict: Exit signal
        """
        if len(df) < self.slow_ma_period:
            return {'exit': False}
        
        try:
            # Calculate indicators
            fast_ma = TechnicalIndicators.sma(df['close'], self.fast_ma_period)
            slow_ma = TechnicalIndicators.sma(df['close'], self.slow_ma_period)
            rsi = TechnicalIndicators.rsi(df['close'], self.rsi_period)
            
            current_price = df['close'].iloc[-1]
            current_fast_ma = fast_ma.iloc[-1]
            current_slow_ma = slow_ma.iloc[-1]
            current_rsi = rsi.iloc[-1]
            
            # Check trend change
            trend = TrendAnalysis.identify_trend(df, self.fast_ma_period, self.slow_ma_period)
            
            # Exit conditions
            if position['type'] == 'buy':
                # Exit long position if trend changes to downtrend
                if trend == 'downtrend' and current_price < current_slow_ma:
                    return {
                        'exit': True,
                        'reason': 'Trend reversal to downtrend',
                        'exit_price': current_price
                    }
                
                # Exit if RSI becomes overbought
                if current_rsi > self.rsi_overbought:
                    return {
                        'exit': True,
                        'reason': 'RSI overbought',
                        'exit_price': current_price
                    }
            
            elif position['type'] == 'sell':
                # Exit short position if trend changes to uptrend
                if trend == 'uptrend' and current_price > current_slow_ma:
                    return {
                        'exit': True,
                        'reason': 'Trend reversal to uptrend',
                        'exit_price': current_price
                    }
                
                # Exit if RSI becomes oversold
                if current_rsi < self.rsi_oversold:
                    return {
                        'exit': True,
                        'reason': 'RSI oversold',
                        'exit_price': current_price
                    }
            
            return {'exit': False}
            
        except Exception as e:
            logger.error(f"Error in trend following exit analysis: {e}")
            return {'exit': False, 'error': str(e)}
    
    def get_strategy_info(self) -> Dict:
        """
        Get strategy information
        
        Returns:
            Dict: Strategy information
        """
        return {
            'name': self.name,
            'enabled': self.enabled,
            'magic_number': self.magic_number,
            'parameters': {
                'fast_ma_period': self.fast_ma_period,
                'slow_ma_period': self.slow_ma_period,
                'rsi_period': self.rsi_period,
                'pullback_threshold': self.pullback_threshold,
                'min_trend_bars': self.min_trend_bars
            }
        }
