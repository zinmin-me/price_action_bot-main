"""
Support & Resistance Bounce Strategy
Implements trading based on price bounces from key support and resistance levels
"""

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError as e:
    import logging
    logging.warning(f"Pandas not available: {e}")
    PANDAS_AVAILABLE = False
    pd = None
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

from config import *
from utils import TechnicalIndicators, SupportResistance, RiskManagement, LoggingUtils

logger = logging.getLogger(__name__)

class SupportResistanceStrategy:
    """Support and Resistance bounce strategy"""
    
    def __init__(self, mt5_connector):
        self.mt5 = mt5_connector
        self.name = "Support & Resistance"
        self.magic_number = MAGIC_SUPPORT_RESISTANCE
        self.enabled = SUPPORT_RESISTANCE_ENABLED
        
        # Strategy parameters
        self.lookback_periods = SR_LOOKBACK_PERIODS
        self.touch_tolerance = SR_TOUCH_TOLERANCE
        self.min_touches = 2
        self.bounce_confirmation_bars = 2
        self.rsi_period = RSI_PERIOD
        self.rsi_oversold = RSI_OVERSOLD
        self.rsi_overbought = RSI_OVERBOUGHT
        
        # Level tracking
        self.support_levels = []
        self.resistance_levels = []
        self.last_level_update = None
        
    def analyze(self, df) -> Dict:
        """
        Analyze market for support/resistance bounce opportunities
        
        Args:
            df: OHLC DataFrame
            
        Returns:
            Dict: Analysis results with signals
        """
        if len(df) < self.lookback_periods:
            return {'signal': 'no_signal', 'reason': 'Insufficient data'}
        
        try:
            # Update support/resistance levels
            self._update_levels(df)
            
            # Calculate indicators
            rsi = TechnicalIndicators.rsi(df['close'], self.rsi_period)
            current_price = df['close'].iloc[-1]
            current_rsi = rsi.iloc[-1]
            
            # Check for bounce signals
            bounce_signal = self._check_bounce(df, current_price, current_rsi)
            
            # Generate signals
            signal = self._generate_signal(bounce_signal, current_price, current_rsi)
            
            return {
                'signal': signal['type'],
                'direction': signal.get('direction'),
                'entry_price': signal.get('entry_price'),
                'stop_loss': signal.get('stop_loss'),
                'take_profit': signal.get('take_profit'),
                'confidence': signal.get('confidence'),
                'level_price': signal.get('level_price'),
                'level_type': signal.get('level_type'),
                'rsi': current_rsi,
                'support_levels': self.support_levels,
                'resistance_levels': self.resistance_levels,
                'bounce_signal': bounce_signal
            }
            
        except Exception as e:
            logger.error(f"Error in support/resistance analysis: {e}")
            return {'signal': 'error', 'reason': str(e)}
    
    def _update_levels(self, df):
        """
        Update support and resistance levels
        
        Args:
            df: OHLC DataFrame
        """
        # Update levels every 10 bars or if levels are empty
        if (self.last_level_update is None or 
            len(df) - self.last_level_update > 10 or 
            not self.support_levels or 
            not self.resistance_levels):
            
            self.support_levels, self.resistance_levels = SupportResistance.find_levels(
                df, self.lookback_periods, self.min_touches
            )
            self.last_level_update = len(df)
            
            logger.info(f"Updated levels - Support: {len(self.support_levels)}, Resistance: {len(self.resistance_levels)}")
    
    def _check_bounce(self, df, current_price: float, rsi: float) -> Dict:
        """
        Check for bounce signals from support/resistance levels
        
        Args:
            df: OHLC DataFrame
            current_price: Current price
            rsi: Current RSI value
            
        Returns:
            Dict: Bounce analysis results
        """
        # Check support bounce
        for support_level in self.support_levels:
            is_near, level_price = SupportResistance.is_near_level(
                current_price, [support_level], self.touch_tolerance * 0.0001
            )
            
            if is_near:
                # Check for bullish reversal pattern
                if self._is_bullish_reversal(df, level_price, rsi):
                    return {
                        'type': 'support_bounce',
                        'level_price': level_price,
                        'level_type': 'support',
                        'strength': self._calculate_level_strength(df, level_price, 'support')
                    }
        
        # Check resistance bounce
        for resistance_level in self.resistance_levels:
            is_near, level_price = SupportResistance.is_near_level(
                current_price, [resistance_level], self.touch_tolerance * 0.0001
            )
            
            if is_near:
                # Check for bearish reversal pattern
                if self._is_bearish_reversal(df, level_price, rsi):
                    return {
                        'type': 'resistance_bounce',
                        'level_price': level_price,
                        'level_type': 'resistance',
                        'strength': self._calculate_level_strength(df, level_price, 'resistance')
                    }
        
        return {'type': 'no_bounce'}
    
    def _is_bullish_reversal(self, df, level_price: float, rsi: float) -> bool:
        """
        Check for bullish reversal pattern at support level
        
        Args:
            df: OHLC DataFrame
            level_price: Support level price
            rsi: Current RSI value
            
        Returns:
            bool: True if bullish reversal detected
        """
        if len(df) < self.bounce_confirmation_bars + 1:
            return False
        
        # Check recent price action
        recent_candles = df.tail(self.bounce_confirmation_bars + 1)
        
        # Price should be near or below support level
        if recent_candles['low'].iloc[-1] > level_price * 1.001:
            return False
        
        # Check for bullish reversal patterns
        # 1. Hammer or doji at support
        last_candle = recent_candles.iloc[-1]
        if self._is_hammer(last_candle) or self._is_doji(last_candle):
            return True
        
        # 2. Bullish engulfing
        if len(recent_candles) >= 2:
            if self._is_bullish_engulfing(recent_candles.iloc[-2:]):
                return True
        
        # 3. RSI divergence (oversold and starting to turn up)
        if rsi < self.rsi_oversold and rsi > 25:
            return True
        
        # 4. Price rejection at support (long lower wick)
        lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
        body_size = abs(last_candle['close'] - last_candle['open'])
        if lower_wick > body_size * 2:
            return True
        
        return False
    
    def _is_bearish_reversal(self, df, level_price: float, rsi: float) -> bool:
        """
        Check for bearish reversal pattern at resistance level
        
        Args:
            df: OHLC DataFrame
            level_price: Resistance level price
            rsi: Current RSI value
            
        Returns:
            bool: True if bearish reversal detected
        """
        if len(df) < self.bounce_confirmation_bars + 1:
            return False
        
        # Check recent price action
        recent_candles = df.tail(self.bounce_confirmation_bars + 1)
        
        # Price should be near or above resistance level
        if recent_candles['high'].iloc[-1] < level_price * 0.999:
            return False
        
        # Check for bearish reversal patterns
        # 1. Shooting star or doji at resistance
        last_candle = recent_candles.iloc[-1]
        if self._is_shooting_star(last_candle) or self._is_doji(last_candle):
            return True
        
        # 2. Bearish engulfing
        if len(recent_candles) >= 2:
            if self._is_bearish_engulfing(recent_candles.iloc[-2:]):
                return True
        
        # 3. RSI divergence (overbought and starting to turn down)
        if rsi > self.rsi_overbought and rsi < 75:
            return True
        
        # 4. Price rejection at resistance (long upper wick)
        upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        body_size = abs(last_candle['close'] - last_candle['open'])
        if upper_wick > body_size * 2:
            return True
        
        return False
    
    def _is_hammer(self, candle) -> bool:
        """Check if candle is a hammer pattern"""
        body_size = abs(candle['close'] - candle['open'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        total_range = candle['high'] - candle['low']
        
        return (body_size < total_range * 0.3 and 
                lower_shadow > body_size * 2 and 
                upper_shadow < body_size * 0.5)
    
    def _is_shooting_star(self, candle) -> bool:
        """Check if candle is a shooting star pattern"""
        body_size = abs(candle['close'] - candle['open'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        total_range = candle['high'] - candle['low']
        
        return (body_size < total_range * 0.3 and 
                upper_shadow > body_size * 2 and 
                lower_shadow < body_size * 0.5)
    
    def _is_doji(self, candle) -> bool:
        """Check if candle is a doji pattern"""
        body_size = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        return body_size < total_range * 0.1
    
    def _is_bullish_engulfing(self, candles) -> bool:
        """Check for bullish engulfing pattern"""
        if len(candles) < 2:
            return False
        
        prev = candles.iloc[-2]
        curr = candles.iloc[-1]
        
        return (prev['close'] < prev['open'] and  # Previous bearish
                curr['close'] > curr['open'] and    # Current bullish
                curr['open'] < prev['close'] and   # Current opens below previous close
                curr['close'] > prev['open'])       # Current closes above previous open
    
    def _is_bearish_engulfing(self, candles) -> bool:
        """Check for bearish engulfing pattern"""
        if len(candles) < 2:
            return False
        
        prev = candles.iloc[-2]
        curr = candles.iloc[-1]
        
        return (prev['close'] > prev['open'] and  # Previous bullish
                curr['close'] < curr['open'] and    # Current bearish
                curr['open'] > prev['close'] and   # Current opens above previous close
                curr['close'] < prev['open'])       # Current closes below previous open
    
    def _calculate_level_strength(self, df, level_price: float, level_type: str) -> float:
        """
        Calculate the strength of a support/resistance level
        
        Args:
            df: OHLC DataFrame
            level_price: Level price
            level_type: 'support' or 'resistance'
            
        Returns:
            float: Level strength (0-100)
        """
        # Count touches within tolerance
        tolerance = self.touch_tolerance * 0.0001
        touches = 0
        
        for _, row in df.iterrows():
            if level_type == 'support':
                if abs(row['low'] - level_price) <= tolerance:
                    touches += 1
            else:  # resistance
                if abs(row['high'] - level_price) <= tolerance:
                    touches += 1
        
        # Calculate strength based on touches and recency
        strength = min(touches * 20, 80)  # Max 80 points for touches
        
        # Add recency bonus (more recent touches are stronger)
        recent_touches = 0
        for _, row in df.tail(20).iterrows():
            if level_type == 'support':
                if abs(row['low'] - level_price) <= tolerance:
                    recent_touches += 1
            else:
                if abs(row['high'] - level_price) <= tolerance:
                    recent_touches += 1
        
        strength += min(recent_touches * 5, 20)  # Max 20 points for recency
        
        return min(strength, 100)
    
    def _generate_signal(self, bounce_signal: Dict, current_price: float, rsi: float) -> Dict:
        """
        Generate trading signal based on bounce analysis
        
        Args:
            bounce_signal: Bounce analysis results
            current_price: Current price
            rsi: Current RSI value
            
        Returns:
            Dict: Trading signal
        """
        if bounce_signal['type'] == 'support_bounce':
            # Calculate stop loss and take profit
            level_price = bounce_signal['level_price']
            stop_loss = level_price * 0.999  # Below support level
            take_profit = current_price + (current_price - stop_loss) * 2  # 1:2 risk-reward
            
            return {
                'type': 'buy',
                'direction': 'long',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': self._calculate_confidence(bounce_signal, rsi, 'support'),
                'level_price': level_price,
                'level_type': 'support'
            }
        
        elif bounce_signal['type'] == 'resistance_bounce':
            # Calculate stop loss and take profit
            level_price = bounce_signal['level_price']
            stop_loss = level_price * 1.001  # Above resistance level
            take_profit = current_price - (stop_loss - current_price) * 2  # 1:2 risk-reward
            
            return {
                'type': 'sell',
                'direction': 'short',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': self._calculate_confidence(bounce_signal, rsi, 'resistance'),
                'level_price': level_price,
                'level_type': 'resistance'
            }
        
        return {'type': 'no_signal'}
    
    def _calculate_confidence(self, bounce_signal: Dict, rsi: float, level_type: str) -> float:
        """
        Calculate signal confidence score
        
        Args:
            bounce_signal: Bounce analysis results
            rsi: Current RSI value
            level_type: Level type ('support' or 'resistance')
            
        Returns:
            float: Confidence score (0-100)
        """
        confidence = 0
        
        # Level strength (40 points)
        confidence += bounce_signal.get('strength', 0) * 0.4
        
        # RSI alignment (30 points)
        if level_type == 'support':
            if self.rsi_oversold < rsi < 50:  # Oversold but not extreme
                confidence += 30
        else:  # resistance
            if 50 < rsi < self.rsi_overbought:  # Overbought but not extreme
                confidence += 30
        
        # Pattern confirmation (20 points)
        confidence += 20  # Basic pattern detected
        
        # Level recency (10 points)
        confidence += 10  # Level was recently updated
        
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
        if len(df) < 5:
            return {'exit': False}
        
        try:
            current_price = df['close'].iloc[-1]
            
            # Update levels
            self._update_levels(df)
            
            # Check if price has moved significantly away from the level
            if position['type'] == 'buy':
                # Check if price has moved significantly above entry
                if current_price > position['price_open'] * 1.005:  # 0.5% profit
                    return {
                        'exit': True,
                        'reason': 'Target reached',
                        'exit_price': current_price
                    }
                
                # Check if support level is broken
                for support_level in self.support_levels:
                    if current_price < support_level * 0.998:  # Below support
                        return {
                            'exit': True,
                            'reason': 'Support broken',
                            'exit_price': current_price
                        }
            
            elif position['type'] == 'sell':
                # Check if price has moved significantly below entry
                if current_price < position['price_open'] * 0.995:  # 0.5% profit
                    return {
                        'exit': True,
                        'reason': 'Target reached',
                        'exit_price': current_price
                    }
                
                # Check if resistance level is broken
                for resistance_level in self.resistance_levels:
                    if current_price > resistance_level * 1.002:  # Above resistance
                        return {
                            'exit': True,
                            'reason': 'Resistance broken',
                            'exit_price': current_price
                        }
            
            return {'exit': False}
            
        except Exception as e:
            logger.error(f"Error in support/resistance exit analysis: {e}")
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
                'lookback_periods': self.lookback_periods,
                'touch_tolerance': self.touch_tolerance,
                'min_touches': self.min_touches,
                'bounce_confirmation_bars': self.bounce_confirmation_bars,
                'rsi_period': self.rsi_period
            },
            'current_levels': {
                'support_levels': self.support_levels,
                'resistance_levels': self.resistance_levels
            }
        }
