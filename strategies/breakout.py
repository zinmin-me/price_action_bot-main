"""
Breakout Strategy
Implements trading based on price breakouts from consolidation patterns and key levels
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

class BreakoutStrategy:
    """Breakout trading strategy"""
    
    def __init__(self, mt5_connector):
        self.mt5 = mt5_connector
        self.name = "Breakout"
        self.magic_number = MAGIC_BREAKOUT
        self.enabled = BREAKOUT_ENABLED
        
        # Strategy parameters
        self.consolidation_periods = 20  # Periods to look for consolidation
        self.breakout_confirmation_bars = BREAKOUT_CONFIRMATION_BARS
        self.min_volume_multiplier = BREAKOUT_MIN_VOLUME
        self.breakout_threshold = 0.001  # 0.1% breakout threshold
        self.atr_period = 14
        self.atr_multiplier = 1.5  # ATR multiplier for stop loss
        
        # Level tracking
        self.consolidation_levels = {'high': 0, 'low': 0}
        self.last_breakout_check = None
        
    def analyze(self, df) -> Dict:
        """
        Analyze market for breakout opportunities
        
        Args:
            df: OHLC DataFrame
            
        Returns:
            Dict: Analysis results with signals
        """
        if len(df) < self.consolidation_periods + 10:
            return {'signal': 'no_signal', 'reason': 'Insufficient data'}
        
        try:
            # Calculate indicators
            atr = TechnicalIndicators.atr(df['high'], df['low'], df['close'], self.atr_period)
            current_price = df['close'].iloc[-1]
            current_atr = atr.iloc[-1]
            
            # Find consolidation patterns
            consolidation = self._find_consolidation(df)
            
            # Check for breakouts
            breakout_signal = self._check_breakout(df, consolidation, current_price, current_atr)
            
            # Generate signals
            signal = self._generate_signal(breakout_signal, current_price, current_atr)
            
            return {
                'signal': signal['type'],
                'direction': signal.get('direction'),
                'entry_price': signal.get('entry_price'),
                'stop_loss': signal.get('stop_loss'),
                'take_profit': signal.get('take_profit'),
                'confidence': signal.get('confidence'),
                'breakout_type': signal.get('breakout_type'),
                'consolidation_range': consolidation,
                'atr': current_atr,
                'breakout_signal': breakout_signal
            }
            
        except Exception as e:
            logger.error(f"Error in breakout analysis: {e}")
            return {'signal': 'error', 'reason': str(e)}
    
    def _find_consolidation(self, df) -> Dict:
        """
        Find consolidation patterns in recent price action
        
        Args:
            df: OHLC DataFrame
            
        Returns:
            Dict: Consolidation analysis results
        """
        if len(df) < self.consolidation_periods:
            return {'type': 'no_consolidation'}
        
        # Get recent data for consolidation analysis
        recent_data = df.tail(self.consolidation_periods)
        
        # Calculate consolidation metrics
        high_range = recent_data['high'].max() - recent_data['high'].min()
        low_range = recent_data['low'].max() - recent_data['low'].min()
        avg_range = (high_range + low_range) / 2
        
        # Calculate average true range for comparison
        atr = TechnicalIndicators.atr(recent_data['high'], recent_data['low'], recent_data['close'], 14)
        avg_atr = atr.mean()
        
        # Check if price is consolidating (low volatility)
        if avg_range < avg_atr * 1.5:  # Consolidation threshold
            # Find key levels
            resistance = recent_data['high'].max()
            support = recent_data['low'].min()
            
            # Count touches of key levels
            resistance_touches = sum(abs(recent_data['high'] - resistance) < avg_atr * 0.1)
            support_touches = sum(abs(recent_data['low'] - support) < avg_atr * 0.1)
            
            return {
                'type': 'consolidation',
                'resistance': resistance,
                'support': support,
                'range': resistance - support,
                'resistance_touches': resistance_touches,
                'support_touches': support_touches,
                'strength': min((resistance_touches + support_touches) * 10, 100)
            }
        
        return {'type': 'no_consolidation'}
    
    def _check_breakout(self, df, consolidation: Dict, 
                       current_price: float, atr: float) -> Dict:
        """
        Check for breakout signals
        
        Args:
            df: OHLC DataFrame
            consolidation: Consolidation analysis
            current_price: Current price
            atr: Current ATR value
            
        Returns:
            Dict: Breakout analysis results
        """
        if consolidation['type'] != 'consolidation':
            return {'type': 'no_breakout'}
        
        resistance = consolidation['resistance']
        support = consolidation['support']
        consolidation_range = consolidation['range']
        
        # Check for bullish breakout
        if current_price > resistance * (1 + self.breakout_threshold):
            # Confirm breakout with volume and momentum
            if self._confirm_breakout(df, 'bullish', resistance, atr):
                return {
                    'type': 'bullish_breakout',
                    'breakout_level': resistance,
                    'strength': consolidation['strength'],
                    'range': consolidation_range
                }
        
        # Check for bearish breakout
        elif current_price < support * (1 - self.breakout_threshold):
            # Confirm breakout with volume and momentum
            if self._confirm_breakout(df, 'bearish', support, atr):
                return {
                    'type': 'bearish_breakout',
                    'breakout_level': support,
                    'strength': consolidation['strength'],
                    'range': consolidation_range
                }
        
        return {'type': 'no_breakout'}
    
    def _confirm_breakout(self, df, direction: str, 
                         breakout_level: float, atr: float) -> bool:
        """
        Confirm breakout with additional criteria
        
        Args:
            df: OHLC DataFrame
            direction: 'bullish' or 'bearish'
            breakout_level: Level that was broken
            atr: Current ATR value
            
        Returns:
            bool: True if breakout is confirmed
        """
        if len(df) < self.breakout_confirmation_bars + 1:
            return False
        
        # Get recent candles for confirmation
        recent_candles = df.tail(self.breakout_confirmation_bars + 1)
        
        # Check for sustained breakout
        if direction == 'bullish':
            # Check if recent candles are above breakout level
            above_level = sum(recent_candles['close'] > breakout_level)
            if above_level < self.breakout_confirmation_bars:
                return False
            
            # Check for bullish momentum
            price_momentum = recent_candles['close'].iloc[-1] - recent_candles['close'].iloc[-2]
            if price_momentum <= 0:
                return False
        
        else:  # bearish
            # Check if recent candles are below breakout level
            below_level = sum(recent_candles['close'] < breakout_level)
            if below_level < self.breakout_confirmation_bars:
                return False
            
            # Check for bearish momentum
            price_momentum = recent_candles['close'].iloc[-1] - recent_candles['close'].iloc[-2]
            if price_momentum >= 0:
                return False
        
        # Check for volume confirmation (if available)
        if 'tick_volume' in df.columns:
            recent_volume = recent_candles['tick_volume'].mean()
            avg_volume = df['tick_volume'].tail(50).mean()
            if recent_volume < avg_volume * self.min_volume_multiplier:
                return False
        
        # Check for significant breakout (at least 0.5 ATR)
        breakout_distance = abs(recent_candles['close'].iloc[-1] - breakout_level)
        if breakout_distance < atr * 0.5:
            return False
        
        return True
    
    def _generate_signal(self, breakout_signal: Dict, current_price: float, atr: float) -> Dict:
        """
        Generate trading signal based on breakout analysis
        
        Args:
            breakout_signal: Breakout analysis results
            current_price: Current price
            atr: Current ATR value
            
        Returns:
            Dict: Trading signal
        """
        if breakout_signal['type'] == 'bullish_breakout':
            # Calculate stop loss and take profit
            breakout_level = breakout_signal['breakout_level']
            stop_loss = breakout_level - (atr * self.atr_multiplier)
            take_profit = current_price + (current_price - stop_loss) * 2  # 1:2 risk-reward
            
            return {
                'type': 'buy',
                'direction': 'long',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': self._calculate_confidence(breakout_signal, atr),
                'breakout_type': 'bullish_breakout'
            }
        
        elif breakout_signal['type'] == 'bearish_breakout':
            # Calculate stop loss and take profit
            breakout_level = breakout_signal['breakout_level']
            stop_loss = breakout_level + (atr * self.atr_multiplier)
            take_profit = current_price - (stop_loss - current_price) * 2  # 1:2 risk-reward
            
            return {
                'type': 'sell',
                'direction': 'short',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': self._calculate_confidence(breakout_signal, atr),
                'breakout_type': 'bearish_breakout'
            }
        
        return {'type': 'no_signal'}
    
    def _calculate_confidence(self, breakout_signal: Dict, atr: float) -> float:
        """
        Calculate signal confidence score
        
        Args:
            breakout_signal: Breakout analysis results
            atr: Current ATR value
            
        Returns:
            float: Confidence score (0-100)
        """
        confidence = 0
        
        # Consolidation strength (30 points)
        confidence += breakout_signal.get('strength', 0) * 0.3
        
        # Breakout distance (25 points)
        breakout_distance = abs(breakout_signal.get('range', 0))
        if breakout_distance > atr * 2:
            confidence += 25
        elif breakout_distance > atr:
            confidence += 15
        
        # Confirmation bars (20 points)
        confidence += 20  # Breakout was confirmed
        
        # Volume confirmation (15 points)
        confidence += 15  # Volume criteria met
        
        # Momentum confirmation (10 points)
        confidence += 10  # Momentum criteria met
        
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
        if len(df) < 10:
            return {'exit': False}
        
        try:
            current_price = df['close'].iloc[-1]
            atr = TechnicalIndicators.atr(df['high'], df['low'], df['close'], self.atr_period).iloc[-1]
            
            # Check for false breakout (price returns to consolidation)
            if position['type'] == 'buy':
                # Check if price has returned below breakout level
                entry_price = position['price_open']
                if current_price < entry_price * 0.998:  # 0.2% below entry
                    return {
                        'exit': True,
                        'reason': 'False breakout - price returned to consolidation',
                        'exit_price': current_price
                    }
                
                # Check for target reached
                if current_price > entry_price * 1.01:  # 1% profit
                    return {
                        'exit': True,
                        'reason': 'Target reached',
                        'exit_price': current_price
                    }
            
            elif position['type'] == 'sell':
                # Check if price has returned above breakout level
                entry_price = position['price_open']
                if current_price > entry_price * 1.002:  # 0.2% above entry
                    return {
                        'exit': True,
                        'reason': 'False breakout - price returned to consolidation',
                        'exit_price': current_price
                    }
                
                # Check for target reached
                if current_price < entry_price * 0.99:  # 1% profit
                    return {
                        'exit': True,
                        'reason': 'Target reached',
                        'exit_price': current_price
                    }
            
            return {'exit': False}
            
        except Exception as e:
            logger.error(f"Error in breakout exit analysis: {e}")
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
                'consolidation_periods': self.consolidation_periods,
                'breakout_confirmation_bars': self.breakout_confirmation_bars,
                'min_volume_multiplier': self.min_volume_multiplier,
                'breakout_threshold': self.breakout_threshold,
                'atr_period': self.atr_period,
                'atr_multiplier': self.atr_multiplier
            },
            'current_consolidation': self.consolidation_levels
        }
