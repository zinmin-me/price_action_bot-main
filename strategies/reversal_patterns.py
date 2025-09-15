"""
Candlestick Reversal Patterns Strategy
Implements trading based on candlestick reversal patterns (pin bar, engulfing, hammer, etc.)
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
from utils import TechnicalIndicators, PriceActionPatterns, RiskManagement, LoggingUtils

logger = logging.getLogger(__name__)

class ReversalPatternsStrategy:
    """Candlestick reversal patterns strategy"""
    
    def __init__(self, mt5_connector):
        self.mt5 = mt5_connector
        self.name = "Reversal Patterns"
        self.magic_number = MAGIC_REVERSAL_PATTERNS
        self.enabled = REVERSAL_PATTERNS_ENABLED
        
        # Strategy parameters
        self.rsi_period = RSI_PERIOD
        self.rsi_oversold = RSI_OVERSOLD
        self.rsi_overbought = RSI_OVERBOUGHT
        self.atr_period = 14
        self.atr_multiplier = 2.0  # ATR multiplier for stop loss
        self.min_body_ratio = 0.3  # Minimum body size ratio for patterns
        self.min_shadow_ratio = 2.0  # Minimum shadow to body ratio
        
        # Pattern weights
        self.pattern_weights = {
            'hammer': 0.8,
            'shooting_star': 0.8,
            'engulfing': 0.9,
            'pin_bar': 0.7,
            'doji': 0.6
        }
        
    def analyze(self, df) -> Dict:
        """
        Analyze market for reversal pattern opportunities
        
        Args:
            df: OHLC DataFrame
            
        Returns:
            Dict: Analysis results with signals
        """
        if len(df) < 20:
            return {'signal': 'no_signal', 'reason': 'Insufficient data'}
        
        try:
            # Calculate indicators
            rsi = TechnicalIndicators.rsi(df['close'], self.rsi_period)
            atr = TechnicalIndicators.atr(df['high'], df['low'], df['close'], self.atr_period)
            
            current_price = df['close'].iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_atr = atr.iloc[-1]
            
            # Detect reversal patterns
            patterns = self._detect_patterns(df)
            
            # Check for confluence with indicators
            confluence = self._check_confluence(patterns, current_rsi, df)
            
            # Generate signals
            signal = self._generate_signal(patterns, confluence, current_price, current_atr, current_rsi)
            
            return {
                'signal': signal['type'],
                'direction': signal.get('direction'),
                'entry_price': signal.get('entry_price'),
                'stop_loss': signal.get('stop_loss'),
                'take_profit': signal.get('take_profit'),
                'confidence': signal.get('confidence'),
                'pattern_type': signal.get('pattern_type'),
                'patterns_detected': patterns,
                'confluence': confluence,
                'rsi': current_rsi,
                'atr': current_atr
            }
            
        except Exception as e:
            logger.error(f"Error in reversal patterns analysis: {e}")
            return {'signal': 'error', 'reason': str(e)}
    
    def _detect_patterns(self, df) -> List[Dict]:
        """
        Detect candlestick reversal patterns
        
        Args:
            df: OHLC DataFrame
            
        Returns:
            List[Dict]: List of detected patterns
        """
        patterns = []
        
        if len(df) < 3:
            return patterns
        
        # Get last few candles for pattern detection
        recent_candles = df.tail(5)
        
        # Check individual patterns
        for i in range(len(recent_candles)):
            candle = recent_candles.iloc[i]
            candle_index = len(df) - len(recent_candles) + i
            
            # Hammer pattern
            if PriceActionPatterns.is_hammer(candle):
                patterns.append({
                    'type': 'hammer',
                    'direction': 'bullish',
                    'candle_index': candle_index,
                    'price': candle['close'],
                    'strength': self._calculate_pattern_strength(candle, 'hammer')
                })
            
            # Shooting star pattern
            if PriceActionPatterns.is_shooting_star(candle):
                patterns.append({
                    'type': 'shooting_star',
                    'direction': 'bearish',
                    'candle_index': candle_index,
                    'price': candle['close'],
                    'strength': self._calculate_pattern_strength(candle, 'shooting_star')
                })
            
            # Doji pattern
            if PriceActionPatterns.is_doji(candle):
                patterns.append({
                    'type': 'doji',
                    'direction': 'neutral',
                    'candle_index': candle_index,
                    'price': candle['close'],
                    'strength': self._calculate_pattern_strength(candle, 'doji')
                })
            
            # Pin bar pattern
            is_pin, pin_direction = PriceActionPatterns.is_pin_bar(candle, df['close'].std())
            if is_pin:
                patterns.append({
                    'type': 'pin_bar',
                    'direction': pin_direction.replace('_pin', ''),
                    'candle_index': candle_index,
                    'price': candle['close'],
                    'strength': self._calculate_pattern_strength(candle, 'pin_bar')
                })
        
        # Check multi-candle patterns
        if len(recent_candles) >= 2:
            # Engulfing patterns
            for i in range(1, len(recent_candles)):
                is_engulfing, engulfing_type = PriceActionPatterns.is_engulfing(
                    recent_candles.iloc[:i+1], i
                )
                if is_engulfing:
                    candle_index = len(df) - len(recent_candles) + i
                    patterns.append({
                        'type': 'engulfing',
                        'direction': engulfing_type.replace('_engulfing', ''),
                        'candle_index': candle_index,
                        'price': recent_candles.iloc[i]['close'],
                        'strength': self._calculate_engulfing_strength(recent_candles.iloc[i-1:i+1])
                    })
        
        return patterns
    
    def _calculate_pattern_strength(self, candle, pattern_type: str) -> float:
        """
        Calculate pattern strength score
        
        Args:
            candle: Candle data
            pattern_type: Type of pattern
            
        Returns:
            float: Pattern strength (0-100)
        """
        body_size = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        
        strength = 0
        
        if pattern_type == 'hammer':
            # Hammer strength based on shadow length and body size
            shadow_ratio = lower_shadow / body_size if body_size > 0 else 0
            body_ratio = body_size / total_range
            
            if shadow_ratio > 2:
                strength += 40
            if body_ratio < 0.3:
                strength += 30
            if upper_shadow < body_size * 0.5:
                strength += 30
        
        elif pattern_type == 'shooting_star':
            # Shooting star strength based on shadow length and body size
            shadow_ratio = upper_shadow / body_size if body_size > 0 else 0
            body_ratio = body_size / total_range
            
            if shadow_ratio > 2:
                strength += 40
            if body_ratio < 0.3:
                strength += 30
            if lower_shadow < body_size * 0.5:
                strength += 30
        
        elif pattern_type == 'doji':
            # Doji strength based on body size
            body_ratio = body_size / total_range
            if body_ratio < 0.1:
                strength += 50
            elif body_ratio < 0.2:
                strength += 30
        
        elif pattern_type == 'pin_bar':
            # Pin bar strength based on shadow length
            if lower_shadow > total_range * 0.6:
                strength += 50
            elif upper_shadow > total_range * 0.6:
                strength += 50
        
        return min(strength, 100)
    
    def _calculate_engulfing_strength(self, candles) -> float:
        """
        Calculate engulfing pattern strength
        
        Args:
            candles: Two candles for engulfing pattern
            
        Returns:
            float: Pattern strength (0-100)
        """
        if len(candles) < 2:
            return 0
        
        prev_candle = candles.iloc[0]
        curr_candle = candles.iloc[1]
        
        prev_body = abs(prev_candle['close'] - prev_candle['open'])
        curr_body = abs(curr_candle['close'] - curr_candle['open'])
        
        strength = 0
        
        # Body size comparison
        if curr_body > prev_body * 1.2:
            strength += 40
        
        # Complete engulfment
        if (curr_candle['open'] < prev_candle['close'] and 
            curr_candle['close'] > prev_candle['open']):
            strength += 30
        
        # Volume confirmation (if available)
        if 'tick_volume' in candles.columns:
            if curr_candle['tick_volume'] > prev_candle['tick_volume']:
                strength += 30
        
        return min(strength, 100)
    
    def _check_confluence(self, patterns: List[Dict], rsi: float, df) -> Dict:
        """
        Enhanced confluence analysis with multiple confirmation criteria
        
        Args:
            patterns: List of detected patterns
            rsi: Current RSI value
            df: OHLC DataFrame
            
        Returns:
            Dict: Confluence analysis
        """
        confluence = {
            'rsi_alignment': False,
            'trend_context': 'neutral',
            'volume_confirmation': False,
            'multiple_patterns': len(patterns) > 1,
            'support_resistance': False,
            'momentum_confirmation': False,
            'session_context': False,
            'confluence_score': 0
        }
        
        if not patterns:
            return confluence
        
        # Check RSI alignment with patterns
        latest_pattern = patterns[-1]  # Most recent pattern
        confluence_score = 0
        
        if latest_pattern['direction'] == 'bullish':
            # RSI should be oversold or recovering from oversold
            if rsi < 35 and rsi > 25:  # Oversold but not extreme
                confluence['rsi_alignment'] = True
                confluence_score += 25
            elif rsi < 45 and rsi > 30:  # Recovering from oversold
                confluence['rsi_alignment'] = True
                confluence_score += 15
            confluence['trend_context'] = 'bullish'
        elif latest_pattern['direction'] == 'bearish':
            # RSI should be overbought or declining from overbought
            if rsi > 65 and rsi < 75:  # Overbought but not extreme
                confluence['rsi_alignment'] = True
                confluence_score += 25
            elif rsi > 55 and rsi < 70:  # Declining from overbought
                confluence['rsi_alignment'] = True
                confluence_score += 15
            confluence['trend_context'] = 'bearish'
        
        # Check volume confirmation (enhanced)
        if len(df) >= 5 and 'tick_volume' in df.columns:
            current_volume = df['tick_volume'].iloc[-1]
            avg_volume = df['tick_volume'].tail(20).mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > 1.5:  # Strong volume
                confluence['volume_confirmation'] = True
                confluence_score += 20
            elif volume_ratio > 1.2:  # Moderate volume
                confluence_score += 10
        
        # Check support/resistance context
        if len(df) >= 20:
            current_price = df['close'].iloc[-1]
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            
            if latest_pattern['direction'] == 'bullish':
                # Check if near support level
                if current_price <= recent_low * 1.002:  # Within 0.2% of recent low
                    confluence['support_resistance'] = True
                    confluence_score += 20
            elif latest_pattern['direction'] == 'bearish':
                # Check if near resistance level
                if current_price >= recent_high * 0.998:  # Within 0.2% of recent high
                    confluence['support_resistance'] = True
                    confluence_score += 20
        
        # Check momentum confirmation
        if len(df) >= 3:
            if latest_pattern['direction'] == 'bullish':
                # Check for bullish momentum divergence
                price_change = df['close'].iloc[-1] - df['close'].iloc[-3]
                if price_change > 0 and rsi > df['close'].rolling(14).apply(lambda x: TechnicalIndicators.rsi(x, 14).iloc[-1] if len(x) >= 14 else 50).iloc[-3]:
                    confluence['momentum_confirmation'] = True
                    confluence_score += 15
            elif latest_pattern['direction'] == 'bearish':
                # Check for bearish momentum divergence
                price_change = df['close'].iloc[-1] - df['close'].iloc[-3]
                if price_change < 0 and rsi < df['close'].rolling(14).apply(lambda x: TechnicalIndicators.rsi(x, 14).iloc[-1] if len(x) >= 14 else 50).iloc[-3]:
                    confluence['momentum_confirmation'] = True
                    confluence_score += 15
        
        # Check session context
        current_hour = datetime.now().hour
        if 8 <= current_hour <= 16 or 20 <= current_hour <= 22:  # Active trading hours
            confluence['session_context'] = True
            confluence_score += 10
        
        # Multiple patterns bonus
        if confluence['multiple_patterns']:
            confluence_score += 10
        
        confluence['confluence_score'] = confluence_score
        return confluence
    
    def _generate_signal(self, patterns: List[Dict], confluence: Dict, 
                        current_price: float, atr: float, rsi: float) -> Dict:
        """
        Generate trading signal based on enhanced pattern analysis
        
        Args:
            patterns: List of detected patterns
            confluence: Confluence analysis
            current_price: Current price
            atr: Current ATR value
            rsi: Current RSI value
            
        Returns:
            Dict: Trading signal
        """
        if not patterns:
            return {'type': 'no_signal'}
        
        # Get the strongest recent pattern
        latest_pattern = max(patterns, key=lambda p: p['strength'])
        
        # Enhanced signal filtering
        # 1. Pattern strength threshold (increased)
        if latest_pattern['strength'] < 60:
            return {'type': 'no_signal', 'reason': 'Pattern strength too low'}
        
        # 2. Confluence score threshold (new)
        if confluence['confluence_score'] < 50:
            return {'type': 'no_signal', 'reason': 'Insufficient confluence'}
        
        # 3. RSI alignment (required)
        if not confluence['rsi_alignment']:
            return {'type': 'no_signal', 'reason': 'RSI not aligned'}
        
        # 4. Support/Resistance context (preferred)
        if not confluence['support_resistance']:
            return {'type': 'no_signal', 'reason': 'Not at key level'}
        
        # 5. Session context (preferred)
        if not confluence['session_context']:
            return {'type': 'no_signal', 'reason': 'Outside active hours'}
        
        # Generate signal based on pattern direction
        if latest_pattern['direction'] == 'bullish':
            # Calculate stop loss and take profit with better risk management
            stop_loss = RiskManagement.calculate_stop_loss(current_price, 'buy', atr, self.atr_multiplier)
            take_profit = RiskManagement.calculate_take_profit(current_price, stop_loss, 'buy', 2.5)  # Better R:R
            
            return {
                'type': 'buy',
                'direction': 'long',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': self._calculate_confidence(latest_pattern, confluence, rsi),
                'pattern_type': latest_pattern['type'],
                'confluence_score': confluence['confluence_score']
            }
        
        elif latest_pattern['direction'] == 'bearish':
            # Calculate stop loss and take profit with better risk management
            stop_loss = RiskManagement.calculate_stop_loss(current_price, 'sell', atr, self.atr_multiplier)
            take_profit = RiskManagement.calculate_take_profit(current_price, stop_loss, 'sell', 2.5)  # Better R:R
            
            return {
                'type': 'sell',
                'direction': 'short',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': self._calculate_confidence(latest_pattern, confluence, rsi),
                'pattern_type': latest_pattern['type'],
                'confluence_score': confluence['confluence_score']
            }
        
        return {'type': 'no_signal'}
    
    def _calculate_confidence(self, pattern: Dict, confluence: Dict, rsi: float) -> float:
        """
        Calculate signal confidence score
        
        Args:
            pattern: Pattern information
            confluence: Confluence analysis
            rsi: Current RSI value
            
        Returns:
            float: Confidence score (0-100)
        """
        confidence = 0
        
        # Pattern strength (40 points)
        confidence += pattern['strength'] * 0.4
        
        # Pattern type weight (20 points)
        pattern_weight = self.pattern_weights.get(pattern['type'], 0.5)
        confidence += pattern_weight * 20
        
        # RSI alignment (20 points)
        if confluence['rsi_alignment']:
            confidence += 20
        
        # Volume confirmation (10 points)
        if confluence['volume_confirmation']:
            confidence += 10
        
        # Multiple patterns (10 points)
        if confluence['multiple_patterns']:
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
        if len(df) < 5:
            return {'exit': False}
        
        try:
            current_price = df['close'].iloc[-1]
            rsi = TechnicalIndicators.rsi(df['close'], self.rsi_period).iloc[-1]
            
            # Check for opposite reversal pattern
            patterns = self._detect_patterns(df)
            if patterns:
                latest_pattern = patterns[-1]
                
                # Exit if opposite pattern appears
                if position['type'] == 'buy' and latest_pattern['direction'] == 'bearish':
                    return {
                        'exit': True,
                        'reason': f'Opposite reversal pattern detected: {latest_pattern["type"]}',
                        'exit_price': current_price
                    }
                elif position['type'] == 'sell' and latest_pattern['direction'] == 'bullish':
                    return {
                        'exit': True,
                        'reason': f'Opposite reversal pattern detected: {latest_pattern["type"]}',
                        'exit_price': current_price
                    }
            
            # Check RSI extremes
            if position['type'] == 'buy' and rsi > self.rsi_overbought:
                return {
                    'exit': True,
                    'reason': 'RSI overbought',
                    'exit_price': current_price
                }
            elif position['type'] == 'sell' and rsi < self.rsi_oversold:
                return {
                    'exit': True,
                    'reason': 'RSI oversold',
                    'exit_price': current_price
                }
            
            # Check for target reached
            entry_price = position['price_open']
            if position['type'] == 'buy' and current_price > entry_price * 1.015:  # 1.5% profit
                return {
                    'exit': True,
                    'reason': 'Target reached',
                    'exit_price': current_price
                }
            elif position['type'] == 'sell' and current_price < entry_price * 0.985:  # 1.5% profit
                return {
                    'exit': True,
                    'reason': 'Target reached',
                    'exit_price': current_price
                }
            
            return {'exit': False}
            
        except Exception as e:
            logger.error(f"Error in reversal patterns exit analysis: {e}")
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
                'rsi_period': self.rsi_period,
                'rsi_oversold': self.rsi_oversold,
                'rsi_overbought': self.rsi_overbought,
                'atr_period': self.atr_period,
                'atr_multiplier': self.atr_multiplier,
                'min_body_ratio': self.min_body_ratio,
                'min_shadow_ratio': self.min_shadow_ratio
            },
            'pattern_weights': self.pattern_weights
        }
