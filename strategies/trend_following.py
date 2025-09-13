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
from ta.trend import MACD, EMAIndicator
from ta.momentum import StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
from ta.trend import EMAIndicator
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
            macd_line = macd_signal = macd_hist = None
            stoch_k = stoch_d = None
            bb_high = bb_low = None
            ema9 = ema20 = ema50 = ema200 = None
            vwap = None
            rsi_fast = None
            obv = None
            try:
                if INDICATORS_ENABLE_MACD and PANDAS_AVAILABLE:
                    macd = MACD(close=df['close'], window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
                    macd_line = macd.macd()
                    macd_signal = macd.macd_signal()
                    macd_hist = macd.macd_diff()
                if INDICATORS_ENABLE_STOCH and PANDAS_AVAILABLE:
                    stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=STOCH_K, smooth_window=STOCH_SMOOTH)
                    stoch_k = stoch.stoch()
                    stoch_d = stoch.stoch_signal() if hasattr(stoch, 'stoch_signal') else stoch_k.rolling(STOCH_D).mean()
                if INDICATORS_ENABLE_BB and PANDAS_AVAILABLE:
                    bb = BollingerBands(close=df['close'], window=BB_PERIOD, window_dev=BB_STD)
                    bb_high = bb.bollinger_hband()
                    bb_low = bb.bollinger_lband()
                # EMA 9/20/50/200
                if PANDAS_AVAILABLE:
                    ema9 = EMAIndicator(close=df['close'], window=EMA_FAST_SHORT).ema_indicator()
                    ema20 = EMAIndicator(close=df['close'], window=EMA_SLOW_SHORT).ema_indicator()
                    ema50 = EMAIndicator(close=df['close'], window=EMA_MEDIUM).ema_indicator()
                    ema200 = EMAIndicator(close=df['close'], window=EMA_LONG).ema_indicator()
                # VWAP (approx using cumulative typical price * volume / cumulative volume if volume available)
                if VWAP_ENABLED and PANDAS_AVAILABLE and 'tick_volume' in df.columns:
                    tp = (df['high'] + df['low'] + df['close']) / 3.0
                    cum_v = df['tick_volume'].cumsum().replace(0, np.nan)
                    vwap = (tp * df['tick_volume']).cumsum() / cum_v
                # Fast RSI
                if PANDAS_AVAILABLE:
                    rsi_fast = TechnicalIndicators.rsi(df['close'], RSI_FAST_PERIOD)
                # OBV
                if OBV_ENABLED and PANDAS_AVAILABLE and 'tick_volume' in df.columns:
                    obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['tick_volume']).on_balance_volume()
            except Exception:
                pass
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
            
            # Additional indicator gates
            indicator_ok = True
            try:
                if INDICATORS_ENABLE_MACD and macd_line is not None and macd_signal is not None:
                    # For longs prefer MACD above signal; for shorts below
                    pass  # decision made later per direction
                if INDICATORS_ENABLE_STOCH and stoch_k is not None:
                    # Avoid long entries if stoch overbought; avoid shorts if oversold
                    pass
                if INDICATORS_ENABLE_BB and bb_high is not None and bb_low is not None:
                    # Optional: avoid entries when price outside bands (exhaustion)
                    pass
            except Exception:
                indicator_ok = True

            # Higher timeframe EMA confirmation (optional)
            if HTF_CONFIRM_ENABLED and PANDAS_AVAILABLE:
                try:
                    # Fetch HTF data via connector if available
                    htf_map = TIMEFRAME_MAPPING.get(HTF_TIMEFRAME, None)
                    if htf_map:
                        htf_df = self.mt5.get_rates(self.mt5.get_symbol(), HTF_TIMEFRAME, 400)
                        if htf_df is not None and len(htf_df) >= EMA_LONG + 5:
                            htf_ema50 = EMAIndicator(close=htf_df['close'], window=EMA_MEDIUM).ema_indicator()
                            htf_ema200 = EMAIndicator(close=htf_df['close'], window=EMA_LONG).ema_indicator()
                            self._htf_bull = htf_ema50.iloc[-1] > htf_ema200.iloc[-1]
                            self._htf_bear = htf_ema50.iloc[-1] < htf_ema200.iloc[-1]
                        else:
                            self._htf_bull = self._htf_bear = False
                    else:
                        self._htf_bull = self._htf_bear = False
                except Exception:
                    self._htf_bull = self._htf_bear = False

            # Generate signals
            signal = self._generate_signal(
                trend, pullback_signal, higher_highs, lower_lows,
                current_rsi, current_price, current_fast_ma, current_slow_ma,
                df,
                macd_line=macd_line, macd_signal=macd_signal, macd_hist=macd_hist,
                stoch_k=stoch_k, stoch_d=stoch_d,
                bb_high=bb_high, bb_low=bb_low,
                ema9=ema9, ema20=ema20, ema50=ema50, ema200=ema200,
                vwap=vwap, rsi_fast=rsi_fast, obv=obv
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
                        fast_ma: float, slow_ma: float, df,
                        macd_line=None, macd_signal=None, macd_hist=None,
                        stoch_k=None, stoch_d=None,
                        bb_high=None, bb_low=None,
                        ema9=None, ema20=None, ema50=None, ema200=None,
                        vwap=None, rsi_fast=None, obv=None) -> Dict:
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
            
            # HTF confirmation
            if HTF_CONFIRM_ENABLED:
                if not getattr(self, '_htf_bull', False):
                    return {'type': 'no_signal', 'reason': 'HTF not bullish'}

            # Indicator gates for long
            if INDICATORS_ENABLE_MACD and macd_line is not None and macd_signal is not None:
                try:
                    if pd.isna(macd_line.iloc[-1]) or pd.isna(macd_signal.iloc[-1]) or macd_line.iloc[-1] <= macd_signal.iloc[-1]:
                        return {'type': 'no_signal', 'reason': 'MACD not supportive'}
                except Exception:
                    pass
            # EMA alignment: price above EMA9>EMA20 and above EMA50 if available
            try:
                if ema9 is not None and ema20 is not None:
                    if ema9.iloc[-1] <= ema20.iloc[-1] or current_price <= ema9.iloc[-1]:
                        return {'type': 'no_signal', 'reason': 'EMA9/20 not aligned'}
                if ema50 is not None and current_price <= ema50.iloc[-1]:
                    return {'type': 'no_signal', 'reason': 'Below EMA50'}
                if ema200 is not None and ema50 is not None and ema50.iloc[-1] <= ema200.iloc[-1]:
                    # Higher timeframe filter: medium above long
                    return {'type': 'no_signal', 'reason': 'EMA50<EMA200'}
            except Exception:
                pass
            if INDICATORS_ENABLE_STOCH and stoch_k is not None:
                try:
                    if stoch_k.iloc[-1] >= STOCH_OVERBOUGHT:
                        return {'type': 'no_signal', 'reason': 'Stoch overbought'}
                except Exception:
                    pass
            if INDICATORS_ENABLE_BB and bb_high is not None and bb_low is not None:
                try:
                    if current_price > bb_high.iloc[-1]:
                        return {'type': 'no_signal', 'reason': 'Price above BB'}
                except Exception:
                    pass
            # VWAP support: prefer price above VWAP for longs
            try:
                if vwap is not None and current_price < vwap.iloc[-1]:
                    return {'type': 'no_signal', 'reason': 'Below VWAP'}
            except Exception:
                pass
            # Fast RSI confirmation (avoid buying when already hot)
            try:
                if rsi_fast is not None and not pd.isna(rsi_fast.iloc[-1]) and rsi_fast.iloc[-1] > 75:
                    return {'type': 'no_signal', 'reason': 'Fast RSI too high'}
            except Exception:
                pass

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
            
            if HTF_CONFIRM_ENABLED:
                if not getattr(self, '_htf_bear', False):
                    return {'type': 'no_signal', 'reason': 'HTF not bearish'}

            # Indicator gates for short
            if INDICATORS_ENABLE_MACD and macd_line is not None and macd_signal is not None:
                try:
                    if pd.isna(macd_line.iloc[-1]) or pd.isna(macd_signal.iloc[-1]) or macd_line.iloc[-1] >= macd_signal.iloc[-1]:
                        return {'type': 'no_signal', 'reason': 'MACD not supportive'}
                except Exception:
                    pass
            # EMA alignment: price below EMA9<EMA20 and below EMA50
            try:
                if ema9 is not None and ema20 is not None:
                    if ema9.iloc[-1] >= ema20.iloc[-1] or current_price >= ema9.iloc[-1]:
                        return {'type': 'no_signal', 'reason': 'EMA9/20 not aligned'}
                if ema50 is not None and current_price >= ema50.iloc[-1]:
                    return {'type': 'no_signal', 'reason': 'Above EMA50'}
                if ema200 is not None and ema50 is not None and ema50.iloc[-1] >= ema200.iloc[-1]:
                    return {'type': 'no_signal', 'reason': 'EMA50>EMA200'}
            except Exception:
                pass
            if INDICATORS_ENABLE_STOCH and stoch_k is not None:
                try:
                    if stoch_k.iloc[-1] <= STOCH_OVERSOLD:
                        return {'type': 'no_signal', 'reason': 'Stoch oversold'}
                except Exception:
                    pass
            if INDICATORS_ENABLE_BB and bb_high is not None and bb_low is not None:
                try:
                    if current_price < bb_low.iloc[-1]:
                        return {'type': 'no_signal', 'reason': 'Price below BB'}
                except Exception:
                    pass
            # VWAP: prefer price below VWAP for shorts
            try:
                if vwap is not None and current_price > vwap.iloc[-1]:
                    return {'type': 'no_signal', 'reason': 'Above VWAP'}
            except Exception:
                pass
            # Fast RSI confirmation (avoid selling when already washed out)
            try:
                if rsi_fast is not None and not pd.isna(rsi_fast.iloc[-1]) and rsi_fast.iloc[-1] < 25:
                    return {'type': 'no_signal', 'reason': 'Fast RSI too low'}
            except Exception:
                pass

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
