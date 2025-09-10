"""
Data Processor for AI Trading Bot
Handles feature engineering, data preprocessing, and dataset creation for ML models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import os

from config import *
from utils import TechnicalIndicators, SupportResistance, TrendAnalysis

logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processor for AI trading models"""
    
    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        self.feature_selector = None
        self.feature_names = []
        self.is_fitted = False
        
        # Feature engineering parameters
        self.technical_indicators = {
            'sma_periods': [5, 10, 20, 50],
            'ema_periods': [12, 26],
            'rsi_periods': [14, 21],
            'macd_params': [(12, 26, 9)],
            'bb_periods': [20],
            'atr_periods': [14, 21],
            'stoch_params': [(14, 3)]
        }
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set from OHLC data
        
        Args:
            df: OHLC DataFrame
            
        Returns:
            DataFrame: Features DataFrame
        """
        if len(df) < self.lookback_periods:
            logger.warning(f"Insufficient data: {len(df)} < {self.lookback_periods}")
            return pd.DataFrame()
        
        try:
            # Ensure DataFrame has proper structure
            if not isinstance(df, pd.DataFrame):
                logger.error("Input is not a DataFrame")
                return pd.DataFrame()
            
            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return pd.DataFrame()
            
            # Ensure data types are numeric
            for col in required_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception as e:
                        logger.error(f"Error converting {col} to numeric: {e}")
                        return pd.DataFrame()
            
            # Remove any rows with NaN values in required columns
            df = df.dropna(subset=required_columns)
            if len(df) == 0:
                logger.error("No valid data after cleaning")
                return pd.DataFrame()
            
            # Create features DataFrame with original index
            features = pd.DataFrame(index=df.index)
            
            # Price-based features
            features = self._add_price_features(features, df)
            
            # Technical indicators
            features = self._add_technical_indicators(features, df)
            
            # Price action patterns
            features = self._add_price_action_features(features, df)
            
            # Support/Resistance features
            features = self._add_sr_features(features, df)
            
            # Trend features
            features = self._add_trend_features(features, df)
            
            # Volatility features
            features = self._add_volatility_features(features, df)
            
            # Time-based features
            features = self._add_time_features(features, df)
            
            # Market microstructure features
            features = self._add_microstructure_features(features, df)
            
            # Remove rows with NaN values
            features = features.dropna()
            
            logger.info(f"Created {len(features.columns)} features for {len(features)} samples")
            return features
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def _add_price_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        try:
            # Price ratios
            features['price_change'] = df['close'].pct_change()
            
            # Avoid division by zero
            low_safe = df['low'].replace(0, np.nan)
            open_safe = df['open'].replace(0, np.nan)
            high_low_range = df['high'] - df['low']
            high_low_range_safe = high_low_range.replace(0, np.nan)
            
            features['high_low_ratio'] = df['high'] / low_safe
            features['close_open_ratio'] = df['close'] / open_safe
            features['body_size'] = abs(df['close'] - df['open']) / high_low_range_safe
            features['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / high_low_range_safe
            features['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / high_low_range_safe
            
            # Price momentum
            for period in [1, 2, 3, 5, 10]:
                if len(df) > period:
                    features[f'price_momentum_{period}'] = df['close'].pct_change(period)
                    features[f'high_momentum_{period}'] = df['high'].pct_change(period)
                    features[f'low_momentum_{period}'] = df['low'].pct_change(period)
            
            # Price position within range
            features['price_position'] = (df['close'] - df['low']) / high_low_range_safe
            
            return features
            
        except Exception as e:
            logger.error(f"Error adding price features: {e}")
            return features
    
    def _add_technical_indicators(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features"""
        try:
            # Moving averages
            for period in self.technical_indicators['sma_periods']:
                if len(df) >= period:
                    sma = TechnicalIndicators.sma(df['close'], period)
                    features[f'sma_{period}'] = sma
                    # Avoid division by zero
                    sma_safe = sma.replace(0, np.nan)
                    features[f'price_sma_ratio_{period}'] = df['close'] / sma_safe
                    features[f'sma_slope_{period}'] = sma.diff()
            
            for period in self.technical_indicators['ema_periods']:
                if len(df) >= period:
                    ema = TechnicalIndicators.ema(df['close'], period)
                    features[f'ema_{period}'] = ema
                    # Avoid division by zero
                    ema_safe = ema.replace(0, np.nan)
                    features[f'price_ema_ratio_{period}'] = df['close'] / ema_safe
            
            # RSI
            for period in self.technical_indicators['rsi_periods']:
                if len(df) >= period:
                    rsi = TechnicalIndicators.rsi(df['close'], period)
                    features[f'rsi_{period}'] = rsi
                    features[f'rsi_oversold_{period}'] = (rsi < 30).astype(int)
                    features[f'rsi_overbought_{period}'] = (rsi > 70).astype(int)
            
            # MACD
            for fast, slow, signal in self.technical_indicators['macd_params']:
                if len(df) >= slow:
                    macd, macd_signal, macd_hist = TechnicalIndicators.macd(df['close'], fast, slow, signal)
                    features[f'macd_{fast}_{slow}'] = macd
                    features[f'macd_signal_{fast}_{slow}'] = macd_signal
                    features[f'macd_hist_{fast}_{slow}'] = macd_hist
                    features[f'macd_bullish_{fast}_{slow}'] = (macd > macd_signal).astype(int)
            
            # Bollinger Bands
            for period in self.technical_indicators['bb_periods']:
                if len(df) >= period:
                    bb_upper, bb_lower, bb_middle = TechnicalIndicators.bollinger_bands(df['close'], period)
                    features[f'bb_upper_{period}'] = bb_upper
                    features[f'bb_lower_{period}'] = bb_lower
                    features[f'bb_middle_{period}'] = bb_middle
                    
                    # Avoid division by zero
                    bb_middle_safe = bb_middle.replace(0, np.nan)
                    bb_width = (bb_upper - bb_lower) / bb_middle_safe
                    features[f'bb_width_{period}'] = bb_width
                    
                    # Avoid division by zero
                    bb_range = bb_upper - bb_lower
                    bb_range_safe = bb_range.replace(0, np.nan)
                    features[f'bb_position_{period}'] = (df['close'] - bb_lower) / bb_range_safe
                    
                    # BB squeeze
                    if len(features) >= 20:
                        bb_width_mean = bb_width.rolling(20).mean()
                        features[f'bb_squeeze_{period}'] = (bb_width < bb_width_mean * 0.8).astype(int)
            
            # ATR
            for period in self.technical_indicators['atr_periods']:
                if len(df) >= period:
                    atr = TechnicalIndicators.atr(df['high'], df['low'], df['close'], period)
                    features[f'atr_{period}'] = atr
                    # Avoid division by zero
                    close_safe = df['close'].replace(0, np.nan)
                    features[f'atr_ratio_{period}'] = atr / close_safe
                    if len(atr) >= 20:
                        features[f'volatility_{period}'] = atr.rolling(20).mean()
            
            # Stochastic
            for k_period, d_period in self.technical_indicators['stoch_params']:
                if len(df) >= k_period:
                    stoch_k, stoch_d = TechnicalIndicators.stochastic(df['high'], df['low'], df['close'], k_period, d_period)
                    features[f'stoch_k_{k_period}'] = stoch_k
                    features[f'stoch_d_{k_period}'] = stoch_d
                    features[f'stoch_oversold_{k_period}'] = (stoch_k < 20).astype(int)
                    features[f'stoch_overbought_{k_period}'] = (stoch_k > 80).astype(int)
            
            return features
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return features
    
    def _add_price_action_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add price action pattern features"""
        # Candlestick patterns
        features['is_hammer'] = 0
        features['is_shooting_star'] = 0
        features['is_doji'] = 0
        features['is_engulfing'] = 0
        
        for i in range(len(df)):
            if i == 0:
                continue
                
            candle = df.iloc[i]
            
            # Hammer pattern
            body_size = abs(candle['close'] - candle['open'])
            lower_shadow = min(candle['open'], candle['close']) - candle['low']
            upper_shadow = candle['high'] - max(candle['open'], candle['close'])
            total_range = candle['high'] - candle['low']
            
            if (body_size < total_range * 0.3 and 
                lower_shadow > body_size * 2 and 
                upper_shadow < body_size * 0.5):
                features.iloc[i, features.columns.get_loc('is_hammer')] = 1
            
            # Shooting star pattern
            if (body_size < total_range * 0.3 and 
                upper_shadow > body_size * 2 and 
                lower_shadow < body_size * 0.5):
                features.iloc[i, features.columns.get_loc('is_shooting_star')] = 1
            
            # Doji pattern
            if body_size < total_range * 0.1:
                features.iloc[i, features.columns.get_loc('is_doji')] = 1
            
            # Engulfing pattern (simplified)
            if i > 0:
                prev_candle = df.iloc[i-1]
                if (prev_candle['close'] < prev_candle['open'] and  # Previous bearish
                    candle['close'] > candle['open'] and    # Current bullish
                    candle['open'] < prev_candle['close'] and   # Current opens below previous close
                    candle['close'] > prev_candle['open']):     # Current closes above previous open
                    features.iloc[i, features.columns.get_loc('is_engulfing')] = 1
        
        return features
    
    def _add_sr_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add support/resistance features"""
        # Find support and resistance levels
        support_levels, resistance_levels = SupportResistance.find_levels(df, 50, 2)
        
        # Distance to nearest support/resistance
        features['dist_to_support'] = 0.0
        features['dist_to_resistance'] = 0.0
        features['near_support'] = 0
        features['near_resistance'] = 0
        
        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            
            # Distance to support
            if support_levels:
                min_dist_support = min(abs(current_price - level) for level in support_levels)
                features.iloc[i, features.columns.get_loc('dist_to_support')] = min_dist_support
                if min_dist_support < current_price * 0.001:  # Within 0.1%
                    features.iloc[i, features.columns.get_loc('near_support')] = 1
            
            # Distance to resistance
            if resistance_levels:
                min_dist_resistance = min(abs(current_price - level) for level in resistance_levels)
                features.iloc[i, features.columns.get_loc('dist_to_resistance')] = min_dist_resistance
                if min_dist_resistance < current_price * 0.001:  # Within 0.1%
                    features.iloc[i, features.columns.get_loc('near_resistance')] = 1
        
        return features
    
    def _add_trend_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend analysis features"""
        # Trend identification
        trend = []
        for i in range(len(df)):
            if i < 50:  # Need enough data for trend analysis
                trend.append('sideways')
            else:
                subset = df.iloc[i-49:i+1]
                current_trend = TrendAnalysis.identify_trend(subset)
                trend.append(current_trend)
        
        # Convert trend to numeric values
        trend_numeric = []
        for t in trend:
            if t == 'uptrend':
                trend_numeric.append(1)
            elif t == 'downtrend':
                trend_numeric.append(-1)
            else:  # sideways
                trend_numeric.append(0)
        
        features['trend'] = pd.Series(trend_numeric, index=df.index)
        features['is_uptrend'] = (features['trend'] == 1).astype(int)
        features['is_downtrend'] = (features['trend'] == -1).astype(int)
        features['is_sideways'] = (features['trend'] == 0).astype(int)
        
        # Higher highs and lower lows
        features['higher_highs'] = 0
        features['lower_lows'] = 0
        
        for i in range(20, len(df)):
            subset = df.iloc[i-19:i+1]
            hh, ll = TrendAnalysis.find_higher_highs_lower_lows(subset)
            features.iloc[i, features.columns.get_loc('higher_highs')] = int(hh)
            features.iloc[i, features.columns.get_loc('lower_lows')] = int(ll)
        
        return features
    
    def _add_volatility_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        try:
            # Rolling volatility
            for period in [5, 10, 20]:
                if len(df) >= period:
                    features[f'volatility_{period}'] = df['close'].rolling(period).std()
                    # Avoid division by zero
                    close_safe = df['close'].replace(0, np.nan)
                    features[f'volatility_ratio_{period}'] = features[f'volatility_{period}'] / close_safe
            
            # Volatility regime
            features['volatility_regime'] = 0  # 0: low, 1: medium, 2: high
            if 'volatility_20' in features.columns and len(features) >= 100:
                vol_20 = features['volatility_20']
                
                # Calculate percentiles separately to avoid list issue
                p33_series = vol_20.rolling(100).quantile(0.33)
                p67_series = vol_20.rolling(100).quantile(0.67)
                
                for i in range(len(features)):
                    if i < 100:
                        continue
                    current_vol = vol_20.iloc[i]
                    p33 = p33_series.iloc[i]
                    p67 = p67_series.iloc[i]
                    
                    if pd.notna(current_vol) and pd.notna(p33) and pd.notna(p67):
                        if current_vol <= p33:
                            features.iloc[i, features.columns.get_loc('volatility_regime')] = 0
                        elif current_vol <= p67:
                            features.iloc[i, features.columns.get_loc('volatility_regime')] = 1
                        else:
                            features.iloc[i, features.columns.get_loc('volatility_regime')] = 2
            
            return features
            
        except Exception as e:
            logger.error(f"Error adding volatility features: {e}")
            return features
    
    def _add_time_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if hasattr(df.index, 'hour'):
            features['hour'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
            
            # Market session features
            features['is_london_session'] = ((features['hour'] >= 8) & (features['hour'] < 16)).astype(int)
            features['is_ny_session'] = ((features['hour'] >= 13) & (features['hour'] < 21)).astype(int)
            features['is_asian_session'] = ((features['hour'] >= 0) & (features['hour'] < 8)).astype(int)
        
        return features
    
    def _add_microstructure_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        # Tick volume features (if available)
        if 'tick_volume' in df.columns:
            features['volume'] = df['tick_volume']
            features['volume_ma_5'] = df['tick_volume'].rolling(5).mean()
            features['volume_ma_20'] = df['tick_volume'].rolling(20).mean()
            features['volume_ratio_5'] = df['tick_volume'] / features['volume_ma_5']
            features['volume_ratio_20'] = df['tick_volume'] / features['volume_ma_20']
            features['high_volume'] = (features['volume_ratio_20'] > 1.5).astype(int)
        
        # Price gaps
        features['gap_up'] = (df['open'] > df['close'].shift(1)).astype(int)
        features['gap_down'] = (df['open'] < df['close'].shift(1)).astype(int)
        features['gap_size'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        return features
    
    def create_targets(self, df: pd.DataFrame, prediction_horizon: int = 5) -> pd.Series:
        """
        Create target variables for supervised learning
        
        Args:
            df: OHLC DataFrame
            prediction_horizon: Number of periods ahead to predict
            
        Returns:
            Series: Target values
        """
        # Future price change
        future_price = df['close'].shift(-prediction_horizon)
        price_change = (future_price - df['close']) / df['close']
        
        # Classification targets
        targets = pd.cut(price_change, 
                        bins=[-np.inf, -0.001, 0.001, np.inf], 
                        labels=[0, 1, 2])  # 0: sell, 1: hold, 2: buy
        
        # Convert to float and handle NaN values
        targets = targets.astype(float)
        
        # The last prediction_horizon rows will be NaN (no future data)
        # This is expected and will be filtered out in prepare_training_data
        
        return targets
    
    def prepare_training_data(self, features_df: pd.DataFrame, original_df: pd.DataFrame = None, prediction_horizon: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare complete training dataset
        
        Args:
            features_df: Features DataFrame (already created)
            original_df: Original OHLC DataFrame (for creating targets)
            prediction_horizon: Prediction horizon
            
        Returns:
            Tuple: (features, targets)
        """
        if features_df.empty:
            return pd.DataFrame(), pd.Series()
        
        # If original_df is provided, use it for targets, otherwise assume features_df has OHLC data
        target_df = original_df if original_df is not None else features_df
        
        # Create targets
        targets = self.create_targets(target_df, prediction_horizon)
        
        # Align features and targets
        common_index = features_df.index.intersection(targets.index)
        features = features_df.loc[common_index]
        targets = targets.loc[common_index]
        
        # Remove rows with NaN targets (last prediction_horizon rows)
        valid_mask = ~targets.isna()
        if valid_mask.any():
            features = features.loc[valid_mask]
            targets = targets.loc[valid_mask]
        else:
            # No valid targets, return empty data
            features = pd.DataFrame()
            targets = pd.Series(dtype=float)
        
        logger.info(f"Prepared training data: {len(features)} samples, {len(features.columns)} features")
        return features, targets
    
    def fit_scalers(self, features: pd.DataFrame):
        """Fit feature scalers"""
        try:
            # Ensure scaler objects are valid estimators (guard against corrupted/legacy loads)
            from sklearn.preprocessing import StandardScaler as _StdScaler, MinMaxScaler as _MinMaxScaler
            if not hasattr(self.scaler, 'fit'):
                self.scaler = _StdScaler()
            if not hasattr(self.feature_scaler, 'fit'):
                self.feature_scaler = _MinMaxScaler()

            # Fit standard scaler for numerical features
            numerical_features = features.select_dtypes(include=[np.number])
            self.scaler.fit(numerical_features)
            
            # Fit feature selector
            self.feature_selector = SelectKBest(score_func=f_regression, k=min(50, len(numerical_features.columns)))
            
            self.feature_names = list(numerical_features.columns)
            self.is_fitted = True
            
            logger.info(f"Fitted scalers for {len(self.feature_names)} features")
            
        except Exception as e:
            logger.error(f"Error fitting scalers: {e}")
    
    def transform_features(self, features: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scalers"""
        if not self.is_fitted:
            logger.warning("Scalers not fitted. Call fit_scalers first.")
            return np.array([])
        
        try:
            # Select numerical features
            numerical_features = features.select_dtypes(include=[np.number])
            
            # Scale features
            scaled_features = self.scaler.transform(numerical_features)
            
            return scaled_features
            
        except Exception as e:
            logger.error(f"Error transforming features: {e}")
            return np.array([])
    
    def save_processor(self, filepath: str):
        """Save processor state"""
        try:
            processor_data = {
                'scaler': self.scaler,
                'feature_scaler': self.feature_scaler,
                'feature_selector': self.feature_selector,
                'feature_names': self.feature_names,
                'is_fitted': self.is_fitted,
                'lookback_periods': self.lookback_periods,
                'technical_indicators': self.technical_indicators
            }
            
            joblib.dump(processor_data, filepath)
            logger.info(f"Saved processor to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving processor: {e}")
    
    def load_processor(self, filepath: str):
        """Load processor state"""
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Processor file not found: {filepath}")
                return False
            
            processor_data = joblib.load(filepath)
            # Support legacy formats and validate contents
            if isinstance(processor_data, dict):
                self.scaler = processor_data.get('scaler', self.scaler)
                self.feature_scaler = processor_data.get('feature_scaler', self.feature_scaler)
                self.feature_selector = processor_data.get('feature_selector', None)
                self.feature_names = processor_data.get('feature_names', [])
                self.is_fitted = processor_data.get('is_fitted', False)
                self.lookback_periods = processor_data.get('lookback_periods', self.lookback_periods)
                self.technical_indicators = processor_data.get('technical_indicators', self.technical_indicators)
            else:
                # Unknown object format; keep defaults
                logger.warning("Unexpected processor format; using defaults where needed")
            
            # Validate estimator objects; fall back if necessary
            from sklearn.preprocessing import StandardScaler as _StdScaler, MinMaxScaler as _MinMaxScaler
            if not hasattr(self.scaler, 'fit'):
                self.scaler = _StdScaler()
                self.is_fitted = False
            if not hasattr(self.feature_scaler, 'fit'):
                self.feature_scaler = _MinMaxScaler()
                self.is_fitted = False
            
            logger.info(f"Loaded processor from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading processor: {e}")
            return False
