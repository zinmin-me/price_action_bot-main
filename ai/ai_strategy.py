"""
AI Strategy for Price Action Trading Bot
Advanced machine learning-based trading strategy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import joblib
import os

from config import *
from utils import TechnicalIndicators, RiskManagement
from .data_processor import DataProcessor
from .model_manager import ModelManager

logger = logging.getLogger(__name__)

class AIStrategy:
    """AI-powered trading strategy using machine learning"""
    
    def __init__(self, mt5_connector, model_dir: str = "ai/models"):
        self.mt5 = mt5_connector
        self.name = "AI Strategy"
        self.magic_number = 2001  # Unique magic number for AI strategy
        self.enabled = True
        
        # AI Components
        self.data_processor = DataProcessor()
        self.model_manager = ModelManager(model_dir)
        self.is_trained = False
        self.last_prediction = None
        self.prediction_confidence = 0.0
        
        # Strategy parameters
        self.prediction_horizon = 5  # Predict 5 periods ahead
        self.min_confidence_threshold = 0.6  # Minimum confidence for trading
        self.risk_reward_ratio = 2.0
        self.atr_multiplier = 1.5
        
        # Performance tracking
        self.prediction_history = []
        self.trade_history = []
        self.accuracy_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy': 0.0
        }
        
        # Load existing models if available
        self._load_models()
        
    def _load_models(self):
        """Load existing AI models"""
        try:
            # Load data processor
            processor_path = os.path.join(self.model_manager.model_dir, "data_processor.joblib")
            if os.path.exists(processor_path):
                self.data_processor.load_processor(processor_path)
                logger.info("Loaded existing data processor")
            
            # Load models
            loading_results = self.model_manager.load_models()
            if any(loading_results.values()):
                self.is_trained = True
                logger.info(f"Loaded {sum(loading_results.values())} AI models")
                try:
                    # Only create ensemble if it wasn't already loaded from disk
                    if self.model_manager.ensemble_model is None:
                        self.model_manager._create_ensemble_model()
                except Exception:
                    logger.warning("Could not create ensemble from loaded models; will fallback to best model at predict time")
            else:
                logger.info("No existing models found - will train new models")
                
        except Exception as e:
            logger.error(f"Error loading AI models: {e}")
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Analyze market using AI models
        
        Args:
            df: OHLC DataFrame
            
        Returns:
            Dict: Analysis results with AI predictions
        """
        if len(df) < self.data_processor.lookback_periods:
            return {'signal': 'no_signal', 'reason': 'Insufficient data for AI analysis'}
        
        try:
            # Create features
            features = self.data_processor.create_features(df)
            if features.empty:
                return {'signal': 'no_signal', 'reason': 'Failed to create features'}
            
            # Align feature columns to training schema and transform consistently
            try:
                expected_cols = self.data_processor.feature_names or list(features.select_dtypes(include=[np.number]).columns)
                # Reindex to expected columns (fill missing with 0), drop extras
                features_aligned = features.reindex(columns=expected_cols, fill_value=0)
                X_all = self.data_processor.transform_features(features_aligned)
                if X_all.size == 0:
                    # Fallback to raw values if scaler not fitted
                    latest_features = features_aligned.iloc[-1:].values
                else:
                    latest_features = X_all[-1:].reshape(1, -1)
            except Exception:
                latest_features = features.iloc[-1:].values
            
            if not self.is_trained:
                return {'signal': 'no_signal', 'reason': 'AI models not trained'}
            
            # Make prediction
            predictions, probabilities = self.model_manager.predict(latest_features, 'ensemble')
            
            if len(predictions) == 0:
                return {'signal': 'no_signal', 'reason': 'Failed to make prediction'}
            
            prediction = predictions[0]
            confidence = probabilities[0].max() if probabilities is not None else 0.0
            
            # Store prediction for tracking
            self.last_prediction = {
                'timestamp': datetime.now(),
                'prediction': prediction,
                'confidence': confidence,
                'features': latest_features[0]
            }
            
            self.prediction_history.append(self.last_prediction)
            self.prediction_confidence = confidence

            # Generate simple explainability for this prediction
            try:
                feature_names = self.data_processor.feature_names or [f"f{i}" for i in range(latest_features.shape[1])]
                self.last_prediction['explain'] = self.model_manager.explain(latest_features, feature_names)
            except Exception:
                self.last_prediction['explain'] = None
            
            # Update accuracy metrics
            self.accuracy_metrics['total_predictions'] += 1
            
            # Generate trading signal
            signal = self._generate_ai_signal(prediction, confidence, df)
            
            return {
                'signal': signal['type'],
                'direction': signal.get('direction'),
                'entry_price': signal.get('entry_price'),
                'stop_loss': signal.get('stop_loss'),
                'take_profit': signal.get('take_profit'),
                'confidence': confidence,
                'ai_prediction': prediction,
                'prediction_horizon': self.prediction_horizon,
                'model_accuracy': self.accuracy_metrics['accuracy'],
                'strategy_type': 'ai_ml'
            }
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return {'signal': 'error', 'reason': str(e)}
    
    def _generate_ai_signal(self, prediction: int, confidence: float, df: pd.DataFrame) -> Dict:
        """
        Generate trading signal from AI prediction
        
        Args:
            prediction: AI prediction (0: sell, 1: hold, 2: buy)
            confidence: Prediction confidence
            df: OHLC DataFrame
            
        Returns:
            Dict: Trading signal
        """
        current_price = df['close'].iloc[-1]
        
        # Check confidence threshold
        if confidence < self.min_confidence_threshold:
            return {'type': 'no_signal'}
        
        # Generate signal based on prediction
        if prediction == 2:  # Buy signal
            # Calculate stop loss and take profit
            atr = TechnicalIndicators.atr(df['high'], df['low'], df['close'], 14).iloc[-1]
            stop_loss = RiskManagement.calculate_stop_loss(current_price, 'buy', atr, self.atr_multiplier)
            take_profit = RiskManagement.calculate_take_profit(current_price, stop_loss, 'buy', self.risk_reward_ratio)
            
            return {
                'type': 'buy',
                'direction': 'long',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': confidence
            }
        
        elif prediction == 0:  # Sell signal
            # Calculate stop loss and take profit
            atr = TechnicalIndicators.atr(df['high'], df['low'], df['close'], 14).iloc[-1]
            stop_loss = RiskManagement.calculate_stop_loss(current_price, 'sell', atr, self.atr_multiplier)
            take_profit = RiskManagement.calculate_take_profit(current_price, stop_loss, 'sell', self.risk_reward_ratio)
            
            return {
                'type': 'sell',
                'direction': 'short',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': confidence
            }
        
        else:  # Hold signal
            return {'type': 'no_signal'}
    
    def train_models(self, df: pd.DataFrame) -> Dict:
        """
        Train AI models with historical data
        
        Args:
            df: Historical OHLC DataFrame
            
        Returns:
            Dict: Training results
        """
        try:
            logger.info("Starting AI model training...")
            
            # Validate input data
            if df is None or df.empty:
                return {'error': 'No data provided for training'}
            
            # Ensure DataFrame has proper index
            if isinstance(df.index, pd.RangeIndex):
                df = df.reset_index(drop=True)
            
            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return {'error': f'Missing required columns: {missing_columns}'}
            
            # Clean the data
            df = df.dropna(subset=required_columns)
            if len(df) < 100:
                return {'error': f'Insufficient data for training: {len(df)} rows (minimum 100 required)'}
            
            logger.info(f"Training with {len(df)} data points")
            
            # Create features first
            features = self.data_processor.create_features(df)
            
            if features.empty:
                return {'error': 'Failed to create features from data'}
            
            logger.info(f"Created {len(features)} feature samples with {len(features.columns)} features")
            
            # Prepare training data with features and original data
            features, targets = self.data_processor.prepare_training_data(features, df, self.prediction_horizon)
            
            if features.empty or len(targets) == 0:
                return {'error': 'Failed to prepare training data'}
            
            # Fit data processor
            self.data_processor.fit_scalers(features)
            
            # Transform features
            X = self.data_processor.transform_features(features)
            y = targets.values
            
            if X.size == 0:
                return {'error': 'Failed to transform features'}
            
            logger.info(f"Transformed features: {X.shape}")
            
            # Create and train models
            self.model_manager.create_models()
            training_results = self.model_manager.train_models(X, y)
            
            # Train ensemble
            self.model_manager.train_ensemble(X, y)
            
            # Mark as trained
            self.is_trained = True
            
            # Save models
            self.model_manager.save_models()
            self.data_processor.save_processor(
                os.path.join(self.model_manager.model_dir, "data_processor.joblib")
            )
            
            logger.info("AI model training completed successfully")
            
            return {
                'status': 'success',
                'training_results': training_results,
                'n_samples': len(X),
                'n_features': X.shape[1],
                'models_trained': len(training_results)
            }
            
        except Exception as e:
            logger.error(f"Error training AI models: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'error': str(e)}
    
    def retrain_models(self, df: pd.DataFrame) -> Dict:
        """
        Retrain models with updated data
        
        Args:
            df: Updated OHLC DataFrame
            
        Returns:
            Dict: Retraining results
        """
        logger.info("Retraining AI models with updated data...")
        return self.train_models(df)
    
    def update_prediction_accuracy(self, actual_outcome: str):
        """
        Update prediction accuracy based on actual market outcome
        
        Args:
            actual_outcome: 'buy', 'sell', or 'hold'
        """
        if not self.last_prediction:
            return
        
        try:
            # Map prediction to outcome
            prediction_map = {0: 'sell', 1: 'hold', 2: 'buy'}
            predicted_outcome = prediction_map.get(self.last_prediction['prediction'], 'hold')
            
            # Check if prediction was correct
            is_correct = (predicted_outcome == actual_outcome)
            
            if is_correct:
                self.accuracy_metrics['correct_predictions'] += 1
            
            # Update accuracy
            if self.accuracy_metrics['total_predictions'] > 0:
                self.accuracy_metrics['accuracy'] = (
                    self.accuracy_metrics['correct_predictions'] / 
                    self.accuracy_metrics['total_predictions']
                )
            
            logger.info(f"Prediction accuracy updated: {self.accuracy_metrics['accuracy']:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating prediction accuracy: {e}")
    
    def should_exit_position(self, position: Dict, df: pd.DataFrame) -> Dict:
        """
        Check if position should be exited using AI
        
        Args:
            position: Current position data
            df: OHLC DataFrame
            
        Returns:
            Dict: Exit signal
        """
        if len(df) < 10:
            return {'exit': False}
        
        try:
            # Get AI prediction for current market
            analysis = self.analyze(df)
            
            if analysis['signal'] == 'error':
                return {'exit': False}
            
            current_price = df['close'].iloc[-1]
            entry_price = position['price_open']
            
            # Exit if AI predicts opposite direction with high confidence
            if analysis['confidence'] > 0.7:
                if (position['type'] == 'buy' and analysis['signal'] == 'sell'):
                    return {
                        'exit': True,
                        'reason': 'AI predicts bearish reversal',
                        'exit_price': current_price
                    }
                elif (position['type'] == 'sell' and analysis['signal'] == 'buy'):
                    return {
                        'exit': True,
                        'reason': 'AI predicts bullish reversal',
                        'exit_price': current_price
                    }
            
            # Exit if confidence drops significantly
            if analysis['confidence'] < 0.3:
                return {
                    'exit': True,
                    'reason': 'AI confidence too low',
                    'exit_price': current_price
                }
            
            # Traditional exit conditions
            profit_threshold = 0.01  # 1% profit
            loss_threshold = -0.005  # 0.5% loss
            
            if position['type'] == 'buy':
                if current_price > entry_price * (1 + profit_threshold):
                    return {
                        'exit': True,
                        'reason': 'Profit target reached',
                        'exit_price': current_price
                    }
                elif current_price < entry_price * (1 + loss_threshold):
                    return {
                        'exit': True,
                        'reason': 'Stop loss triggered',
                        'exit_price': current_price
                    }
            
            elif position['type'] == 'sell':
                if current_price < entry_price * (1 - profit_threshold):
                    return {
                        'exit': True,
                        'reason': 'Profit target reached',
                        'exit_price': current_price
                    }
                elif current_price > entry_price * (1 - loss_threshold):
                    return {
                        'exit': True,
                        'reason': 'Stop loss triggered',
                        'exit_price': current_price
                    }
            
            return {'exit': False}
            
        except Exception as e:
            logger.error(f"Error in AI exit analysis: {e}")
            return {'exit': False, 'error': str(e)}
    
    def get_model_performance(self) -> Dict:
        """
        Get AI model performance metrics
        
        Returns:
            Dict: Performance metrics
        """
        return {
            'is_trained': self.is_trained,
            'prediction_accuracy': self.accuracy_metrics,
            'last_prediction': self.last_prediction,
            'model_metadata': self.model_manager.get_model_performance(),
            'total_predictions': len(self.prediction_history)
        }
    
    def get_strategy_info(self) -> Dict:
        """
        Get AI strategy information
        
        Returns:
            Dict: Strategy information
        """
        return {
            'name': self.name,
            'enabled': self.enabled,
            'magic_number': self.magic_number,
            'is_trained': self.is_trained,
            'prediction_horizon': self.prediction_horizon,
            'min_confidence_threshold': self.min_confidence_threshold,
            'risk_reward_ratio': self.risk_reward_ratio,
            'atr_multiplier': self.atr_multiplier,
            'accuracy_metrics': self.accuracy_metrics,
            'available_models': list(self.model_manager.models.keys())
        }
    
    def set_parameters(self, **kwargs):
        """
        Update strategy parameters
        
        Args:
            **kwargs: Parameter updates
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated {key} to {value}")
    
    def reset_models(self):
        """Reset all AI models"""
        try:
            self.model_manager.models.clear()
            self.model_manager.model_metadata.clear()
            self.is_trained = False
            self.prediction_history.clear()
            self.accuracy_metrics = {
                'total_predictions': 0,
                'correct_predictions': 0,
                'accuracy': 0.0
            }
            logger.info("AI models reset")
        except Exception as e:
            logger.error(f"Error resetting models: {e}")
