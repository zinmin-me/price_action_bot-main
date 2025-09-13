"""
Auto Trainer for AI Trading Bot
Handles automatic model training and retraining based on market data and performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import threading
import time
import schedule
import os
import json

from config import *
from .ai_strategy import AIStrategy
from .data_processor import DataProcessor
from .model_manager import ModelManager

logger = logging.getLogger(__name__)

class AutoTrainer:
    """Automatic training and retraining system for AI models"""
    
    def __init__(self, ai_strategy: AIStrategy, mt5_connector):
        self.ai_strategy = ai_strategy
        self.mt5 = mt5_connector
        self.is_running = False
        self.training_thread = None
        
        # Training configuration
        self.retrain_interval_hours = AI_RETRAIN_INTERVAL_HOURS
        self.min_data_points = AI_MIN_TRAIN_SAMPLES
        self.performance_threshold = AI_PERF_THRESHOLD
        self.max_retrain_attempts = 3
        
        # Training history
        self.training_history = []
        self.last_training_time = None
        self.training_stats = {
            'total_trainings': 0,
            'successful_trainings': 0,
            'failed_trainings': 0,
            'last_accuracy': 0.0,
            'best_accuracy': 0.0
        }
        
        # Performance monitoring
        self.performance_window = AI_DRIFT_WINDOW
        self.recent_predictions = []
        self.performance_alerts = []
        
        # Auto-training triggers
        self.auto_retrain_enabled = True
        self.performance_based_retrain = True
        self.time_based_retrain = True
        
    def start_auto_training(self):
        """Start automatic training system"""
        if self.is_running:
            logger.warning("Auto trainer is already running")
            return
        
        self.is_running = True
        
        # Schedule training tasks
        if self.time_based_retrain:
            schedule.every(self.retrain_interval_hours).hours.do(self._scheduled_retrain)
        
        # Start monitoring thread
        self.training_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.training_thread.start()
        
        logger.info("Auto trainer started")
    
    def stop_auto_training(self):
        """Stop automatic training system"""
        self.is_running = False
        schedule.clear()
        
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5)
        
        logger.info("Auto trainer stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop for auto-training"""
        while self.is_running:
            try:
                # Check if retraining is needed
                if self.auto_retrain_enabled:
                    self._check_retrain_conditions()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Sleep for 1 minute
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def _check_retrain_conditions(self):
        """Check if models need retraining"""
        try:
            # Check performance-based retraining
            if self.performance_based_retrain:
                current_accuracy = self.ai_strategy.accuracy_metrics.get('accuracy', 0.0)
                if (current_accuracy < self.performance_threshold and 
                    self.ai_strategy.accuracy_metrics.get('total_predictions', 0) > AI_MIN_PREDICTIONS_FOR_PERF):
                    logger.info(f"Performance below threshold ({current_accuracy:.3f} < {self.performance_threshold})")
                    self._trigger_retrain("performance_based")
            
            # Check time-based retraining
            if self.time_based_retrain:
                schedule.run_pending()
            
        except Exception as e:
            logger.error(f"Error checking retrain conditions: {e}")
    
    def _scheduled_retrain(self):
        """Scheduled retraining task"""
        logger.info("Scheduled retraining triggered")
        self._trigger_retrain("scheduled")
    
    def _trigger_retrain(self, trigger_type: str):
        """Trigger model retraining"""
        try:
            logger.info(f"Triggering retrain: {trigger_type}")
            
            # Get historical data
            historical_data = self._get_historical_data()
            
            if historical_data is None or len(historical_data) < self.min_data_points:
                logger.warning(f"Insufficient data for retraining: {len(historical_data) if historical_data is not None else 0} < {self.min_data_points}")
                return
            
            # Perform retraining
            result = self.ai_strategy.retrain_models(historical_data)
            
            # Update training history
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'trigger_type': trigger_type,
                'result': result,
                'data_points': len(historical_data),
                'accuracy_before': self.training_stats['last_accuracy'],
                'accuracy_after': self.ai_strategy.accuracy_metrics.get('accuracy', 0.0)
            }
            
            self.training_history.append(training_record)
            self.last_training_time = datetime.now()
            
            # Update stats
            self.training_stats['total_trainings'] += 1
            if 'error' not in result:
                self.training_stats['successful_trainings'] += 1
                self.training_stats['last_accuracy'] = self.ai_strategy.accuracy_metrics.get('accuracy', 0.0)
                
                if self.training_stats['last_accuracy'] > self.training_stats['best_accuracy']:
                    self.training_stats['best_accuracy'] = self.training_stats['last_accuracy']
            else:
                self.training_stats['failed_trainings'] += 1
            
            logger.info(f"Retraining completed: {result.get('status', 'failed')}")
            
        except Exception as e:
            logger.error(f"Error in retrain trigger: {e}")
            self.training_stats['failed_trainings'] += 1
    
    def _get_historical_data(self, periods: int = 2000) -> Optional[pd.DataFrame]:
        """
        Get historical data for training
        
        Args:
            periods: Number of periods to retrieve
            
        Returns:
            DataFrame: Historical OHLC data
        """
        try:
            # Get data for all configured symbols
            all_data = []
            
            for symbol in SYMBOLS:
                try:
                    # Switch to symbol
                    self.mt5.change_symbol(symbol)
                    
                    # Get historical data
                    df = self.mt5.get_rates(symbol, TIMEFRAME, periods)
                    
                    if df is not None and not df.empty:
                        df['symbol'] = symbol
                        all_data.append(df)
                        
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {e}")
                    continue
            
            if not all_data:
                logger.error("No historical data available")
                return None
            
            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Sort by time
            combined_data = combined_data.sort_values('time' if 'time' in combined_data.columns else combined_data.index)
            
            # Remove duplicates
            combined_data = combined_data.drop_duplicates()
            
            logger.info(f"Retrieved {len(combined_data)} historical data points")
            return combined_data
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return None
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Get recent predictions
            recent_predictions = self.ai_strategy.prediction_history[-self.performance_window:]
            
            if len(recent_predictions) < 10:
                return
            
            # Estimate drift via recent loss rate using accuracy metrics deltas
            total_predictions = self.ai_strategy.accuracy_metrics.get('total_predictions', 0)
            correct_predictions = self.ai_strategy.accuracy_metrics.get('correct_predictions', 0)
            # Naive drift proxy: if accuracy over the window would imply high loss rate, alert
            if total_predictions > 0:
                current_accuracy = self.ai_strategy.accuracy_metrics.get('accuracy', 0.0)
                loss_rate = 1.0 - current_accuracy
                if AI_DRIFT_CHECK_ENABLED and loss_rate >= AI_DRIFT_LOSS_RATE_ALERT and total_predictions >= AI_MIN_PREDICTIONS_FOR_PERF:
                    alert = {
                        'timestamp': datetime.now().isoformat(),
                        'type': 'drift_suspected',
                        'loss_rate': loss_rate,
                        'threshold': AI_DRIFT_LOSS_RATE_ALERT
                    }
                    self.performance_alerts.append(alert)
                    logger.warning(f"AI drift suspected: loss_rate={loss_rate:.3f} >= {AI_DRIFT_LOSS_RATE_ALERT}")
                    if AI_AUTORETRAIN_ON_DRIFT:
                        self._trigger_retrain("drift_based")
            
            # Check for performance alerts
            current_accuracy = self.ai_strategy.accuracy_metrics.get('accuracy', 0.0)
            
            if current_accuracy < 0.4:  # Very low accuracy
                alert = {
                    'timestamp': datetime.now().isoformat(),
                    'type': 'low_accuracy',
                    'value': current_accuracy,
                    'threshold': 0.4
                }
                self.performance_alerts.append(alert)
                logger.warning(f"Performance alert: Accuracy {current_accuracy:.3f} below 0.4")
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def manual_retrain(self, data_periods: int = 2000) -> Dict:
        """
        Manually trigger retraining
        
        Args:
            data_periods: Number of periods to use for training
            
        Returns:
            Dict: Training results
        """
        try:
            logger.info("Manual retraining triggered")
            
            # Get historical data
            historical_data = self._get_historical_data(data_periods)
            
            if historical_data is None:
                return {'error': 'Failed to get historical data'}
            
            # Perform training
            result = self.ai_strategy.train_models(historical_data)
            
            # Update training history
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'trigger_type': 'manual',
                'result': result,
                'data_points': len(historical_data)
            }
            
            self.training_history.append(training_record)
            self.last_training_time = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in manual retrain: {e}")
            return {'error': str(e)}
    
    def get_training_status(self) -> Dict:
        """
        Get current training status
        
        Returns:
            Dict: Training status information
        """
        return {
            'is_running': self.is_running,
            'auto_retrain_enabled': self.auto_retrain_enabled,
            'performance_based_retrain': self.performance_based_retrain,
            'time_based_retrain': self.time_based_retrain,
            'retrain_interval_hours': self.retrain_interval_hours,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'training_stats': self.training_stats,
            'recent_alerts': self.performance_alerts[-5:],  # Last 5 alerts
            'ai_strategy_trained': self.ai_strategy.is_trained,
            'ai_accuracy': self.ai_strategy.accuracy_metrics.get('accuracy', 0.0)
        }
    
    def update_config(self, **kwargs):
        """
        Update auto-trainer configuration
        
        Args:
            **kwargs: Configuration updates
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(f"Updated {key} to {value}")
    
    def save_training_history(self, filepath: str = None):
        """
        Save training history to file
        
        Args:
            filepath: Path to save file
        """
        if filepath is None:
            filepath = os.path.join(self.ai_strategy.model_manager.model_dir, "training_history.json")
        
        try:
            history_data = {
                'training_history': self.training_history,
                'training_stats': self.training_stats,
                'performance_alerts': self.performance_alerts,
                'last_saved': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            logger.info("Saved training history successfully")
            
        except Exception as e:
            logger.error(f"Error saving training history: {e}")
    
    def load_training_history(self, filepath: str = None):
        """
        Load training history from file
        
        Args:
            filepath: Path to load file
        """
        if filepath is None:
            filepath = os.path.join(self.ai_strategy.model_manager.model_dir, "training_history.json")
        
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Training history file not found: {filepath}")
                return
            
            with open(filepath, 'r') as f:
                history_data = json.load(f)
            
            self.training_history = history_data.get('training_history', [])
            self.training_stats = history_data.get('training_stats', self.training_stats)
            self.performance_alerts = history_data.get('performance_alerts', [])
            
            logger.info("Loaded training history successfully")
            
        except Exception as e:
            logger.error(f"Error loading training history: {e}")
    
    def get_performance_report(self) -> Dict:
        """
        Generate performance report
        
        Returns:
            Dict: Performance report
        """
        try:
            # Calculate metrics
            total_trainings = self.training_stats['total_trainings']
            success_rate = (self.training_stats['successful_trainings'] / max(total_trainings, 1)) * 100
            
            # Recent performance
            recent_trainings = self.training_history[-10:] if self.training_history else []
            recent_success_rate = 0
            if recent_trainings:
                recent_successes = sum(1 for t in recent_trainings if 'error' not in t.get('result', {}))
                recent_success_rate = (recent_successes / len(recent_trainings)) * 100
            
            # Performance trends
            accuracy_trend = "stable"
            if len(self.training_history) >= 2:
                recent_accuracy = self.training_history[-1].get('accuracy_after', 0)
                previous_accuracy = self.training_history[-2].get('accuracy_after', 0)
                
                if recent_accuracy > previous_accuracy + 0.05:
                    accuracy_trend = "improving"
                elif recent_accuracy < previous_accuracy - 0.05:
                    accuracy_trend = "declining"
            
            report = {
                'summary': {
                    'total_trainings': total_trainings,
                    'success_rate': success_rate,
                    'recent_success_rate': recent_success_rate,
                    'best_accuracy': self.training_stats['best_accuracy'],
                    'current_accuracy': self.training_stats['last_accuracy'],
                    'accuracy_trend': accuracy_trend
                },
                'recent_trainings': recent_trainings,
                'performance_alerts': self.performance_alerts[-10:],
                'recommendations': self._generate_recommendations()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on performance"""
        recommendations = []
        
        try:
            # Check accuracy
            current_accuracy = self.training_stats['last_accuracy']
            if current_accuracy < 0.5:
                recommendations.append("Consider increasing training data or adjusting model parameters")
            
            # Check success rate
            success_rate = (self.training_stats['successful_trainings'] / max(self.training_stats['total_trainings'], 1)) * 100
            if success_rate < 70:
                recommendations.append("Training success rate is low - check data quality and model configuration")
            
            # Check retrain frequency
            if self.training_stats['total_trainings'] > 10:
                recent_trainings = self.training_history[-5:] if len(self.training_history) >= 5 else self.training_history
                if recent_trainings:
                    time_diffs = []
                    for i in range(1, len(recent_trainings)):
                        prev_time = datetime.fromisoformat(recent_trainings[i-1]['timestamp'])
                        curr_time = datetime.fromisoformat(recent_trainings[i]['timestamp'])
                        time_diffs.append((curr_time - prev_time).total_seconds() / 3600)  # hours
                    
                    avg_interval = sum(time_diffs) / len(time_diffs) if time_diffs else 0
                    if avg_interval < 12:
                        recommendations.append("Consider reducing retrain frequency - models may be overfitting")
            
            # Check alerts
            recent_alerts = [a for a in self.performance_alerts if 
                           (datetime.now() - datetime.fromisoformat(a['timestamp'])).total_seconds() < 86400]  # Last 24 hours
            if len(recent_alerts) > 3:
                recommendations.append("Multiple performance alerts detected - review model performance and data quality")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
