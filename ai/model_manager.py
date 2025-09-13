"""
Model Manager for AI Trading Bot
Handles machine learning model training, prediction, and management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import joblib
import os
import json

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages AI trading models"""
    
    def __init__(self, model_dir: str = "ai/models"):
        self.model_dir = model_dir
        self.models = {}
        self.model_metadata = {}
        self.performance_history = {}
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            },
            'gradient_boosting': {
                'class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 50,  # Reduced for faster training
                    'learning_rate': 0.1,
                    'max_depth': 4,  # Reduced for faster training
                    'random_state': 42,
                    'max_features': 'sqrt'  # Added for faster training
                }
            },
            'xgboost': {
                'class': xgb.XGBClassifier,
                'params': {
                    'n_estimators': 50,  # Reduced for faster training
                    'max_depth': 4,  # Reduced for faster training
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'n_jobs': -1  # Use all CPU cores
                }
            },
            'lightgbm': {
                'class': lgb.LGBMClassifier,
                'params': {
                    'n_estimators': 50,  # Reduced for faster training
                    'max_depth': 4,  # Reduced for faster training
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'n_jobs': -1,  # Use all CPU cores
                    'verbosity': -1
                }
            },
            'svm': {
                'class': SVC,
                'params': {
                    'kernel': 'rbf',
                    'C': 0.5,  # Slightly higher C for better fit while stable
                    'gamma': 'scale',
                    'probability': True,
                    'random_state': 42,
                    'max_iter': -1,  # Unlimited iterations to avoid early termination
                    'tol': 5e-4,  # Slightly stricter tolerance
                    'cache_size': 500  # Larger kernel cache (MB) speeds convergence
                }
            },
            'neural_network': {
                'class': MLPClassifier,
                'params': {
                    'hidden_layer_sizes': (50, 25),  # Reduced for faster training
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.001,
                    'random_state': 42,
                    'max_iter': 500,  # Increased iterations for better convergence
                    'tol': 1e-4,  # Relaxed tolerance
                    'learning_rate_init': 0.001,  # Lower learning rate for stability
                    'early_stopping': True,  # Enable early stopping
                    'validation_fraction': 0.1,  # 10% for validation
                    'n_iter_no_change': 10  # Stop if no improvement for 10 iterations
                }
            }
        }
        
        # Ensemble model
        self.ensemble_model = None
        
    def create_models(self, model_types: List[str] = None) -> Dict[str, Any]:
        """
        Create and initialize models
        
        Args:
            model_types: List of model types to create
            
        Returns:
            Dict: Created models
        """
        if model_types is None:
            model_types = list(self.model_configs.keys())
        
        created_models = {}
        
        for model_type in model_types:
            if model_type not in self.model_configs:
                logger.warning(f"Unknown model type: {model_type}")
                continue
            
            try:
                config = self.model_configs[model_type]
                model = config['class'](**config['params'])
                created_models[model_type] = model
                logger.info(f"Created {model_type} model")
                
            except Exception as e:
                logger.error(f"Error creating {model_type} model: {e}")
        
        self.models.update(created_models)
        return created_models
    
    def train_models(self, X: np.ndarray, y: np.ndarray, 
                    validation_split: float = 0.2) -> Dict[str, Dict]:
        """
        Train all models
        
        Args:
            X: Feature matrix
            y: Target vector
            validation_split: Validation split ratio
            
        Returns:
            Dict: Training results
        """
        if X.size == 0 or len(y) == 0:
            logger.error("Empty training data")
            return {}
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        training_results = {}
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Training {model_name} model...")
                
                # Use smaller dataset for faster training
                if len(X_train) > 5000:
                    # Sample data for faster training
                    sample_size = min(5000, len(X_train))
                    indices = np.random.choice(len(X_train), sample_size, replace=False)
                    X_train_sample = X_train[indices]
                    y_train_sample = y_train[indices]
                    logger.info(f"Using {sample_size} samples for faster training")
                else:
                    X_train_sample = X_train
                    y_train_sample = y_train
                
                # Train model (wrap SVM/NN with a scaling pipeline for consistency)
                start_time = datetime.now()

                if model_name in ['svm', 'neural_network']:
                    from sklearn.pipeline import make_pipeline
                    if model_name == 'svm':
                        from sklearn.preprocessing import MinMaxScaler
                        scaler = MinMaxScaler()
                    else:
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                    pipeline_model = make_pipeline(scaler, model)
                    pipeline_model.fit(X_train_sample, y_train_sample)
                    # Persist the pipeline so prediction/cv use identical preprocessing
                    self.models[model_name] = pipeline_model
                    fitted_model = pipeline_model
                else:
                    model.fit(X_train_sample, y_train_sample)
                    fitted_model = model

                training_time = (datetime.now() - start_time).total_seconds()
                
                # Evaluate model
                train_score = self.models[model_name].score(X_train, y_train)
                val_score = self.models[model_name].score(X_val, y_val)
                
                # Cross-validation score (pipeline handles preprocessing if needed)
                cv_scores = cross_val_score(self.models[model_name], X_train, y_train, cv=5)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Store results
                training_results[model_name] = {
                    'train_score': train_score,
                    'val_score': val_score,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'training_time': training_time,
                    'model': model
                }
                
                # Store metadata
                self.model_metadata[model_name] = {
                    'created_at': datetime.now().isoformat(),
                    'train_score': train_score,
                    'val_score': val_score,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'training_time': training_time,
                    'n_features': X.shape[1],
                    'n_samples': len(X)
                }
                
                logger.info(f"{model_name} - Train: {train_score:.3f}, Val: {val_score:.3f}, CV: {cv_mean:.3f}±{cv_std:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                training_results[model_name] = {'error': str(e)}
        
        # Create ensemble model
        self._create_ensemble_model()
        
        return training_results
    
    def _create_ensemble_model(self):
        """Create ensemble model from trained models"""
        try:
            # Get best performing models
            valid_models = []
            for name, model in self.models.items():
                if hasattr(model, 'predict_proba') and hasattr(model, 'fit'):
                    valid_models.append((name, model))
            
            if len(valid_models) < 2:
                logger.warning("Not enough valid models for ensemble")
                return
            
            # Create voting classifier
            estimators = [(name, model) for name, model in valid_models]
            self.ensemble_model = VotingClassifier(
                estimators=estimators,
                voting='soft'  # Use predicted probabilities
            )
            
            logger.info(f"Created ensemble model with {len(valid_models)} models")
            
        except Exception as e:
            logger.error(f"Error creating ensemble model: {e}")
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray):
        """Train ensemble model"""
        if self.ensemble_model is None:
            logger.warning("No ensemble model to train")
            return
        
        try:
            logger.info("Training ensemble model...")
            start_time = datetime.now()
            
            self.ensemble_model.fit(X, y)
            
            training_time = (datetime.now() - start_time).total_seconds()
            ensemble_score = self.ensemble_model.score(X, y)
            
            self.model_metadata['ensemble'] = {
                'created_at': datetime.now().isoformat(),
                'score': ensemble_score,
                'training_time': training_time,
                'n_models': len(self.ensemble_model.estimators_)
            }
            
            logger.info(f"Ensemble model trained - Score: {ensemble_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
    
    def predict(self, X: np.ndarray, model_name: str = 'ensemble') -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions
        
        Args:
            X: Feature matrix
            model_name: Model to use for prediction
            
        Returns:
            Tuple: (predictions, probabilities)
        """
        if X.size == 0:
            return np.array([]), np.array([])
        
        # Ensure contiguous array to avoid LightGBM subset warnings
        try:
            if isinstance(X, pd.DataFrame):
                X_use = X.values.copy()
            else:
                X_use = np.ascontiguousarray(X)
        except Exception:
            X_use = X
        
        try:
            if model_name == 'ensemble' and self.ensemble_model is not None:
                model = self.ensemble_model
            elif model_name in self.models:
                model = self.models[model_name]
            else:
                # Fallback to best available model
                best_name, best_model = self.get_best_model()
                if best_model is None:
                    logger.error(f"Model {model_name} not found and no fallback available")
                    return np.array([]), np.array([])
                model = best_model

            # Try prediction; if model (e.g., ensemble) is not fitted, try manual soft-ensemble, then fallback
            probabilities = None
            try:
                predictions = model.predict(X_use)
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X_use)
            except Exception as e:
                logger.error(f"Error making predictions with {model_name}: {e}")
                # Try manual soft-ensemble across available models with predict_proba
                try:
                    pred_me, proba_me = self._manual_soft_ensemble(X_use)
                    if pred_me.size:
                        return pred_me, proba_me
                except Exception as e_me:
                    logger.warning(f"Manual ensemble failed: {e_me}")
                # Fallback to best single model
                best_name, best_model = self.get_best_model()
                if best_model is None or best_model is model:
                    return np.array([]), np.array([])
                try:
                    predictions = best_model.predict(X_use)
                    if hasattr(best_model, 'predict_proba'):
                        probabilities = best_model.predict_proba(X_use)
                except Exception as e2:
                    logger.error(f"Fallback model prediction failed: {e2}")
                    return np.array([]), np.array([])
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Error making predictions with {model_name}: {e}")
            return np.array([]), np.array([])

    def _manual_soft_ensemble(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Average predict_proba across available classifiers to emulate soft voting.
        Returns (predictions, averaged_probabilities). Assumes 3 classes [0,1,2].
        """
        import numpy as _np
        # Ensure contiguous
        try:
            if isinstance(X, pd.DataFrame):
                X_use = X.values.copy()
            else:
                X_use = _np.ascontiguousarray(X)
        except Exception:
            X_use = X
        probas = []
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    p = model.predict_proba(X_use)
                    # Align to 3 classes if model has class subset
                    cls = getattr(model, 'classes_', None)
                    if cls is not None and len(cls) != 3:
                        aligned = _np.zeros((p.shape[0], 3))
                        for idx, c in enumerate(cls):
                            if int(c) in (0,1,2):
                                aligned[:, int(c)] = p[:, idx]
                        p = aligned
                    probas.append(p)
            except Exception:
                continue
        if not probas:
            return _np.array([]), _np.array([])
        avg = _np.mean(probas, axis=0)
        preds = _np.argmax(avg, axis=1)
        return preds, avg
    
    def get_model_performance(self, model_name: str = None) -> Dict:
        """
        Get model performance metrics
        
        Args:
            model_name: Specific model name, or None for all models
            
        Returns:
            Dict: Performance metrics
        """
        if model_name:
            return self.model_metadata.get(model_name, {})
        else:
            return self.model_metadata

    # --- Explainability helpers ---
    def get_global_feature_importance(self, feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Return global feature importances per model when supported."""
        importances: Dict[str, Dict[str, float]] = {}
        try:
            for name, model in self.models.items():
                imp_map: Dict[str, float] = {}
                try:
                    if hasattr(model, 'feature_importances_'):
                        vals = model.feature_importances_
                        for i, f in enumerate(feature_names[:len(vals)]):
                            imp_map[f] = float(vals[i])
                    elif hasattr(model, 'coef_'):
                        coef = getattr(model, 'coef_', None)
                        if coef is not None:
                            arr = coef[0] if len(coef.shape) > 1 else coef
                            for i, f in enumerate(feature_names[:len(arr)]):
                                imp_map[f] = abs(float(arr[i]))
                except Exception:
                    imp_map = {}
                if imp_map:
                    # Normalize to sum=1
                    s = sum(imp_map.values()) or 1.0
                    importances[name] = {k: v / s for k, v in imp_map.items()}
        except Exception:
            pass
        return importances

    def get_local_contributions(self, X: np.ndarray, feature_names: List[str], model_name: str = None) -> Optional[Dict[str, float]]:
        """Return per-sample feature contributions using model-native pred_contribs when available.
        Supports xgboost and lightgbm. Falls back to None if unsupported.
        """
        try:
            target_model = None
            if model_name and model_name in self.models:
                target_model = self.models[model_name]
            elif self.ensemble_model is not None:
                # fallback to best individual model for contributions
                best_name, best_model = self.get_best_model()
                target_model = best_model
            # XGBoost
            try:
                import xgboost as _xgb
                if hasattr(target_model, 'get_booster'):
                    dmx = _xgb.DMatrix(X)
                    contribs = target_model.get_booster().predict(dmx, pred_contribs=True)
                    vals = contribs[0]
                    return {feature_names[i]: float(vals[i]) for i in range(min(len(vals), len(feature_names)))}
            except Exception:
                pass
            # LightGBM
            try:
                if hasattr(target_model, 'booster_'):
                    vals = target_model.booster_.predict(X, pred_contrib=True)
                    row = vals[0]
                    return {feature_names[i]: float(row[i]) for i in range(min(len(row), len(feature_names)))}
            except Exception:
                pass
            return None
        except Exception:
            return None

    def explain(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        """Produce a simple explanation: top positive/negative contributors if available."""
        try:
            contribs = self.get_local_contributions(X, feature_names)
            if not contribs:
                # Fall back to global importances
                global_imps = self.get_global_feature_importance(feature_names) or {}
                # Merge average across models
                avg: Dict[str, float] = {}
                for _, m in global_imps.items():
                    for f, v in m.items():
                        avg[f] = avg.get(f, 0.0) + v
                if avg:
                    total_models = max(1, len(global_imps))
                    for f in list(avg.keys()):
                        avg[f] /= total_models
                    top = sorted(avg.items(), key=lambda x: x[1], reverse=True)[:5]
                    return {'top_positive': top, 'top_negative': []}
                return {'top_positive': [], 'top_negative': []}
            # Split positive/negative
            items = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
            top_pos = [(k, v) for k, v in items if v > 0][:5]
            top_neg = [(k, v) for k, v in reversed(items) if v < 0][:5]
            return {'top_positive': top_pos, 'top_negative': top_neg}
        except Exception:
            return {'top_positive': [], 'top_negative': []}
    
    def hyperparameter_tuning(self, X: np.ndarray, y: np.ndarray, 
                             model_name: str, param_grid: Dict) -> Dict:
        """
        Perform hyperparameter tuning
        
        Args:
            X: Feature matrix
            y: Target vector
            model_name: Model to tune
            param_grid: Parameter grid for tuning
            
        Returns:
            Dict: Tuning results
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return {}
        
        try:
            logger.info(f"Starting hyperparameter tuning for {model_name}")
            
            model = self.models[model_name]
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            # Update model with best parameters
            self.models[model_name] = grid_search.best_estimator_
            
            results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning for {model_name}: {e}")
            return {}
    
    def create_lstm_model(self, input_shape: Tuple, num_classes: int = 3) -> Optional[Any]:
        """
        Create LSTM model for time series prediction
        
        Args:
            input_shape: Input shape (timesteps, features)
            num_classes: Number of output classes
            
        Returns:
            LSTM model or None
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available for LSTM model")
            return None
        
        try:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("Created LSTM model")
            return model
            
        except Exception as e:
            logger.error(f"Error creating LSTM model: {e}")
            return None
    
    def train_lstm_model(self, X: np.ndarray, y: np.ndarray, 
                        model: Any, epochs: int = 100) -> Dict:
        """
        Train LSTM model
        
        Args:
            X: Feature matrix (3D: samples, timesteps, features)
            y: Target vector
            model: LSTM model
            epochs: Number of training epochs
            
        Returns:
            Dict: Training results
        """
        if not TENSORFLOW_AVAILABLE or model is None:
            return {'error': 'LSTM model not available'}
        
        try:
            # Callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]
            
            # Train model
            history = model.fit(
                X, y,
                epochs=epochs,
                batch_size=32,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            # Store model
            self.models['lstm'] = model
            
            # Store metadata
            self.model_metadata['lstm'] = {
                'created_at': datetime.now().isoformat(),
                'epochs_trained': len(history.history['loss']),
                'final_loss': history.history['loss'][-1],
                'final_accuracy': history.history['accuracy'][-1],
                'val_loss': history.history['val_loss'][-1],
                'val_accuracy': history.history['val_accuracy'][-1]
            }
            
            logger.info(f"LSTM model trained - Final accuracy: {history.history['accuracy'][-1]:.3f}")
            
            return {
                'history': history.history,
                'final_accuracy': history.history['accuracy'][-1],
                'final_val_accuracy': history.history['val_accuracy'][-1]
            }
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return {'error': str(e)}
    
    def save_models(self, model_names: List[str] = None):
        """
        Save models to disk
        
        Args:
            model_names: List of model names to save, or None for all
        """
        if model_names is None:
            model_names = list(self.models.keys())
        
        for model_name in model_names:
            if model_name not in self.models:
                continue
            
            try:
                model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
                
                if model_name == 'lstm' and TENSORFLOW_AVAILABLE:
                    # Save TensorFlow model
                    model_path = os.path.join(self.model_dir, f"{model_name}.h5")
                    self.models[model_name].save(model_path)
                else:
                    # Save scikit-learn model
                    joblib.dump(self.models[model_name], model_path)
                
                logger.info(f"Saved {model_name} model successfully")
                
            except Exception as e:
                logger.error(f"Error saving {model_name} model: {e}")
        
        # Save ensemble model if it exists and is fitted
        if self.ensemble_model is not None and hasattr(self.ensemble_model, 'estimators_'):
            try:
                ensemble_path = os.path.join(self.model_dir, "ensemble_model.joblib")
                joblib.dump(self.ensemble_model, ensemble_path)
                logger.info("Saved ensemble model successfully")
            except Exception as e:
                logger.error(f"Error saving ensemble model: {e}")
        
        # Save metadata
        metadata_path = os.path.join(self.model_dir, "model_metadata.json")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self.model_metadata, f, indent=2)
            logger.info("Saved model metadata successfully")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def load_models(self, model_names: List[str] = None) -> Dict[str, bool]:
        """
        Load models from disk
        
        Args:
            model_names: List of model names to load, or None for all
            
        Returns:
            Dict: Loading results
        """
        if model_names is None:
            # Try to load all available models
            model_files = [f for f in os.listdir(self.model_dir) if f.endswith(('.joblib', '.h5'))]
            model_names = [f.split('.')[0] for f in model_files]
        
        loading_results = {}
        
        for model_name in model_names:
            try:
                if model_name == 'lstm' and TENSORFLOW_AVAILABLE:
                    # Load TensorFlow model
                    model_path = os.path.join(self.model_dir, f"{model_name}.h5")
                    if os.path.exists(model_path):
                        self.models[model_name] = tf.keras.models.load_model(model_path)
                        loading_results[model_name] = True
                    else:
                        loading_results[model_name] = False
                else:
                    # Load scikit-learn model
                    model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
                    if os.path.exists(model_path):
                        self.models[model_name] = joblib.load(model_path)
                        loading_results[model_name] = True
                    else:
                        loading_results[model_name] = False
                
                if loading_results[model_name]:
                    logger.info(f"Loaded {model_name} model")
                
            except Exception as e:
                logger.error(f"Error loading {model_name} model: {e}")
                loading_results[model_name] = False
        
        # Load ensemble model if it exists
        ensemble_path = os.path.join(self.model_dir, "ensemble_model.joblib")
        ensemble_loaded = False
        try:
            if os.path.exists(ensemble_path):
                self.ensemble_model = joblib.load(ensemble_path)
                logger.info("Loaded ensemble model")
                ensemble_loaded = True
            else:
                logger.info("No ensemble model found, will create new one if needed")
        except Exception as e:
            logger.error(f"Error loading ensemble model: {e}")
            self.ensemble_model = None
        
        # Create ensemble model if not loaded and we have individual models
        if not ensemble_loaded and self.models:
            self._create_ensemble_model()
        
        # Load metadata
        metadata_path = os.path.join(self.model_dir, "model_metadata.json")
        try:
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info("Loaded model metadata")
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
        
        return loading_results
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best performing model
        
        Returns:
            Tuple: (model_name, model)
        """
        if not self.model_metadata:
            return None, None
        
        # Find model with highest validation score
        best_model = None
        best_score = -1
        
        for model_name, metadata in self.model_metadata.items():
            if 'val_score' in metadata:
                score = metadata['val_score']
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        if best_model and best_model in self.models:
            return best_model, self.models[best_model]
        
        # Fallback to ensemble if available
        if self.ensemble_model is not None:
            return 'ensemble', self.ensemble_model
        
        return None, None
    
    def evaluate_model(self, X: np.ndarray, y: np.ndarray, 
                      model_name: str = 'ensemble') -> Dict:
        """
        Evaluate model performance
        
        Args:
            X: Feature matrix
            y: Target vector
            model_name: Model to evaluate
            
        Returns:
            Dict: Evaluation results
        """
        if model_name not in self.models and model_name != 'ensemble':
            return {'error': f'Model {model_name} not found'}
        
        try:
            model = self.models.get(model_name, self.ensemble_model)
            if model is None:
                return {'error': f'Model {model_name} not available'}
            
            # Make predictions
            predictions = model.predict(X)
            
            # Calculate metrics
            accuracy = accuracy_score(y, predictions)
            
            # Classification report
            report = classification_report(y, predictions, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y, predictions)
            
            results = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'model_name': model_name
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            return {'error': str(e)}
