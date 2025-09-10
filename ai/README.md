# AI Trading Bot - Advanced Machine Learning System

## Overview

The AI Trading Bot is an advanced machine learning system that enhances the traditional price action trading bot with sophisticated AI capabilities. It uses multiple machine learning models to analyze market data and make intelligent trading decisions.

## Features

### 🤖 AI Strategy Engine
- **Multiple ML Models**: Random Forest, XGBoost, LightGBM, SVM, Neural Networks, and LSTM
- **Ensemble Learning**: Combines multiple models for better predictions
- **Feature Engineering**: 100+ technical and price action features
- **Real-time Predictions**: Live market analysis with confidence scores

### 🎯 Auto-Training System
- **Automatic Retraining**: Models retrain based on performance and time intervals
- **Performance Monitoring**: Continuous accuracy tracking and alerts
- **Data Management**: Automatic historical data collection and preprocessing
- **Model Persistence**: Save and load trained models

### 📱 Enhanced Telegram Interface
- **AI Status Commands**: Check training status and model performance
- **Training Controls**: Manual and automatic training triggers
- **Performance Reports**: Detailed accuracy and performance metrics
- **Real-time Analysis**: AI-powered market analysis

## Architecture

```
ai/
├── __init__.py              # Module initialization
├── ai_strategy.py           # Main AI trading strategy
├── data_processor.py        # Feature engineering and data preprocessing
├── model_manager.py         # ML model training and management
├── auto_trainer.py          # Automatic training system
├── ai_telegram_bot.py       # AI-enhanced Telegram interface
└── README.md               # This file
```

## Components

### 1. AIStrategy (`ai_strategy.py`)
The core AI trading strategy that:
- Integrates with existing trading bot architecture
- Makes predictions using trained ML models
- Manages risk and position sizing
- Tracks prediction accuracy

### 2. DataProcessor (`data_processor.py`)
Advanced feature engineering including:
- **Price Features**: Momentum, ratios, shadows, body sizes
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic
- **Price Action Patterns**: Hammer, shooting star, doji, engulfing, pin bars
- **Support/Resistance**: Distance to levels, level strength
- **Trend Analysis**: Trend identification, higher highs/lower lows
- **Volatility Features**: Rolling volatility, volatility regimes
- **Time Features**: Market sessions, day of week, hour
- **Microstructure**: Volume analysis, gaps, tick data

### 3. ModelManager (`model_manager.py`)
Comprehensive ML model management:
- **Model Types**: Random Forest, XGBoost, LightGBM, SVM, Neural Networks, LSTM
- **Training Pipeline**: Cross-validation, hyperparameter tuning
- **Ensemble Methods**: Voting classifiers with soft voting
- **Model Persistence**: Save/load models and metadata
- **Performance Evaluation**: Accuracy metrics, confusion matrices

### 4. AutoTrainer (`auto_trainer.py`)
Intelligent training automation:
- **Scheduled Retraining**: Time-based model updates
- **Performance-Based Retraining**: Retrain when accuracy drops
- **Data Management**: Automatic historical data collection
- **Performance Monitoring**: Continuous accuracy tracking
- **Alert System**: Performance degradation alerts

### 5. AITelegramBot (`ai_telegram_bot.py`)
Enhanced Telegram interface with AI commands:
- `/ai_status` - Show AI strategy status
- `/ai_train` - Train AI models
- `/ai_retrain` - Retrain existing models
- `/ai_performance` - Show performance report
- `/ai_analyze` - Perform AI analysis
- `/ai_models` - List available models
- `/ai_config` - Show AI configuration
- `/ai_reset` - Reset AI models
- `/ai_auto_train` - Toggle auto-training

## Usage

### 1. Initial Setup
```python
from ai import AIStrategy, AutoTrainer, AITelegramBot

# Initialize AI strategy
ai_strategy = AIStrategy(mt5_connector)

# Initialize auto-trainer
auto_trainer = AutoTrainer(ai_strategy, mt5_connector)

# Start auto-training
auto_trainer.start_auto_training()
```

### 2. Training Models
```python
# Get historical data
historical_data = mt5_connector.get_rates('EURUSD', 'M15', 2000)

# Train models
result = ai_strategy.train_models(historical_data)
print(f"Training completed: {result['status']}")
```

### 3. Making Predictions
```python
# Get current market data
current_data = mt5_connector.get_rates('EURUSD', 'M15', 200)

# Analyze with AI
analysis = ai_strategy.analyze(current_data)
print(f"Signal: {analysis['signal']}, Confidence: {analysis['confidence']}")
```

### 4. Telegram Commands
```
/ai_status          # Check AI status
/ai_train 3000      # Train with 3000 data points
/ai_performance     # Show performance report
/ai_analyze         # Analyze current market
```

## Configuration

### AI Strategy Parameters
```python
ai_strategy.set_parameters(
    prediction_horizon=5,           # Predict 5 periods ahead
    min_confidence_threshold=0.6,   # Minimum confidence for trading
    risk_reward_ratio=2.0,         # Risk to reward ratio
    atr_multiplier=1.5             # ATR multiplier for stop loss
)
```

### Auto-Trainer Settings
```python
auto_trainer.update_config(
    retrain_interval_hours=24,      # Retrain every 24 hours
    performance_threshold=0.55,     # Retrain if accuracy < 55%
    min_data_points=1000,          # Minimum data for training
    auto_retrain_enabled=True      # Enable automatic retraining
)
```

## Model Performance

### Supported Models
1. **Random Forest**: Robust ensemble method, good for feature importance
2. **XGBoost**: Gradient boosting, excellent for tabular data
3. **LightGBM**: Fast gradient boosting, memory efficient
4. **SVM**: Support Vector Machine, good for classification
5. **Neural Network**: Multi-layer perceptron, non-linear patterns
6. **LSTM**: Long Short-Term Memory, time series patterns
7. **Ensemble**: Voting classifier combining all models

### Performance Metrics
- **Accuracy**: Overall prediction accuracy
- **Cross-Validation**: 5-fold CV scores
- **Confidence Scores**: Prediction confidence levels
- **Feature Importance**: Most important features for predictions

## Data Requirements

### Minimum Data
- **Training**: 1000+ historical data points
- **Features**: 100+ engineered features
- **Timeframe**: M15 recommended for best results
- **Symbols**: Multi-symbol training supported

### Data Quality
- Clean OHLC data without gaps
- Sufficient market volatility
- Recent data for better performance
- Multiple market conditions

## Best Practices

### 1. Training
- Train with at least 2000 data points
- Use multiple symbols for robustness
- Retrain regularly (daily/weekly)
- Monitor performance metrics

### 2. Trading
- Use confidence thresholds (0.6+ recommended)
- Combine with traditional strategies
- Monitor prediction accuracy
- Adjust parameters based on performance

### 3. Risk Management
- Set appropriate stop losses
- Use position sizing based on confidence
- Monitor drawdowns
- Regular performance reviews

## Troubleshooting

### Common Issues

1. **Models Not Training**
   - Check data availability
   - Verify feature engineering
   - Ensure sufficient data points

2. **Low Accuracy**
   - Increase training data
   - Adjust feature selection
   - Try different models
   - Check data quality

3. **Memory Issues**
   - Reduce data points
   - Use feature selection
   - Enable model persistence
   - Monitor system resources

### Performance Optimization

1. **Feature Selection**: Use SelectKBest for feature reduction
2. **Model Persistence**: Save/load models to avoid retraining
3. **Batch Processing**: Process data in batches for large datasets
4. **Parallel Training**: Use multiple cores for model training

## Future Enhancements

### Planned Features
- **Reinforcement Learning**: Q-learning for trading decisions
- **Deep Learning**: CNN for pattern recognition
- **Sentiment Analysis**: News and social media integration
- **Portfolio Optimization**: Multi-asset allocation
- **Real-time Learning**: Online learning algorithms

### Advanced Models
- **Transformer Models**: Attention-based architectures
- **GANs**: Generative adversarial networks for data augmentation
- **AutoML**: Automated model selection and hyperparameter tuning
- **Federated Learning**: Distributed training across multiple bots

## Support

For issues and questions:
1. Check the logs for error messages
2. Verify data quality and availability
3. Ensure all dependencies are installed
4. Review configuration parameters
5. Check system resources (CPU, memory)

## License

This AI system is part of the Price Action Trading Bot project. Please refer to the main project license for usage terms.
