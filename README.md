# activate venv
.\venv\Scripts\activate

# run 
python main.py

# Price Action Trading Bot

An advanced algorithmic trading bot that uses multiple price action strategies with enhanced risk management, AI integration, and comprehensive database management for MetaTrader 5.

## 🚀 Features

### Core Trading Strategies
- **Breakout Strategy**: Detects and trades price breakouts from consolidation patterns with enhanced confirmation
- **Reversal Patterns**: Identifies candlestick reversal patterns (hammer, shooting star, engulfing, etc.)
- **Support & Resistance**: Trades bounces off key support and resistance levels
- **Trend Following**: Follows established trends with pullback entries
- **AI Strategy**: Machine learning-powered trading decisions with 6+ ML models

### Enhanced Risk Management
- **Dynamic Position Sizing**: Adjusts position size based on signal confidence and market volatility
- **Multi-factor Confirmation**: Requires multiple criteria before entering trades
- **Trailing Stops**: Automatic profit protection
- **Session-based Trading**: Optimized for different market sessions
- **Volatility Adjustment**: Reduces risk during high volatility periods

### AI/ML Capabilities
- **Multiple ML Models**: Random Forest, XGBoost, LightGBM, SVM, Neural Networks, LSTM
- **Ensemble Learning**: Combines multiple models for better predictions
- **Feature Engineering**: 100+ technical and price action features
- **Auto-Training**: Models retrain automatically based on performance
- **Real-time Predictions**: Live market analysis with confidence scores

### Technical Features
- **Multi-symbol Trading**: Trade multiple currency pairs simultaneously
- **Real-time Analysis**: Live market data processing
- **Telegram Integration**: Remote monitoring and control with AI commands
- **Comprehensive Logging**: Detailed trade and performance logs
- **Database Storage**: Persistent trade history and user management
- **User Authorization**: Secure access control with MT5 account associations

## 📊 Recent Performance Improvements

### Problem Analysis
The bot was experiencing significant losses due to:
- False breakout signals
- Poor risk-reward ratios
- Lack of market context analysis
- Inadequate confirmation criteria

### Solutions Implemented
- **Enhanced Breakout Detection**: 70% confirmation threshold with 6 validation criteria
- **Improved Reversal Patterns**: Comprehensive confluence analysis requiring 50+ points
- **Dynamic Risk Management**: Smart position sizing based on multiple factors
- **Market Context Analysis**: Session timing and volatility considerations

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- MetaTrader 5 terminal
- Windows OS (for MT5 integration)

### Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd price_action_bot-main
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your MT5 credentials and settings
   ```

5. **Initialize database**
   ```bash
   python setup_admin.py
   ```

6. **Setup AI (Optional)**
   ```bash
   python setup_ai.py
   ```

## ⚙️ Configuration

### Environment Variables (.env)
```env
# MT5 Connection
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server

# Trading Settings
SYMBOL=EURUSD
SYMBOLS=EURUSD,GBPUSD,USDJPY,AUDUSD,XAUUSD
TIMEFRAME=M15
LOT_SIZE=0.01

# Risk Management
RISK_PERCENTAGE=1.5
MAX_OPEN_POSITIONS=3
DYNAMIC_POSITION_SIZING=True
VOLATILITY_ADJUSTMENT=True

# Strategy Settings
TREND_FOLLOWING_ENABLED=True
SUPPORT_RESISTANCE_ENABLED=True
BREAKOUT_ENABLED=True
REVERSAL_PATTERNS_ENABLED=True

# Telegram Bot
TELEGRAM_BOT_TOKEN=your_bot_token
ADMIN_CHAT_ID=your_chat_id
```

### Key Configuration Parameters

#### Risk Management
- `RISK_PERCENTAGE`: Risk per trade as % of account (default: 1.5%)
- `MAX_OPEN_POSITIONS`: Maximum concurrent positions (default: 3)
- `DYNAMIC_POSITION_SIZING`: Enable smart position sizing (default: True)

#### Strategy Parameters
- `BREAKOUT_CONFIRMATION_BARS`: Bars required for breakout confirmation (default: 3)
- `RSI_PERIOD`: RSI calculation period (default: 14)
- `ATR_MULTIPLIER`: ATR multiplier for stop losses (default: 2.0)

## 🚀 Usage

### Start the Bot
```bash
python main.py
```

### Telegram Commands

#### Basic Commands
- `/start` - Initialize trading session
- `/status` - Check bot status and performance
- `/positions` - View current positions
- `/history` - View trade history
- `/stop` - Stop trading
- `/help` - Show all commands

#### AI Commands
- `/ai_status` - Show AI strategy status
- `/ai_train [periods]` - Train AI models (default: 2000 periods)
- `/ai_retrain` - Retrain existing models
- `/ai_performance` - Show performance report
- `/ai_analyze` - Perform AI analysis on current market
- `/ai_models` - List available models and their performance
- `/ai_config` - Show AI configuration
- `/ai_reset` - Reset AI models (with confirmation)
- `/ai_auto_train` - Toggle auto-training on/off

#### Database Management (Admin)
- `/add_user <telegram_chat_id>` - Add a new authorized user
- `/list_users` - List all authorized users and their MT account status
- `/db_stats` - Show database statistics

## 📈 Trading Strategies

### 1. Breakout Strategy
**Purpose**: Trade breakouts from consolidation patterns

**Entry Criteria**:
- Price breaks above resistance or below support
- 70% confirmation score required
- Volume confirmation (>1.5x average)
- Strong momentum (>0.3 ATR)
- Active trading session

**Exit Criteria**:
- Trailing stop after 0.5% profit
- False breakout detection (3 consecutive closes)
- Target reached (1.5% profit)

### 2. Reversal Patterns Strategy
**Purpose**: Trade candlestick reversal patterns

**Entry Criteria**:
- Strong reversal pattern (hammer, shooting star, engulfing)
- Pattern strength >60/100
- Confluence score >50/100
- RSI alignment (oversold/overbought)
- Near support/resistance levels

**Exit Criteria**:
- Opposite reversal pattern appears
- RSI reaches extreme levels
- Target reached (1.5% profit)

### 3. AI Strategy
**Purpose**: Machine learning-powered trading

**Features**:
- Ensemble model with 6+ ML algorithms
- Real-time market data analysis
- Confidence-based position sizing
- Automatic model retraining
- 100+ engineered features

## 🤖 AI/ML System

### AI Flow Overview
```
📊 Market Data → 🔧 100+ Features → 🤖 6 ML Models → 📈 Trading Signal
```

### Machine Learning Models
1. **Random Forest**: 100 trees, robust predictions
2. **XGBoost**: 50 estimators, fast and efficient
3. **LightGBM**: 50 estimators, memory efficient
4. **SVM**: RBF kernel, high-dimensional data
5. **Neural Network**: (50,25) layers, pattern recognition
6. **LSTM**: Long Short-Term Memory for time series
7. **Ensemble**: Voting classifier combining all models

### Feature Engineering (100+ Features)
- **Price Features (15)**: Ratios, momentum, body size
- **Technical Indicators (45)**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic
- **Price Action (12)**: Candlestick patterns, pattern strength
- **Support/Resistance (8)**: Level strength, distance, breakouts
- **Trend Analysis (6)**: Direction, higher highs/lower lows
- **Volatility (5)**: Rolling volatility, regime classification
- **Time Features (3)**: Hour, day, market sessions

### Auto-Training System
- **Time-based**: Retrains every 24 hours
- **Performance-based**: Retrains when accuracy drops below threshold
- **Data-driven**: Uses historical market data for training
- **Multi-symbol**: Trains on all configured trading symbols

## 🗄️ Database System

### Database Schema

#### bot_user Table
Stores authorized Telegram users who can access the bot.

| Column | Type | Description |
|--------|------|-------------|
| bot_user_id | INTEGER PRIMARY KEY | Auto-incrementing unique identifier |
| telegram_chat_id | INTEGER UNIQUE | Telegram chat ID (must be unique) |
| created_at | TIMESTAMP | When the user was added |
| updated_at | TIMESTAMP | Last update time |

#### mt_account Table
Stores MT5 account associations for bot users.

| Column | Type | Description |
|--------|------|-------------|
| mt_account_id | INTEGER PRIMARY KEY | Auto-incrementing unique identifier |
| bot_user_id | INTEGER | Foreign key to bot_user table |
| mt_account_number | INTEGER | MT5 account number |
| created_at | TIMESTAMP | When the account was associated |
| updated_at | TIMESTAMP | Last update time |

### Key Features
- **User Authorization**: Only authorized Telegram users can access the bot
- **Account Management**: One-to-one relationship between Telegram users and MT5 accounts
- **Session Tracking**: Track which users are logged into which MT5 accounts
- **Admin Tools**: Commands to manage users and view statistics

### Management Commands
```bash
# Add a user
python manage_users.py add 123456789

# List all users
python manage_users.py list

# Show database statistics
python manage_users.py stats

# List MT accounts
python manage_users.py mt_accounts

# Remove a user
python manage_users.py remove 123456789
```

## 📊 Performance Monitoring

### Log Files
- `logs/trade_log.csv` - All trade entries and exits
- `logs/close_log.csv` - Position closures
- `logs/loss_log.csv` - Loss tracking
- `logs/trading_bot.log` - Detailed system logs

### Key Metrics
- **Win Rate**: Percentage of profitable trades
- **Risk-Reward Ratio**: Average profit vs loss
- **Confluence Score**: Signal quality assessment
- **Position Sizing**: Dynamic sizing effectiveness
- **AI Accuracy**: Model prediction accuracy
- **Training Performance**: Model training metrics

## 🔧 Advanced Features

### Dynamic Position Sizing
Position size is calculated based on:
- Signal confidence (0.5x to 1.5x multiplier)
- Market volatility (0.3x to 1.2x multiplier)
- Confluence score (0.7x to 1.3x multiplier)

### Market Context Analysis
- **Volatility Assessment**: Current vs average ATR
- **Trend Strength**: Price vs moving averages
- **Session Detection**: London, New York, Asian, Overlap
- **Support/Resistance**: Key level identification

### Risk Management
- **Stop Losses**: ATR-based with dynamic multipliers
- **Take Profits**: Risk-reward ratios (1:2.5)
- **Trailing Stops**: Automatic profit protection
- **Position Limits**: Maximum concurrent positions

## 🐛 Troubleshooting

### Common Issues

1. **MT5 Connection Failed**
   - Check login credentials in .env
   - Ensure MT5 terminal is running
   - Verify server name is correct

2. **No Trading Signals**
   - Check strategy enable flags in config
   - Verify market hours and session settings
   - Review confirmation thresholds

3. **High Losses**
   - Reduce RISK_PERCENTAGE
   - Enable VOLATILITY_ADJUSTMENT
   - Check MAX_OPEN_POSITIONS

4. **AI Models Not Training**
   - Check data availability
   - Verify feature engineering
   - Ensure sufficient data points

### Debug Mode
```bash
python main.py --debug
```

## 📚 File Structure

```
price_action_bot-main/
├── main.py                 # Main bot application
├── config.py              # Configuration settings
├── utils.py               # Utility functions and indicators
├── mt5_connector.py       # MetaTrader 5 integration
├── telegram_bot.py        # Telegram interface
├── database.py            # Database operations
├── manage_users.py        # User management script
├── setup_admin.py         # Admin setup script
├── setup_ai.py           # AI setup script
├── strategies/            # Trading strategies
│   ├── breakout.py
│   ├── reversal_patterns.py
│   ├── support_resistance.py
│   └── trend_following.py
├── ai/                    # AI strategy components
│   ├── ai_strategy.py
│   ├── model_manager.py
│   ├── data_processor.py
│   ├── auto_trainer.py
│   └── ai_telegram_bot.py
├── logs/                  # Log files
├── trading_bot.db         # SQLite database
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎯 Expected Performance

### Training Time
- **Initial Training**: 2-5 minutes
- **Retraining**: 1-3 minutes
- **Prediction**: <1 second

### Accuracy Expectations
- **Overall Accuracy**: 60-75%
- **High Confidence Trades**: 70-85%
- **Win Rate**: 55-70%
- **Risk/Reward**: 1:2 average

### Resource Usage
- **CPU**: Moderate during training
- **Memory**: ~500MB for models
- **Storage**: ~50MB for model files

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Always test thoroughly in demo mode before live trading.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the logs for error details

---

**Version**: 2.0  
**Last Updated**: 2025  
**Author**: Price Action Trading Bot Team
