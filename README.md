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
- **Multi-Terminal Support**: Dedicated terminals for each user with complete isolation

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
| is_admin | BOOLEAN | Whether user has admin privileges |
| created_at | TIMESTAMP | When the user was added |
| updated_at | TIMESTAMP | Last update time |

#### mt_account Table
Stores MT5 account associations for bot users.

| Column | Type | Description |
|--------|------|-------------|
| mt_account_id | INTEGER PRIMARY KEY | Auto-incrementing unique identifier |
| bot_user_id | INTEGER | Foreign key to bot_user table |
| mt_account_number | INTEGER | MT5 account number |
| terminal_name | TEXT | Terminal name (format: tmn_{telegram_chat_id}) |
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

## 🖥️ Multi-Terminal System

### Overview
The multi-terminal system allows you to:
- Run separate MT5 terminal instances for different accounts
- Isolate trading operations between accounts
- Manage multiple brokers simultaneously
- Have dedicated terminals for demo vs live accounts

### Quick Start

#### 1. Setup Multi-Terminal Configuration
Run the setup script to configure your terminals:

```bash
python setup_multi_terminal.py
```

This will:
- Find all available MT5 installations
- Guide you through configuring each terminal
- Create a `terminals_config.json` file
- Test the configuration

#### 2. Example Configuration
Your `terminals_config.json` will look like this:

```json
{
  "terminals": [
    {
      "name": "demo_account_1",
      "terminal_path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
      "login": 12345,
      "password": "demo_password_1",
      "server": "DemoServer",
      "symbol": "EURUSD",
      "timeframe": "M15",
      "auto_start": true,
      "port_offset": 0
    },
    {
      "name": "live_account_1",
      "terminal_path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
      "login": 67890,
      "password": "live_password_1",
      "server": "LiveServer",
      "symbol": "GBPUSD",
      "timeframe": "M15",
      "auto_start": true,
      "port_offset": 0
    }
  ]
}
```

### Usage Methods

#### Method 1: Programmatic Usage
```python
from terminal_manager import TerminalManager, TerminalConfig
from mt5_connector import MT5Connector

# Initialize terminal manager
manager = TerminalManager()

# Load configuration
manager.create_terminal_configs_from_file("terminals_config.json")

# Start all terminals
manager.start_all_terminals()

# Create connectors for each account
demo_connector = MT5Connector(
    login=12345,
    password="demo_password_1",
    server="DemoServer",
    terminal_name="demo_account_1",
    dedicated_terminal=True
)

live_connector = MT5Connector(
    login=67890,
    password="live_password_1",
    server="LiveServer",
    terminal_name="live_account_1",
    dedicated_terminal=True
)

# Connect to accounts
demo_connector.connect()
live_connector.connect()

# Now you can trade on both accounts simultaneously
```

#### Method 2: Telegram Bot Usage

##### Login with Dedicated Terminal
```
/login 12345 demo_password_1 DemoServer demo_account_1
```

##### Login with Shared Terminal (default behavior)
```
/login 12345 demo_password_1 DemoServer
```

##### Terminal Management Commands (Admin Only)
```
/terminals                    # Show all terminal status
/terminal_start demo_account_1 # Start specific terminal
/terminal_stop demo_account_1  # Stop specific terminal
/terminal_restart demo_account_1 # Restart specific terminal
/sessions                     # Show active sessions with terminal info
```

### Terminal Management

#### TerminalManager Class
The `TerminalManager` class handles all terminal operations:

```python
from terminal_manager import terminal_manager

# Get terminal status
status = terminal_manager.get_terminal_status()

# Start specific terminal
terminal_manager.start_terminal("demo_account_1")

# Stop specific terminal
terminal_manager.stop_terminal("demo_account_1")

# Restart terminal
terminal_manager.restart_terminal("demo_account_1")

# Start monitoring (auto-restart failed terminals)
terminal_manager.start_monitoring()
```

#### Terminal Status Types
- 🟢 **running**: Terminal is active and connected
- 🔴 **stopped**: Terminal is not running
- 🟡 **starting**: Terminal is in the process of starting
- ❌ **failed**: Terminal failed to start
- 💥 **crashed**: Terminal crashed and needs restart
- ⚪ **configured**: Terminal is configured but not started

### MT5Connector Updates
The `MT5Connector` class now supports dedicated terminals:

```python
# Shared terminal (default)
connector = MT5Connector(
    login=12345,
    password="password",
    server="server"
)

# Dedicated terminal
connector = MT5Connector(
    login=12345,
    password="password",
    server="server",
    terminal_name="demo_account_1",
    dedicated_terminal=True
)

# Get terminal information
terminal_info = connector.get_terminal_info()
print(terminal_info)
# Output: {'type': 'dedicated', 'terminal_name': 'demo_account_1', 'status': {...}}
```

### Configuration Options

#### Terminal Configuration Fields
- **name**: Unique identifier for the terminal
- **terminal_path**: Path to MT5 executable
- **login**: Account login number
- **password**: Account password
- **server**: Broker server name
- **symbol**: Default trading symbol
- **timeframe**: Default timeframe
- **auto_start**: Whether to start automatically
- **port_offset**: Port offset for data connections (if needed)

#### Settings
- **monitoring_enabled**: Enable background monitoring
- **auto_restart_failed_terminals**: Auto-restart crashed terminals
- **max_restart_attempts**: Maximum restart attempts before giving up
- **restart_delay_seconds**: Delay between restart attempts
- **health_check_interval_seconds**: How often to check terminal health

### Monitoring and Troubleshooting

#### Health Monitoring
The system automatically monitors terminal health:

```python
# Start monitoring
terminal_manager.start_monitoring()

# Check terminal health
status = terminal_manager.get_terminal_status("demo_account_1")
if status['status']['status'] == 'crashed':
    print("Terminal crashed, attempting restart...")
    terminal_manager.restart_terminal("demo_account_1")
```

#### Common Issues
1. **Terminal won't start**
   - Check if MT5 is installed at the specified path
   - Verify account credentials
   - Ensure no other MT5 instances are blocking

2. **Connection failures**
   - Check internet connection
   - Verify server name is correct
   - Ensure account is not already logged in elsewhere

3. **Multiple terminals conflict**
   - Each terminal should use the same MT5 installation
   - Different accounts can use the same terminal path
   - The system handles process isolation automatically

### Best Practices
1. **Terminal Naming**: Use descriptive names for terminals
2. **Auto-Start Configuration**: Enable auto-start for frequently used accounts
3. **Monitoring**: Always enable monitoring for production use
4. **Security**: Store passwords securely and use environment variables

## 👥 User Management

### Admin Features

#### Add User Button
The "➕ Add User" button provides a simplified user creation process:

1. **Click "➕ Add User" button**
2. **Enter Telegram Chat ID** (e.g., `123456789`)
3. **Bot automatically:**
   - Creates user in database
   - Generates terminal name: `tmn_123456789`
   - Creates terminal configuration
   - Sets up dedicated terminal
4. **User can now login** and provide their MT5 credentials

#### Delete User Button
The "🗑️ Delete User" button provides a user-friendly way to delete specific users:

1. **Click "🗑️ Delete User" button**
2. **Bot shows available users for deletion**
3. **Enter Telegram Chat ID** when prompted
4. **Bot shows user details** and asks for confirmation
5. **Type exact confirmation** text
6. **User deleted** safely and completely

#### Terminal Display
The "🖥️ Terminals" button shows comprehensive terminal and user information:

```
🖥️ Terminal Management (Admin Only)

👥 Users and Terminals:

👑 1662162192 (Admin)
   MT Account: 11045991 (🟢 Active)  ← User currently logged in
   Terminal: tmn_1662162192 🟢

👤 123456789 (User)
   MT Account: 11012345              ← Stored in database
   Terminal: tmn_123456789 🟢

👤 987654321 (User)
   MT Account: 11098765 (Terminal)   ← From terminal config
   Terminal: tmn_987654321 🟢

👤 555666777 (User)
   MT Account: None                  ← No login info
   Terminal: tmn_555666777 🔴
```

### User Management Commands

#### Command Line Scripts
```bash
# Create terminal for existing user
python create_terminal_for_existing_user.py 123456789 11096557

# Delete specific user
python delete_specific_user.py 987654321

# List users
python manage_users.py list
```

#### Telegram Bot Commands (Admin Only)
```
/add_user <telegram_chat_id>        # Add new user
/delete_user <telegram_chat_id>     # Delete specific user
/create_terminal <chat_id> <account> # Create terminal for existing user
/list_users                         # List all users
/db_stats                          # Database statistics
```

### Terminal Naming Convention
- **Format**: `tmn_` + telegram_chat_id
- **Examples**: 
  - `tmn_123456789`
  - `tmn_001`
  - `tmn_1662162192`

### Status Indicators

#### MT Account Status
- **🟢 Active** - User is currently logged in and active
- **(no indicator)** - MT account stored in database
- **(Terminal)** - Login info from terminal configuration
- **None** - No MT account information available

#### Terminal Status
- **🟢** - Terminal running
- **🔴** - Terminal stopped
- **🟡** - Terminal starting
- **❌** - Terminal failed
- **💥** - Terminal crashed
- **⚪** - Terminal configured
- **❓** - Terminal status unknown

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

5. **Terminal Issues**
   - Check if MT5 is installed at the specified path
   - Verify account credentials
   - Ensure no other MT5 instances are blocking

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
├── terminal_manager.py    # Multi-terminal management
├── auto_terminal_manager.py # Automatic terminal management
├── create_terminal_for_existing_user.py # Terminal creation script
├── delete_specific_user.py # User deletion script
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
├── terminals_config.json  # Terminal configuration
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