"""
MetaTrader 5 Connector Module
Handles all MT5 API interactions including connection, data retrieval, and order management
"""

import MetaTrader5 as mt5
import numpy as np
import os
import time
import subprocess
try:
    import psutil  # optional
    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    _PSUTIL_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError as e:
    import logging
    logging.warning(f"Pandas not available: {e}")
    PANDAS_AVAILABLE = False
    pd = None
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging
from config import *

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# Global registry to track all MT5Connector instances
_active_connectors = []

class MT5Troubleshooter:
    """MT5 connection troubleshooting utilities"""
    
    @staticmethod
    def find_mt5_installations() -> List[str]:
        """Find all possible MT5 installation paths"""
        possible_paths = [
            r"C:\Program Files\MetaTrader 5\terminal64.exe",
            r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
            r"C:\Program Files\MetaTrader 5\terminal.exe",
            r"C:\Program Files (x86)\MetaTrader 5\terminal.exe"
        ]
        
        # Add user-specific paths
        username = os.getenv('USERNAME', '')
        if username:
            user_paths = [
                rf"C:\Users\{username}\AppData\Roaming\MetaQuotes\Terminal\*\terminal64.exe",
                rf"C:\Users\{username}\AppData\Roaming\MetaQuotes\Terminal\*\terminal.exe"
            ]
            possible_paths.extend(user_paths)
        
        valid_paths = []
        for path in possible_paths:
            if '*' in path:
                # Handle wildcard paths
                import glob
                matches = glob.glob(path)
                valid_paths.extend(matches)
            elif os.path.exists(path):
                valid_paths.append(path)
        
        return valid_paths
    
    @staticmethod
    def is_mt5_running() -> bool:
        """Check if MT5 terminal is currently running"""
        if not _PSUTIL_AVAILABLE:
            logger.warning("psutil not available; skipping MT5 running check")
            return False
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] and 'terminal' in proc.info['name'].lower():
                    return True
            return False
        except Exception:
            return False
    
    @staticmethod
    def kill_mt5_processes():
        """Kill all running MT5 processes"""
        if not _PSUTIL_AVAILABLE:
            logger.warning("psutil not available; cannot kill MT5 processes")
            return False
        try:
            killed_count = 0
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] and 'terminal' in proc.info['name'].lower():
                    try:
                        proc.kill()
                        killed_count += 1
                        logger.info(f"Killed MT5 process: {proc.info['name']} (PID: {proc.info['pid']})")
                    except Exception as e:
                        logger.warning(f"Could not kill MT5 process {proc.info['pid']}: {e}")
            if killed_count > 0:
                logger.info(f"Killed {killed_count} MT5 processes")
                time.sleep(2)  # Wait for processes to fully terminate
            return killed_count > 0
        except Exception as e:
            logger.error(f"Error killing MT5 processes: {e}")
            return False
    
    @staticmethod
    def start_mt5_terminal(terminal_path: str) -> bool:
        """Start MT5 terminal if not running"""
        try:
            if MT5Troubleshooter.is_mt5_running():
                logger.info("MT5 terminal is already running")
                return True
            
            if not os.path.exists(terminal_path):
                logger.error(f"MT5 terminal not found at: {terminal_path}")
                return False
            
            logger.info(f"Starting MT5 terminal: {terminal_path}")
            subprocess.Popen([terminal_path], shell=True)
            time.sleep(5)  # Wait for terminal to start
            return True
        except Exception as e:
            logger.error(f"Error starting MT5 terminal: {e}")
            return False
    
    @staticmethod
    def diagnose_connection_issue() -> Dict:
        """Diagnose common MT5 connection issues"""
        diagnosis = {
            'mt5_installed': False,
            'mt5_running': False,
            'valid_paths': [],
            'recommendations': []
        }
        
        # Check for MT5 installations
        valid_paths = MT5Troubleshooter.find_mt5_installations()
        diagnosis['valid_paths'] = valid_paths
        diagnosis['mt5_installed'] = len(valid_paths) > 0
        
        # Check if MT5 is running
        diagnosis['mt5_running'] = MT5Troubleshooter.is_mt5_running()
        
        # Generate recommendations
        if not diagnosis['mt5_installed']:
            diagnosis['recommendations'].append("Install MetaTrader 5 terminal")
        elif not diagnosis['mt5_running']:
            diagnosis['recommendations'].append("Start MetaTrader 5 terminal manually")
            if valid_paths:
                diagnosis['recommendations'].append(f"Try starting: {valid_paths[0]}")
        else:
            diagnosis['recommendations'].append("MT5 is running - check credentials and server")
            diagnosis['recommendations'].append("Ensure MT5 is logged in to your account")
            diagnosis['recommendations'].append("Check if MT5 is in demo/live mode as expected")
        
        return diagnosis

class MT5Connector:
    """MetaTrader 5 connection and trading operations handler.

    Supports per-instance credentials and symbol/timeframe overrides for multi-account usage.
    Each instance maintains its own connection state and credentials.
    """
    
    def __init__(self, login: int = None, password: str = None, server: str = None,
                 terminal_path: str = None, symbol: str = None, timeframe: str = None):
        self.connected = False
        self.account_info = None
        self.symbol_info = None
        # Per-instance settings with fallbacks to global config
        self._login = login if login is not None else MT5_LOGIN
        self._password = password if password is not None else MT5_PASSWORD
        self._server = server if server is not None else MT5_SERVER
        self._terminal_path = terminal_path if terminal_path is not None else MT5_PATH
        self._symbol = symbol if symbol is not None else SYMBOL
        self._timeframe = timeframe if timeframe is not None else TIMEFRAME
        # Track if this instance is currently active (connected to MT5)
        self._is_active = False
        # Store last MT5 error (code, description)
        self._last_error = None
        # Register this instance globally
        _active_connectors.append(self)
        
    def connect(self, max_retries: int = 3, retry_delay: int = 5) -> bool:
        """
        Establish connection to MetaTrader 5 with retry logic and troubleshooting
        
        Args:
            max_retries: Maximum number of connection attempts
            retry_delay: Delay between retry attempts in seconds
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"MT5 connection attempt {attempt + 1}/{max_retries}")
                
                # Check if MT5 terminal is running
                if not MT5Troubleshooter.is_mt5_running():
                    logger.warning("MT5 terminal is not running, attempting to start...")
                    if not MT5Troubleshooter.start_mt5_terminal(self._terminal_path):
                        logger.error("Failed to start MT5 terminal")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        else:
                            # Final attempt - try to diagnose the issue
                            diagnosis = MT5Troubleshooter.diagnose_connection_issue()
                            logger.error("MT5 Connection Diagnosis:")
                            logger.error(f"  MT5 Installed: {diagnosis['mt5_installed']}")
                            logger.error(f"  MT5 Running: {diagnosis['mt5_running']}")
                            logger.error(f"  Valid Paths: {diagnosis['valid_paths']}")
                            logger.error("  Recommendations:")
                            for rec in diagnosis['recommendations']:
                                logger.error(f"    - {rec}")
                            return False
                
                # Try to initialize MT5
                if not mt5.initialize(path=self._terminal_path):
                    error_code, error_desc = mt5.last_error()
                    self._last_error = (error_code, error_desc)
                    logger.error(f"MT5 initialization failed (attempt {attempt + 1}): {error_code} - {error_desc}")
                    
                    # Handle specific error codes
                    if error_code == -10005:  # IPC timeout
                        logger.warning("IPC timeout detected - this usually means MT5 is not responding")
                        if attempt < max_retries - 1:
                            logger.info(f"Waiting {retry_delay} seconds before retry...")
                            time.sleep(retry_delay)
                            continue
                    elif error_code == -10004:  # Common initialization error
                        logger.warning("MT5 initialization error - trying to restart terminal")
                        MT5Troubleshooter.kill_mt5_processes()
                        time.sleep(3)
                        if not MT5Troubleshooter.start_mt5_terminal(self._terminal_path):
                            logger.error("Failed to restart MT5 terminal")
                            if attempt < max_retries - 1:
                                time.sleep(retry_delay)
                                continue
                    
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        return False
                
                # Check if already connected to the same account
                current_account = mt5.account_info()
                if current_account and current_account.login == self._login:
                    logger.info(f"Already connected to account {self._login}")
                    self.connected = True
                    self._is_active = True
                    self.account_info = current_account
                    
                    # Mark all other connectors as inactive
                    for connector in _active_connectors:
                        if connector != self:
                            connector._is_active = False
                            connector.connected = False
                    
                    return True
                
                # Login to account (this will disconnect any existing connection)
                if not mt5.login(login=self._login, password=self._password, server=self._server):
                    self._last_error = mt5.last_error()
                    logger.error(f"MT5 login failed: {self._last_error}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        return False
                
                # Get account info
                self.account_info = mt5.account_info()
                if self.account_info is None:
                    self._last_error = mt5.last_error()
                    logger.error("Failed to get account info")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        return False
                
                # Get symbol info
                self.symbol_info = mt5.symbol_info(self._symbol)
                if self.symbol_info is None:
                    self._last_error = mt5.last_error()
                    logger.error(f"Failed to get symbol info for {self._symbol}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        return False
                
                # Enable symbol for trading
                if not mt5.symbol_select(self._symbol, True):
                    self._last_error = mt5.last_error()
                    logger.error(f"Failed to select symbol {self._symbol}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        return False
                
                self.connected = True
                self._is_active = True
                
                # Mark all other connectors as inactive
                for connector in _active_connectors:
                    if connector != self:
                        connector._is_active = False
                        connector.connected = False
                
                logger.info(f"Successfully connected to MT5. Account: {self.account_info.login}")
                logger.info(f"Balance: {self.account_info.balance}, Equity: {self.account_info.equity}")
                
                return True
                
            except Exception as e:
                logger.error(f"Connection error (attempt {attempt + 1}): {e}")
                try:
                    self._last_error = mt5.last_error()
                except Exception:
                    pass
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    return False
        
        return False
    
    def disconnect(self):
        """Disconnect from MetaTrader 5"""
        if self._is_active:
            # Only shutdown if this instance is currently active
            mt5.shutdown()
            self._is_active = False
            logger.info(f"Disconnected from MT5 account {self._login}")
        self.connected = False
    
    def __del__(self):
        """Cleanup when connector is destroyed"""
        try:
            if self in _active_connectors:
                _active_connectors.remove(self)
        except Exception:
            pass  # Ignore errors during cleanup
    
    def ensure_connection(self) -> bool:
        """
        Ensure this instance is connected to MT5.
        If another instance is connected, switch to this one.
        
        Returns:
            bool: True if connected, False otherwise
        """
        if self.connected and self._is_active:
            # Double-check that we're still connected to the right account
            current_account = mt5.account_info()
            if current_account and current_account.login == self._login:
                return True
            else:
                # Connection was hijacked by another instance
                self._is_active = False
                self.connected = False
        
        # Check if MT5 is connected to a different account
        current_account = mt5.account_info()
        if current_account and current_account.login != self._login:
            logger.info(f"Switching from account {current_account.login} to {self._login}")
        
        return self.connect()
    
    def is_active_connection(self) -> bool:
        """
        Check if this instance is currently the active MT5 connection
        
        Returns:
            bool: True if this instance is active, False otherwise
        """
        if not self.connected or not self._is_active:
            return False
        
        current_account = mt5.account_info()
        return current_account and current_account.login == self._login
    
    def get_connection_status(self) -> Dict:
        """
        Get detailed connection status information
        
        Returns:
            Dict: Connection status details
        """
        current_account = mt5.account_info()
        is_active = self.is_active_connection()
        
        return {
            'login': self._login,
            'connected': self.connected,
            'is_active': is_active,
            'current_mt5_account': current_account.login if current_account else None,
            'account_matches': current_account and current_account.login == self._login
        }

    def _sanitize_comment(self, text: str) -> str:
        """Sanitize order comment to ASCII and MT5 length limits (~27-31 chars)."""
        try:
            ascii_text = (text or "").encode("ascii", "ignore").decode("ascii")
        except Exception:
            ascii_text = str(text or "")
        # Remove newlines and trim spaces
        ascii_text = ascii_text.replace("\n", " ").replace("\r", " ").strip()
        # MT5 typically allows up to 27 chars shown in terminal; keep conservative
        return ascii_text[:27]
    
    def get_account_info(self) -> Optional[Dict]:
        """
        Get current account information
        
        Returns:
            Dict: Account information or None if failed
        """
        if not self.ensure_connection():
            return None
        
        account_info = mt5.account_info()
        if account_info is None:
            return None
        
        # Verify that we're connected to the correct account
        if account_info.login != self._login:
            logger.warning(f"Account mismatch: expected {self._login}, got {account_info.login}")
            return None
        
        # Extract optional fields defensively (server/company may not exist in some APIs)
        server = getattr(account_info, 'server', None)
        company = getattr(account_info, 'company', None)
        name = getattr(account_info, 'name', None)
        trade_mode = getattr(account_info, 'trade_mode', None)
        return {
            'login': account_info.login,
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'margin_level': account_info.margin_level,
            'currency': account_info.currency,
            'leverage': account_info.leverage,
            'server': server,
            'company': company,
            'name': name,
            'trade_mode': trade_mode,
        }

    def change_symbol(self, symbol: str) -> bool:
        """Switch working symbol and select it in terminal."""
        try:
            if not symbol:
                return False
            self._symbol = symbol
            # ensure selected
            ok = mt5.symbol_select(symbol, True)
            try:
                # refresh cached symbol_info for downstream consumers
                self.symbol_info = mt5.symbol_info(symbol)
            except Exception:
                self.symbol_info = None
            return ok
        except Exception:
            return False
    
    def get_symbol_info(self, symbol: str = None) -> Optional[Dict]:
        """
        Get symbol information
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict: Symbol information or None if failed
        """
        symbol_info = mt5.symbol_info(symbol or self._symbol)
        if symbol_info is None:
            return None
        
        return {
            'name': symbol_info.name,
            'point': symbol_info.point,
            'digits': symbol_info.digits,
            'spread': symbol_info.spread,
            'trade_contract_size': symbol_info.trade_contract_size,
            'trade_tick_size': symbol_info.trade_tick_size,
            'trade_tick_value': symbol_info.trade_tick_value,
            'trade_mode': symbol_info.trade_mode,
            'trade_stops_level': symbol_info.trade_stops_level,
            'trade_freeze_level': symbol_info.trade_freeze_level
        }

    def get_symbol(self) -> str:
        """Return the current working symbol for this connector."""
        return self._symbol
    
    def has_symbol(self, symbol: str) -> bool:
        """Check if a symbol exists and is available to trade/select."""
        try:
            info = mt5.symbol_info(symbol)
            if info:
                # Try selecting to confirm availability
                return mt5.symbol_select(symbol, True)
            return False
        except Exception:
            return False

    def detect_symbol_variant(self, base_symbols: list) -> list:
        """Choose appropriate symbol variant based on availability (e.g., with '+' suffix).

        Given base symbols like ['EURUSD','GBPUSD'], returns either ['EURUSD+','GBPUSD+']
        if that variant exists, otherwise returns the base list.
        """
        try:
            if not base_symbols:
                return base_symbols
            plus_candidate = f"{base_symbols[0]}+"
            if self.has_symbol(plus_candidate):
                return [f"{s}+" for s in base_symbols]
            # Fallback to base if base exists
            if self.has_symbol(base_symbols[0]):
                return list(base_symbols)
            # As a last resort, return base list (controller may handle errors)
            return list(base_symbols)
        except Exception:
            return list(base_symbols)

    def get_last_error_message(self) -> str:
        """Return a formatted last MT5 error message for user display."""
        try:
            if not self._last_error:
                return "Unknown error"
            code, desc = self._last_error
            hint = ""
            # Common MT5 error hints
            if code in (-10013,):
                hint = " (check account number/password/server)"
            elif code in (-10004, -10005):
                hint = " (terminal not responding; try again)"
            # Return user-friendly message without error code prefix
            return f"{desc}{hint}"
        except Exception:
            return "Unknown error"
    
    def get_rates(self, symbol: str = None, timeframe: str = None, count: int = 1000):
        """
        Get historical price data
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M1, M5, M15, etc.)
            count: Number of bars to retrieve
            
        Returns:
            DataFrame: OHLCV data or None if failed
        """
        if not self.ensure_connection():
            logger.warning("MT5 not connected, cannot get rates")
            return None
        
        try:
            # Convert timeframe string to MT5 constant
            use_tf = timeframe or self._timeframe
            tf = TIMEFRAME_MAPPING.get(use_tf, 15)
            
            if use_tf not in TIMEFRAME_MAPPING:
                logger.error(f"Unsupported timeframe: {use_tf}. Supported: {list(TIMEFRAME_MAPPING.keys())}")
                return None
            
            # Get rates
            use_symbol = symbol or self._symbol
            rates = mt5.copy_rates_from_pos(use_symbol, tf, 0, count)
            if rates is None or len(rates) == 0:
                logger.error(f"Failed to get rates for {use_symbol} on {use_tf} (MT5 constant: {tf}). MT5 error: {mt5.last_error()}")
                return None
            
            # Convert to DataFrame
            if PANDAS_AVAILABLE:
                df = pd.DataFrame(rates)
                if df.empty:
                    logger.error(f"Empty DataFrame received for {use_symbol} on {use_tf}")
                    return None
                    
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                
                # Basic validation
                required_cols = ['open', 'high', 'low', 'close']
                if not all(col in df.columns for col in required_cols):
                    logger.error(f"Missing required columns in rates data: {df.columns.tolist()}")
                    return None
                
                return df
            else:
                # Return raw data if pandas not available
                return rates
            
        except Exception as e:
            logger.error(f"Error getting rates for {use_symbol} on {use_tf}: {e}")
            return None
    
    def get_current_price(self, symbol: str = None) -> Optional[Dict]:
        """
        Get current bid/ask prices
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict: Current prices or None if failed
        """
        if not self.ensure_connection():
            return None
        
        try:
            use_symbol = symbol or self._symbol
            tick = mt5.symbol_info_tick(use_symbol)
            if tick is None:
                return None
            
            return {
                'symbol': use_symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'time': datetime.fromtimestamp(tick.time)
            }
            
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None
    
    def calculate_lot_size(self, risk_amount: float, stop_loss_points: int) -> float:
        """
        Calculate lot size based on risk amount and stop loss
        
        Args:
            risk_amount: Risk amount in account currency
            stop_loss_points: Stop loss in points
            
        Returns:
            float: Calculated lot size
        """
        if not self.symbol_info or stop_loss_points <= 0:
            return LOT_SIZE
        
        try:
            # Calculate lot size based on risk
            tick_value = self.symbol_info.trade_tick_value
            tick_size = self.symbol_info.trade_tick_size
            point = self.symbol_info.point
            
            # Convert points to price difference
            price_diff = stop_loss_points * point
            
            # Calculate lot size
            lot_size = risk_amount / (price_diff / tick_size * tick_value)
            
            # Round to valid lot size and clamp to limits
            lot_step = max(self.symbol_info.volume_step, 1e-2)
            lot_min = max(self.symbol_info.volume_min, lot_step)
            lot_max = max(self.symbol_info.volume_max, lot_min)
            lot_size = round(lot_size / lot_step) * lot_step
            lot_size = min(max(lot_size, lot_min), lot_max)
            
            return lot_size
            
        except Exception as e:
            logger.error(f"Error calculating lot size: {e}")
            return LOT_SIZE
    
    def place_market_order(self, order_type: str, volume: float, price: float = None, 
                          sl: float = None, tp: float = None, comment: str = "", 
                          magic: int = 0) -> Optional[Dict]:
        """
        Place a market order
        
        Args:
            order_type: 'buy' or 'sell'
            volume: Order volume
            price: Order price (None for market)
            sl: Stop loss price
            tp: Take profit price
            comment: Order comment
            magic: Magic number for strategy identification
            
        Returns:
            Dict: Order result or None if failed
        """
        if not self.ensure_connection():
            return None
        
        try:
            # Get current price if not specified
            if price is None:
                current_price = self.get_current_price()
                if current_price is None:
                    return None
                price = current_price['ask'] if order_type == 'buy' else current_price['bid']
            
            # Enforce stops level and freeze level using fresh symbol info for the current symbol
            sym_info = mt5.symbol_info(self._symbol)
            if sym_info is None:
                logger.error(f"No symbol info for {self._symbol}")
                return None
            stops_level_points = getattr(sym_info, 'trade_stops_level', 0)
            freeze_level_points = getattr(sym_info, 'trade_freeze_level', 0)
            point = getattr(sym_info, 'point', 0.0001)
            # Add small safety buffer of 2 points
            min_stop_dist = (max(stops_level_points, freeze_level_points) + 2) * point

            # Normalize SL/TP against direction and enforce minimum distances
            if order_type.lower() == 'buy':
                if sl is not None:
                    # Ensure SL is below price by at least min_stop_dist
                    if sl >= price - min_stop_dist:
                        sl = price - (min_stop_dist * 1.2)
                if tp is not None:
                    # Ensure TP is above price by at least min_stop_dist
                    if tp <= price + min_stop_dist:
                        tp = price + (min_stop_dist * 1.2)
            else:  # sell
                if sl is not None:
                    # Ensure SL is above price by at least min_stop_dist
                    if sl <= price + min_stop_dist:
                        sl = price + (min_stop_dist * 1.2)
                if tp is not None:
                    # Ensure TP is below price by at least min_stop_dist
                    if tp >= price - min_stop_dist:
                        tp = price - (min_stop_dist * 1.2)

            # Round price and stops to symbol digits
            digits = getattr(sym_info, 'digits', 5)
            def _round(v):
                return None if v is None else round(v, digits)
            price = _round(price)
            sl = _round(sl)
            tp = _round(tp)

            # Prepare order request
            if order_type.lower() == 'buy':
                order_type_mt5 = mt5.ORDER_TYPE_BUY
            elif order_type.lower() == 'sell':
                order_type_mt5 = mt5.ORDER_TYPE_SELL
            else:
                logger.error(f"Invalid order type: {order_type}")
                return None
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self._symbol,
                "volume": volume,
                "type": order_type_mt5,
                "price": price,
                "sl": sl,
                "tp": tp,
                # Be a bit more permissive on slippage
                "deviation": 80,
                "magic": magic,
                "comment": self._sanitize_comment(comment),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            if result is None or getattr(result, 'retcode', None) != mt5.TRADE_RETCODE_DONE:
                # If invalid stops, retry once with SL/TP removed
                ret = getattr(result, 'retcode', None) if result is not None else None
                if ret == mt5.TRADE_RETCODE_INVALID_STOPS or ret == 10016:
                    request['sl'] = None
                    request['tp'] = None
                    retry = mt5.order_send(request)
                    if retry is None or getattr(retry, 'retcode', None) != mt5.TRADE_RETCODE_DONE:
                        logger.error(f"Order failed: {getattr(retry, 'retcode', ret)} - {getattr(retry, 'comment', '')} last_error={mt5.last_error()}")
                        return None
                    result = retry
                else:
                    logger.error(f"Order failed: {ret} - {getattr(result, 'comment', '') if result else ''} last_error={mt5.last_error()}")
                    return None
            
            logger.info(f"Market order placed: {order_type} {volume} {self._symbol} at {price}")
            return {
                'retcode': result.retcode,
                'deal': result.deal,
                'order': result.order,
                'volume': result.volume,
                'price': result.price,
                'comment': result.comment
            }
            
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None
    
    def place_pending_order(self, order_type: str, volume: float, price: float,
                           sl: float = None, tp: float = None, comment: str = "",
                           magic: int = 0) -> Optional[Dict]:
        """
        Place a pending order
        
        Args:
            order_type: 'buy_limit', 'sell_limit', 'buy_stop', 'sell_stop'
            volume: Order volume
            price: Order price
            sl: Stop loss price
            tp: Take profit price
            comment: Order comment
            magic: Magic number for strategy identification
            
        Returns:
            Dict: Order result or None if failed
        """
        if not self.ensure_connection():
            return None
        
        try:
            # Map order types
            order_type_mapping = {
                'buy_limit': mt5.ORDER_TYPE_BUY_LIMIT,
                'sell_limit': mt5.ORDER_TYPE_SELL_LIMIT,
                'buy_stop': mt5.ORDER_TYPE_BUY_STOP,
                'sell_stop': mt5.ORDER_TYPE_SELL_STOP
            }
            
            if order_type not in order_type_mapping:
                logger.error(f"Invalid pending order type: {order_type}")
                return None
            
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": self._symbol,
                "volume": volume,
                "type": order_type_mapping[order_type],
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": magic,
                "comment": self._sanitize_comment(comment),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            if result is None or getattr(result, 'retcode', None) != mt5.TRADE_RETCODE_DONE:
                ret = getattr(result, 'retcode', None) if result is not None else None
                comment_text = getattr(result, 'comment', '') if result is not None else ''
                logger.error(f"Pending order failed: {ret} - {comment_text} last_error={mt5.last_error()}")
                return None

            logger.info(f"Pending order placed: {order_type} {volume} {self._symbol} at {price}")
            return {
                'retcode': result.retcode,
                'deal': result.deal,
                'order': result.order,
                'volume': result.volume,
                'price': result.price,
                'comment': result.comment
            }
            
        except Exception as e:
            logger.error(f"Error placing pending order: {e}")
            return None
    
    def get_positions(self, symbol: str = None) -> List[Dict]:
        """
        Get open positions
        
        Args:
            symbol: Trading symbol (None for all symbols)
            
        Returns:
            List[Dict]: List of open positions
        """
        if not self.ensure_connection():
            return []
        
        try:
            # If symbol is None, fetch all positions; else filter by symbol
            positions = mt5.positions_get() if symbol is None else mt5.positions_get(symbol=symbol)
            if positions is None:
                return []
            
            position_list = []
            for pos in positions:
                position_list.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'buy' if pos.type == 0 else 'sell',
                    'volume': pos.volume,
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'magic': pos.magic,
                    'comment': pos.comment,
                    'time': datetime.fromtimestamp(pos.time)
                })
            
            return position_list
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_orders(self, symbol: str = None) -> List[Dict]:
        """
        Get pending orders
        
        Args:
            symbol: Trading symbol (None for all symbols)
            
        Returns:
            List[Dict]: List of pending orders
        """
        if not self.ensure_connection():
            return []
        
        try:
            # If symbol is None, fetch all orders; else filter by symbol
            orders = mt5.orders_get() if symbol is None else mt5.orders_get(symbol=symbol)
            if orders is None:
                return []
            
            order_list = []
            for order in orders:
                order_list.append({
                    'ticket': order.ticket,
                    'symbol': order.symbol,
                    'type': order.type,
                    'volume': order.volume_initial,
                    'price_open': order.price_open,
                    'sl': order.sl,
                    'tp': order.tp,
                    'magic': order.magic,
                    'comment': order.comment,
                    'time_setup': datetime.fromtimestamp(order.time_setup)
                })
            
            return order_list
            
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []
    
    def close_position(self, ticket: int, reason: str = "Manual close") -> bool:
        """
        Close a position by ticket
        
        Args:
            ticket: Position ticket
            reason: Reason for closing the position
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.ensure_connection():
            return False
        
        try:
            # Get position info
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                logger.error(f"Position {ticket} not found")
                return False
            
            position = positions[0]
            
            # Prepare close request (with tick availability check)
            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                logger.error(f"No tick data available for symbol {position.symbol} while closing position {ticket}")
                return False
            if position.type == 0:  # Buy position
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:  # Sell position
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": position.magic,
                "comment": self._sanitize_comment(f"Close: {reason}"),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send close request
            result = mt5.order_send(request)
            success = not (result is None or getattr(result, 'retcode', None) != mt5.TRADE_RETCODE_DONE)
            if not success:
                ret = getattr(result, 'retcode', None) if result is not None else None
                comment_text = getattr(result, 'comment', '') if result is not None else ''
                logger.error(f"Failed to close position {ticket}: {ret} - {comment_text} last_error={mt5.last_error()}")
            else:
                logger.info(f"Position {ticket} closed successfully - Reason: {reason}")

            # Append to trade log regardless of success
            try:
                self._append_trade_log({
                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                    'login': self._login or 'unknown',
                    'ticket': ticket,
                    'symbol': position.symbol,
                    'type': position.type,
                    'volume': position.volume,
                    'profit': position.profit,
                    'reason': f"{reason} ({'closed' if success else 'failed'})"
                })
                # If it was a losing close, save to a dedicated losses log for learning
                try:
                    if success and float(position.profit) < 0:
                        self._append_loss_log({
                            'timestamp': datetime.now().isoformat(timespec='seconds'),
                            'login': self._login or 'unknown',
                            'ticket': ticket,
                            'symbol': position.symbol,
                            'type': position.type,
                            'volume': position.volume,
                            'profit': position.profit,
                            'reason': reason
                        })
                except Exception:
                    logger.exception("Failed to append to loss_log.csv")
            except Exception:
                logger.exception("Failed to append to trade_log.csv")

            return success
            
        except Exception as e:
            logger.error(f"Error closing position {ticket}: {e}")
            return False
    
    def close_all_positions(self, symbol: str = None, reason: str = "Close all positions") -> Dict:
        """
        Close all open positions
        
        Args:
            symbol: Trading symbol (None for all symbols)
            reason: Reason for closing positions
            
        Returns:
            Dict: Summary of closed positions with success/failure counts
        """
        if not self.ensure_connection():
            return {'success': 0, 'failed': 0, 'total': 0, 'details': []}
        
        try:
            # Try up to 3 passes to close all positions (handles fills/freeze delays)
            aggregate = {'success': 0, 'failed': 0, 'total': 0, 'details': [], 'total_profit': 0.0}
            for attempt in range(3):
                positions = self.get_positions(symbol)
                if attempt == 0:
                    aggregate['total'] = len(positions)
                if not positions:
                    break
                for pos in positions:
                    ticket = pos['ticket']
                    profit = pos['profit']
                    symbol_name = pos['symbol']
                    pos_type = pos['type']
                    volume = pos['volume']
                    success = self.close_position(ticket, reason)
                    detail = {
                        'ticket': ticket,
                        'symbol': symbol_name,
                        'type': pos_type,
                        'volume': volume,
                        'profit': profit,
                        'status': 'closed' if success else 'failed'
                    }
                    aggregate['details'].append(detail)
                    # Count aggregates
                    if success:
                        aggregate['success'] += 1
                        aggregate['total_profit'] += profit
                    else:
                        aggregate['failed'] += 1
                    # Append to trade log CSV for both success and failure
                    try:
                        self._append_trade_log({
                            'timestamp': datetime.now().isoformat(timespec='seconds'),
                            'login': self._login or 'unknown',
                            'ticket': ticket,
                            'symbol': symbol_name,
                            'type': pos_type,
                            'volume': volume,
                            'profit': profit,
                            'reason': f"{reason} ({'closed' if success else 'failed'})"
                        })
                        # Save losing closes to dedicated losses log
                        try:
                            if success and float(profit) < 0:
                                self._append_loss_log({
                                    'timestamp': datetime.now().isoformat(timespec='seconds'),
                                    'login': self._login or 'unknown',
                                    'ticket': ticket,
                                    'symbol': symbol_name,
                                    'type': pos_type,
                                    'volume': volume,
                                    'profit': profit,
                                    'reason': reason
                                })
                        except Exception:
                            logger.exception("Failed to append to loss_log.csv")
                    except Exception:
                        logger.exception("Failed to append to trade_log.csv")
                # small pause between attempts
                import time as _t
                _t.sleep(0.3)
            if aggregate['total'] == 0:
                aggregate['message'] = 'No open positions found'
            # Final check: report remaining open tickets if any
            remaining = self.get_positions(symbol)
            if remaining:
                aggregate['remaining'] = [p['ticket'] for p in remaining]
            # Persist trade close log to logs folder
            try:
                self._write_close_log(aggregate, reason)
            except Exception:
                logger.exception("Failed to write close positions log")
            return aggregate
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return {'success': 0, 'failed': 0, 'total': 0, 'details': [], 'error': str(e)}

    def close_partial_position(self, ticket: int, close_fraction: float, reason: str = "Partial take profit") -> bool:
        """Close part of an open position volume."""
        if not self.ensure_connection():
            return False
        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return False
            pos = positions[0]
            if not (0 < close_fraction < 1.0):
                return False
            close_volume = max(round(pos.volume * close_fraction, 2), 0.01)
            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                return False
            if pos.type == 0:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            req = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': pos.symbol,
                'volume': close_volume,
                'type': order_type,
                'position': ticket,
                'price': price,
                'deviation': 40,
                'magic': pos.magic,
                'comment': self._sanitize_comment(f"Partial: {reason}"),
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(req)
            if result is None or getattr(result, 'retcode', None) != mt5.TRADE_RETCODE_DONE:
                logger.warning(f"Partial close failed {ticket}: {getattr(result, 'retcode', None)} {getattr(result, 'comment', '')}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error partial closing position {ticket}: {e}")
            return False

    def cancel_all_orders(self, symbol: str = None) -> Dict:
        """Cancel all pending orders (optionally filter by symbol)."""
        if not self.ensure_connection():
            return {'success': 0, 'failed': 0, 'total': 0, 'details': []}
        try:
            orders = self.get_orders(symbol)
            if not orders:
                return {'success': 0, 'failed': 0, 'total': 0, 'details': [], 'message': 'No pending orders found'}
            result = {'success': 0, 'failed': 0, 'total': len(orders), 'details': []}
            for o in orders:
                ok = self.cancel_order(o['ticket'])
                result['success' if ok else 'failed'] += 1
                result['details'].append({
                    'ticket': o['ticket'], 'symbol': o['symbol'], 'type': o['type'], 'status': 'cancelled' if ok else 'failed'
                })
            return result
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return {'success': 0, 'failed': 0, 'total': 0, 'details': [], 'error': str(e)}
    
    def cancel_order(self, ticket: int) -> bool:
        """
        Cancel a pending order
        
        Args:
            ticket: Order ticket
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.ensure_connection():
            return False
        
        try:
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": ticket,
            }
            
            result = mt5.order_send(request)
            if result is None or getattr(result, 'retcode', None) != mt5.TRADE_RETCODE_DONE:
                ret = getattr(result, 'retcode', None) if result is not None else None
                comment_text = getattr(result, 'comment', '') if result is not None else ''
                logger.error(f"Failed to cancel order {ticket}: {ret} - {comment_text} last_error={mt5.last_error()}")
                return False
            
            logger.info(f"Order {ticket} cancelled successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error cancelling order {ticket}: {e}")
            return False

    # --- Logging helpers ---
    def _write_close_log(self, aggregate: Dict, reason: str):
        """Append to a single CSV trade log file when positions are closed."""
        import os, csv
        from datetime import datetime as _dt
        os.makedirs('logs', exist_ok=True)
        login = self._login or 'unknown'
        filename = os.path.join('logs', 'close_log.csv')
        headers = ['timestamp','login','reason','ticket','symbol','type','volume','profit','status']
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(filename)
        
        now_iso = _dt.now().isoformat(timespec='seconds')
        lines = []
        for d in aggregate.get('details', []):
            lines.append([
                now_iso,
                login,
                reason,
                d.get('ticket',''),
                d.get('symbol',''),
                d.get('type',''),
                d.get('volume',''),
                d.get('profit',''),
                d.get('status',''),
            ])
        
        # Append to the single close_log.csv file
        with open(filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerows(lines)
        logger.info(f"Close positions log appended to: {filename}")

    def _append_trade_log(self, row: Dict):
        """Append a single closed-trade row to logs/trade_log.csv"""
        import os, csv
        os.makedirs('logs', exist_ok=True)
        filename = os.path.join('logs', 'trade_log.csv')
        file_exists = os.path.exists(filename)
        headers = ['timestamp','login','ticket','symbol','type','volume','profit','reason']
        with open(filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            # sanitize minimal fields
            safe = {k: row.get(k, '') for k in headers}
            writer.writerow(safe)

    def _append_loss_log(self, row: Dict):
        """Append losing closed-trade rows to logs/loss_log.csv for focused learning"""
        import os, csv
        os.makedirs('logs', exist_ok=True)
        filename = os.path.join('logs', 'loss_log.csv')
        file_exists = os.path.exists(filename)
        headers = ['timestamp','login','ticket','symbol','type','volume','profit','reason']
        with open(filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            safe = {k: row.get(k, '') for k in headers}
            writer.writerow(safe)
    
    def get_trading_hours(self) -> bool:
        """
        Check if current time is within trading hours
        
        Returns:
            bool: True if within trading hours, False otherwise
        """
        current_hour = datetime.now().hour
        return TRADING_START_HOUR <= current_hour <= TRADING_END_HOUR
    
    def get_spread(self, symbol: str = SYMBOL) -> Optional[float]:
        """
        Get current spread in points
        
        Args:
            symbol: Trading symbol
            
        Returns:
            float: Spread in points or None if failed
        """
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None
            
            return symbol_info.spread
            
        except Exception as e:
            logger.error(f"Error getting spread: {e}")
            return None

    def modify_position_sl_tp(self, ticket: int, new_sl: Optional[float], new_tp: Optional[float]) -> bool:
        """Modify an open position's SL/TP.
        Uses TRADE_ACTION_SLTP to update stops without opening a new position.
        """
        if not self.ensure_connection():
            return False
        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return False
            pos = positions[0]
            req = {
                'action': mt5.TRADE_ACTION_SLTP,
                'position': ticket,
                'symbol': pos.symbol,
                'sl': new_sl,
                'tp': new_tp,
                'magic': pos.magic,
                'comment': self._sanitize_comment('Modify SL/TP'),
            }
            result = mt5.order_send(req)
            if result is None or getattr(result, 'retcode', None) != mt5.TRADE_RETCODE_DONE:
                logger.warning(f"SL/TP modify failed for {ticket}: {getattr(result, 'retcode', None)} {getattr(result, 'comment', '')}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error modifying SL/TP for {ticket}: {e}")
            return False

    def get_history_deals(self, start: datetime, end: datetime) -> List[Dict]:
        """
        Get account deal history between start and end.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (inclusive)

        Returns:
            List[Dict]: List of deals with basic fields
        """
        if not self.ensure_connection():
            return []
        try:
            deals = mt5.history_deals_get(start, end)
            if deals is None:
                logger.warning("No deals returned from MT5 history_deals_get")
                return []
            
            # Debug: log what symbols we found
            symbols_found = set()
            results: List[Dict] = []
            for d in deals:
                symbols_found.add(d.symbol)
                results.append({
                    'ticket': d.ticket,
                    'order': d.order,
                    'symbol': d.symbol,
                    'type': d.type,
                    'volume': d.volume,
                    'price': d.price,
                    'profit': d.profit,
                    'commission': d.commission,
                    'swap': d.swap,
                    'magic': getattr(d, 'magic', 0),
                    'comment': d.comment,
                    'time': datetime.fromtimestamp(d.time)
                })
            
            logger.info(f"History deals found: {len(results)} deals across symbols: {list(symbols_found)}")
            return results
        except Exception as e:
            logger.error(f"Error getting history deals: {e}")
            return []

    def get_recent_history(self, period: str = 'day') -> List[Dict]:
        """
        Convenience helper to get recent history by named period.
        period: 'day' | 'week' | 'month'
        """
        now = datetime.now()
        if period == 'day':
            start = now - timedelta(days=1)
        elif period == 'week':
            start = now - timedelta(weeks=1)
        else:
            start = now - timedelta(days=30)
        return self.get_history_deals(start, now)
    
    def test_connection(self) -> Dict:
        """
        Test MT5 connection and return detailed status information
        
        Returns:
            Dict: Connection test results
        """
        test_results = {
            'connection_successful': False,
            'mt5_initialized': False,
            'account_connected': False,
            'symbol_available': False,
            'data_retrieval': False,
            'errors': [],
            'warnings': [],
            'account_info': None,
            'symbol_info': None
        }
        
        try:
            # Test 1: Check if MT5 can be initialized
            logger.info("Testing MT5 initialization...")
            if mt5.initialize(path=self._terminal_path):
                test_results['mt5_initialized'] = True
                logger.info("✓ MT5 initialized successfully")
            else:
                error_code, error_desc = mt5.last_error()
                test_results['errors'].append(f"MT5 initialization failed: {error_code} - {error_desc}")
                logger.error(f"✗ MT5 initialization failed: {error_code} - {error_desc}")
                return test_results
            
            # Test 2: Check account connection
            logger.info("Testing account connection...")
            account_info = mt5.account_info()
            if account_info:
                test_results['account_connected'] = True
                test_results['account_info'] = {
                    'login': account_info.login,
                    'server': account_info.server,
                    'balance': account_info.balance,
                    'equity': account_info.equity,
                    'currency': account_info.currency
                }
                logger.info(f"✓ Connected to account {account_info.login} on {account_info.server}")
            else:
                test_results['errors'].append("No account information available")
                logger.error("✗ No account information available")
            
            # Test 3: Check symbol availability
            logger.info(f"Testing symbol availability: {self._symbol}")
            symbol_info = mt5.symbol_info(self._symbol)
            if symbol_info:
                test_results['symbol_available'] = True
                test_results['symbol_info'] = {
                    'name': symbol_info.name,
                    'point': symbol_info.point,
                    'digits': symbol_info.digits,
                    'spread': symbol_info.spread,
                    'trade_mode': symbol_info.trade_mode
                }
                logger.info(f"✓ Symbol {self._symbol} is available")
            else:
                test_results['errors'].append(f"Symbol {self._symbol} not available")
                logger.error(f"✗ Symbol {self._symbol} not available")
            
            # Test 4: Test data retrieval
            logger.info("Testing data retrieval...")
            rates = mt5.copy_rates_from_pos(self._symbol, TIMEFRAME_MAPPING.get(self._timeframe, 15), 0, 10)
            if rates is not None and len(rates) > 0:
                test_results['data_retrieval'] = True
                logger.info(f"✓ Successfully retrieved {len(rates)} bars of data")
            else:
                test_results['errors'].append("Failed to retrieve market data")
                logger.error("✗ Failed to retrieve market data")
            
            # Overall connection status
            test_results['connection_successful'] = (
                test_results['mt5_initialized'] and 
                test_results['account_connected'] and 
                test_results['symbol_available'] and 
                test_results['data_retrieval']
            )
            
            if test_results['connection_successful']:
                logger.info("✓ All connection tests passed!")
            else:
                logger.error("✗ Some connection tests failed")
            
        except Exception as e:
            test_results['errors'].append(f"Connection test error: {str(e)}")
            logger.error(f"✗ Connection test error: {e}")
        
        finally:
            # Clean up
            try:
                mt5.shutdown()
            except Exception:
                pass
        
        return test_results
