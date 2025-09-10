"""
MetaTrader 5 Connector Module
Handles all MT5 API interactions including connection, data retrieval, and order management
"""

import MetaTrader5 as mt5
import numpy as np

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

class MT5Connector:
    """MetaTrader 5 connection and trading operations handler.

    Supports per-instance credentials and symbol/timeframe overrides for multi-account usage.
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
        
    def connect(self) -> bool:
        """
        Establish connection to MetaTrader 5
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Initialize MT5
            if not mt5.initialize(path=self._terminal_path):
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Login to account
            if not mt5.login(login=self._login, password=self._password, server=self._server):
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                return False
            
            # Get account info
            self.account_info = mt5.account_info()
            if self.account_info is None:
                logger.error("Failed to get account info")
                return False
            
            # Get symbol info
            self.symbol_info = mt5.symbol_info(self._symbol)
            if self.symbol_info is None:
                logger.error(f"Failed to get symbol info for {self._symbol}")
                return False
            
            # Enable symbol for trading
            if not mt5.symbol_select(self._symbol, True):
                logger.error(f"Failed to select symbol {self._symbol}")
                return False
            
            self.connected = True
            logger.info(f"Successfully connected to MT5. Account: {self.account_info.login}")
            logger.info(f"Balance: {self.account_info.balance}, Equity: {self.account_info.equity}")
            
            return True
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MetaTrader 5"""
        mt5.shutdown()
        self.connected = False
        logger.info("Disconnected from MT5")

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
        if not self.connected:
            return None
        
        account_info = mt5.account_info()
        if account_info is None:
            return None
        
        return {
            'login': account_info.login,
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'margin_level': account_info.margin_level,
            'currency': account_info.currency,
            'leverage': account_info.leverage
        }

    def change_symbol(self, symbol: str) -> bool:
        """Switch working symbol and select it in terminal."""
        try:
            if not symbol:
                return False
            self._symbol = symbol
            # ensure selected
            return mt5.symbol_select(symbol, True)
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
        if not self.connected:
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
        if not self.connected:
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
        if not self.connected:
            return None
        
        try:
            # Get current price if not specified
            if price is None:
                current_price = self.get_current_price()
                if current_price is None:
                    return None
                price = current_price['ask'] if order_type == 'buy' else current_price['bid']
            
            # Enforce stops level and freeze level
            stops_level_points = getattr(self.symbol_info, 'trade_stops_level', 0)
            freeze_level_points = getattr(self.symbol_info, 'trade_freeze_level', 0)
            point = self.symbol_info.point if self.symbol_info else 0.0001
            min_stop_dist = max(stops_level_points, freeze_level_points) * point

            if sl is not None and abs(price - sl) < min_stop_dist:
                sl = price - min_stop_dist if order_type.lower() == 'buy' else price + min_stop_dist
            if tp is not None and abs(tp - price) < min_stop_dist:
                tp = price + min_stop_dist if order_type.lower() == 'buy' else price - min_stop_dist

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
                "deviation": 50,
                "magic": magic,
                "comment": self._sanitize_comment(comment),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            if result is None:
                logger.error(f"order_send returned None: {mt5.last_error()}")
                return None
            
            if getattr(result, 'retcode', None) != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {getattr(result, 'retcode', 'N/A')} - {getattr(result, 'comment', '')} last_error={mt5.last_error()}")
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
        if not self.connected:
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
                "symbol": SYMBOL,
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
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Pending order failed: {result.retcode} - {result.comment}")
                return None
            
            logger.info(f"Pending order placed: {order_type} {volume} {SYMBOL} at {price}")
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
        if not self.connected:
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
        if not self.connected:
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
    
    def close_position(self, ticket: int) -> bool:
        """
        Close a position by ticket
        
        Args:
            ticket: Position ticket
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connected:
            return False
        
        try:
            # Get position info
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                logger.error(f"Position {ticket} not found")
                return False
            
            position = positions[0]
            
            # Prepare close request
            if position.type == 0:  # Buy position
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(position.symbol).bid
            else:  # Sell position
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(position.symbol).ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": position.magic,
                "comment": "Close position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send close request
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Failed to close position {ticket}: {result.retcode} - {result.comment}")
                return False
            
            logger.info(f"Position {ticket} closed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position {ticket}: {e}")
            return False
    
    def close_all_positions(self, symbol: str = None) -> Dict:
        """
        Close all open positions
        
        Args:
            symbol: Trading symbol (None for all symbols)
            
        Returns:
            Dict: Summary of closed positions with success/failure counts
        """
        if not self.connected:
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
                    success = self.close_position(ticket)
                    detail = {
                        'ticket': ticket,
                        'symbol': symbol_name,
                        'type': pos_type,
                        'volume': volume,
                        'profit': profit,
                        'status': 'closed' if success else 'failed'
                    }
                    aggregate['details'].append(detail)
                    if success:
                        aggregate['success'] += 1
                        aggregate['total_profit'] += profit
                    else:
                        aggregate['failed'] += 1
                # small pause between attempts
                import time as _t
                _t.sleep(0.3)
            if aggregate['total'] == 0:
                aggregate['message'] = 'No open positions found'
            # Final check: report remaining open tickets if any
            remaining = self.get_positions(symbol)
            if remaining:
                aggregate['remaining'] = [p['ticket'] for p in remaining]
            return aggregate
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return {'success': 0, 'failed': 0, 'total': 0, 'details': [], 'error': str(e)}

    def cancel_all_orders(self, symbol: str = None) -> Dict:
        """Cancel all pending orders (optionally filter by symbol)."""
        if not self.connected:
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
        if not self.connected:
            return False
        
        try:
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": ticket,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Failed to cancel order {ticket}: {result.retcode} - {result.comment}")
                return False
            
            logger.info(f"Order {ticket} cancelled successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {ticket}: {e}")
            return False
    
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

    def get_history_deals(self, start: datetime, end: datetime) -> List[Dict]:
        """
        Get account deal history between start and end.

        Args:
            start: Start datetime (inclusive)
            end: End datetime (inclusive)

        Returns:
            List[Dict]: List of deals with basic fields
        """
        if not self.connected:
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
