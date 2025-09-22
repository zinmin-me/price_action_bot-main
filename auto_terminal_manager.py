"""
Automatic Terminal Manager
Loads terminal configurations from database and manages them automatically
"""

import os
import logging
from typing import Dict, List, Optional
from terminal_manager import TerminalManager, TerminalConfig
from database import DatabaseManager

logger = logging.getLogger(__name__)

class AutoTerminalManager:
    """
    Automatically manages terminals based on database user accounts
    Each user gets their own dedicated terminal
    """
    
    def __init__(self, db_manager: DatabaseManager = None):
        self.db_manager = db_manager or DatabaseManager()
        # Use the global terminal manager instance instead of creating a new one
        from terminal_manager import terminal_manager
        self.terminal_manager = terminal_manager
        self._initialized = False
        
    def initialize(self) -> bool:
        """
        Initialize the auto terminal manager by loading terminals from database
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Initializing Auto Terminal Manager...")
            
            # Get all MT accounts from database
            mt_accounts = self.db_manager.get_all_mt_accounts()
            
            if not mt_accounts:
                logger.info("No MT accounts found in database")
                return True
            
            logger.info(f"Found {len(mt_accounts)} MT accounts in database")
            
            # Create terminal configurations for each account
            terminals_created = 0
            for account in mt_accounts:
                if self._create_terminal_for_account(account):
                    terminals_created += 1
            
            logger.info(f"Created {terminals_created} terminal configurations")
            
            # Start auto-start terminals
            auto_start_terminals = [
                name for name, config in self.terminal_manager.terminals.items() 
                if config.auto_start
            ]
            
            if auto_start_terminals:
                logger.info(f"Auto-starting terminals: {auto_start_terminals}")
                for terminal_name in auto_start_terminals:
                    self.terminal_manager.start_terminal(terminal_name)
            
            # Start monitoring
            self.terminal_manager.start_monitoring()
            logger.info("Terminal monitoring started")
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Auto Terminal Manager: {e}")
            return False
    
    def _create_terminal_for_account(self, account: Dict) -> bool:
        """
        Create a terminal configuration for a database account (without credentials)
        
        Args:
            account: Account dictionary from database
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Skip accounts without required information
            if not account.get('mt_account_number') or not account.get('terminal_name'):
                logger.warning(f"Skipping account {account.get('mt_account_number')} - missing required info")
                return False
            
            # Create terminal configuration (credentials will be provided during login)
            config = TerminalConfig(
                name=account['terminal_name'],
                terminal_path=self._get_mt5_path(),
                login=account['mt_account_number'],
                password="",  # Will be provided during login
                server="",    # Will be provided during login
                symbol="EURUSD",  # Use global config
                timeframe="M15",  # Use global config
                auto_start=False  # Don't auto-start without credentials
            )
            
            # Add to terminal manager
            if self.terminal_manager.add_terminal(config):
                logger.info(f"Created terminal '{config.name}' for account {config.login}")
                return True
            else:
                logger.error(f"Failed to create terminal for account {config.login}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating terminal for account {account.get('mt_account_number')}: {e}")
            return False
    
    def _get_mt5_path(self) -> str:
        """Get the default MT5 terminal path"""
        # 1) Environment override
        try:
            env_path = os.getenv('MT5_TERMINAL_PATH')
            if env_path and os.path.exists(env_path):
                return env_path
        except Exception:
            pass

        # 2) Common install and user-writable locations
        possible_paths = [
            r"C:\\Program Files\\MetaTrader 5\\terminal64.exe",
            r"C:\\Program Files (x86)\\MetaTrader 5\\terminal64.exe",
            r"C:\\Program Files\\MetaTrader 5\\terminal.exe",
            r"C:\\Program Files (x86)\\MetaTrader 5\\terminal.exe",
            r"C:\\MT5\\terminal64.exe",
            r"C:\\MT5\\terminal.exe",
        ]
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    return path
            except Exception:
                continue

        # 3) User profile installs
        try:
            username = os.getenv('USERNAME', '')
            if username:
                import glob
                user_patterns = [
                    rf"C:\\Users\\{username}\\AppData\\Roaming\\MetaQuotes\\Terminal\\*\\terminal64.exe",
                    rf"C:\\Users\\{username}\\AppData\\Roaming\\MetaQuotes\\Terminal\\*\\terminal.exe",
                    rf"C:\\Users\\{username}\\AppData\\Local\\Programs\\MetaTrader 5\\terminal64.exe",
                    rf"C:\\Users\\{username}\\AppData\\Local\\Programs\\MetaTrader 5\\terminal.exe",
                ]
                for pattern in user_patterns:
                    for match in glob.glob(pattern):
                        if os.path.exists(match):
                            return match
        except Exception:
            pass

        # Final fallback
        return r"C:\\Program Files\\MetaTrader 5\\terminal64.exe"
    
    def create_terminal_for_user(self, bot_user_id: int, mt_account_number: int) -> bool:
        """
        Create a terminal for a new user (minimal storage)
        
        Args:
            bot_user_id: Bot user ID
            mt_account_number: MT5 account number
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get telegram chat ID for correct terminal naming
            telegram_chat_id = None
            try:
                user = self.db_manager.get_bot_user_by_id(bot_user_id)
                if user:
                    telegram_chat_id = user['telegram_chat_id']
            except Exception as e:
                logger.warning(f"Could not get telegram chat ID for user {bot_user_id}: {e}")
            
            # Add account to database (minimal storage)
            mt_account_id = self.db_manager.add_mt_account(
                bot_user_id, mt_account_number, telegram_chat_id
            )
            
            if not mt_account_id:
                logger.error(f"Failed to add MT account to database for user {bot_user_id}")
                return False
            
            # Get the account from database to get the terminal name
            account = self.db_manager.get_mt_account_by_bot_user_id(bot_user_id)
            if not account:
                logger.error(f"Failed to retrieve MT account from database for user {bot_user_id}")
                return False
            
            # Create terminal configuration (credentials will be provided during login)
            config = TerminalConfig(
                name=account['terminal_name'],
                terminal_path=self._get_mt5_path(),
                login=account['mt_account_number'],
                password="",  # Will be provided during login
                server="",    # Will be provided during login
                symbol="EURUSD",  # Use global config
                timeframe="M15",  # Use global config
                auto_start=False  # Don't auto-start without credentials
            )
            
            # Add to terminal manager
            if self.terminal_manager.add_terminal(config):
                logger.info(f"Created terminal '{config.name}' for new user {bot_user_id}")
                return True
            else:
                logger.error(f"Failed to create terminal for new user {bot_user_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating terminal for user {bot_user_id}: {e}")
            return False
    
    def get_terminal_for_user(self, bot_user_id: int) -> Optional[str]:
        """
        Get the terminal name for a specific user
        
        Args:
            bot_user_id: Bot user ID
            
        Returns:
            Terminal name if found, None otherwise
        """
        try:
            account = self.db_manager.get_mt_account_by_bot_user_id(bot_user_id)
            if account and account.get('terminal_name'):
                return account['terminal_name']
            return None
        except Exception as e:
            logger.error(f"Error getting terminal for user {bot_user_id}: {e}")
            return None
    
    def get_terminal_for_account(self, mt_account_number: int) -> Optional[str]:
        """
        Get the terminal name for a specific MT account number
        
        Args:
            mt_account_number: MT5 account number
            
        Returns:
            Terminal name if found, None otherwise
        """
        try:
            # Find account by MT account number
            accounts = self.db_manager.get_all_mt_accounts()
            for account in accounts:
                if account['mt_account_number'] == mt_account_number:
                    return account.get('terminal_name')
            return None
        except Exception as e:
            logger.error(f"Error getting terminal for account {mt_account_number}: {e}")
            return None
    
    def remove_terminal_for_user(self, bot_user_id: int) -> bool:
        """
        Remove terminal for a user
        
        Args:
            bot_user_id: Bot user ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get terminal name
            terminal_name = self.get_terminal_for_user(bot_user_id)
            if not terminal_name:
                logger.warning(f"No terminal found for user {bot_user_id}")
                return True
            
            # Stop and remove terminal
            self.terminal_manager.stop_terminal(terminal_name)
            self.terminal_manager.remove_terminal(terminal_name)
            
            logger.info(f"Removed terminal '{terminal_name}' for user {bot_user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing terminal for user {bot_user_id}: {e}")
            return False
    
    def get_all_terminals_status(self) -> Dict:
        """
        Get status of all terminals
        
        Returns:
            Dict: Terminal status information
        """
        return self.terminal_manager.get_terminal_status()
    
    def refresh_terminals(self) -> bool:
        """
        Refresh terminals from database (useful when database is updated)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Refreshing terminals from database...")
            
            # Stop all current terminals
            self.terminal_manager.stop_all_terminals()
            
            # Clear current terminals
            for terminal_name in list(self.terminal_manager.terminals.keys()):
                self.terminal_manager.remove_terminal(terminal_name)
            
            # Reinitialize
            return self.initialize()
            
        except Exception as e:
            logger.error(f"Error refreshing terminals: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.terminal_manager:
                self.terminal_manager.cleanup()
            logger.info("Auto Terminal Manager cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up Auto Terminal Manager: {e}")

# Global instance
auto_terminal_manager = AutoTerminalManager()
