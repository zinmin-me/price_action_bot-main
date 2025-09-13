"""
Database module for Price Action Trading Bot
Manages bot_user and mt_account tables for Telegram bot access control
"""

import sqlite3
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations for bot users and MT5 accounts"""
    
    def __init__(self, db_path: str = "trading_bot.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database and create tables if they don't exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create bot_user table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS bot_user (
                        bot_user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        telegram_chat_id INTEGER UNIQUE NOT NULL,
                        is_admin BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Add is_admin column if it doesn't exist (for existing databases)
                try:
                    cursor.execute("ALTER TABLE bot_user ADD COLUMN is_admin BOOLEAN DEFAULT FALSE")
                    conn.commit()
                except sqlite3.OperationalError:
                    # Column already exists, ignore
                    pass
                
                # Create mt_account table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS mt_account (
                        mt_account_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        bot_user_id INTEGER NOT NULL,
                        mt_account_number INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (bot_user_id) REFERENCES bot_user (bot_user_id),
                        UNIQUE(bot_user_id, mt_account_number)
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_bot_user_telegram_chat_id 
                    ON bot_user (telegram_chat_id)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_mt_account_bot_user_id 
                    ON mt_account (bot_user_id)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_mt_account_number 
                    ON mt_account (mt_account_number)
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def add_bot_user(self, telegram_chat_id: int, is_admin: bool = False) -> Optional[int]:
        """
        Add a new bot user to the database
        
        Args:
            telegram_chat_id: Telegram chat ID
            is_admin: Whether this user is an admin
            
        Returns:
            bot_user_id if successful, None if failed
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if user already exists
                cursor.execute(
                    "SELECT bot_user_id FROM bot_user WHERE telegram_chat_id = ?",
                    (telegram_chat_id,)
                )
                existing = cursor.fetchone()
                
                if existing:
                    logger.info(f"Bot user with telegram_chat_id {telegram_chat_id} already exists")
                    return existing[0]
                
                # Insert new user
                cursor.execute(
                    "INSERT INTO bot_user (telegram_chat_id, is_admin) VALUES (?, ?)",
                    (telegram_chat_id, is_admin)
                )
                
                bot_user_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Added bot user {bot_user_id} for telegram_chat_id {telegram_chat_id}")
                return bot_user_id
                
        except Exception as e:
            logger.error(f"Failed to add bot user: {e}")
            return None
    
    def get_bot_user_by_telegram_chat_id(self, telegram_chat_id: int) -> Optional[Dict]:
        """
        Get bot user by telegram chat ID
        
        Args:
            telegram_chat_id: Telegram chat ID
            
        Returns:
            Dictionary with user info or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT bot_user_id, telegram_chat_id, is_admin, created_at, updated_at FROM bot_user WHERE telegram_chat_id = ?",
                    (telegram_chat_id,)
                )
                
                result = cursor.fetchone()
                
                if result:
                    return {
                        'bot_user_id': result[0],
                        'telegram_chat_id': result[1],
                        'is_admin': bool(result[2]),
                        'created_at': result[3],
                        'updated_at': result[4]
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get bot user: {e}")
            return None
    
    def is_telegram_user_authorized(self, telegram_chat_id: int) -> bool:
        """
        Check if a telegram chat ID is authorized to use the bot
        
        Args:
            telegram_chat_id: Telegram chat ID
            
        Returns:
            True if authorized, False otherwise
        """
        user = self.get_bot_user_by_telegram_chat_id(telegram_chat_id)
        return user is not None
    
    def is_user_admin(self, telegram_chat_id: int) -> bool:
        """
        Check if a telegram chat ID is an admin
        
        Args:
            telegram_chat_id: Telegram chat ID
            
        Returns:
            True if admin, False otherwise
        """
        user = self.get_bot_user_by_telegram_chat_id(telegram_chat_id)
        return user is not None and user.get('is_admin', False)
    
    def add_mt_account(self, bot_user_id: int, mt_account_number: int) -> Optional[int]:
        """
        Add or update MT5 account for a bot user
        
        Args:
            bot_user_id: Bot user ID
            mt_account_number: MT5 account number
            
        Returns:
            mt_account_id if successful, None if failed
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if user already has an MT account
                cursor.execute(
                    "SELECT mt_account_id FROM mt_account WHERE bot_user_id = ?",
                    (bot_user_id,)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing account
                    cursor.execute(
                        "UPDATE mt_account SET mt_account_number = ?, updated_at = CURRENT_TIMESTAMP WHERE bot_user_id = ?",
                        (mt_account_number, bot_user_id)
                    )
                    mt_account_id = existing[0]
                    logger.info(f"Updated MT account {mt_account_id} for bot_user_id {bot_user_id}")
                else:
                    # Insert new account
                    cursor.execute(
                        "INSERT INTO mt_account (bot_user_id, mt_account_number) VALUES (?, ?)",
                        (bot_user_id, mt_account_number)
                    )
                    mt_account_id = cursor.lastrowid
                    logger.info(f"Added MT account {mt_account_id} for bot_user_id {bot_user_id}")
                
                conn.commit()
                return mt_account_id
                
        except Exception as e:
            logger.error(f"Failed to add/update MT account: {e}")
            return None
    
    def get_mt_account_by_bot_user_id(self, bot_user_id: int) -> Optional[Dict]:
        """
        Get MT5 account by bot user ID
        
        Args:
            bot_user_id: Bot user ID
            
        Returns:
            Dictionary with MT account info or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT mt_account_id, bot_user_id, mt_account_number, created_at, updated_at FROM mt_account WHERE bot_user_id = ?",
                    (bot_user_id,)
                )
                
                result = cursor.fetchone()
                
                if result:
                    return {
                        'mt_account_id': result[0],
                        'bot_user_id': result[1],
                        'mt_account_number': result[2],
                        'created_at': result[3],
                        'updated_at': result[4]
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get MT account: {e}")
            return None
    
    def get_mt_account_by_telegram_chat_id(self, telegram_chat_id: int) -> Optional[Dict]:
        """
        Get MT5 account by telegram chat ID
        
        Args:
            telegram_chat_id: Telegram chat ID
            
        Returns:
            Dictionary with MT account info or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT ma.mt_account_id, ma.bot_user_id, ma.mt_account_number, 
                           ma.created_at, ma.updated_at
                    FROM mt_account ma
                    JOIN bot_user bu ON ma.bot_user_id = bu.bot_user_id
                    WHERE bu.telegram_chat_id = ?
                """, (telegram_chat_id,))
                
                result = cursor.fetchone()
                
                if result:
                    return {
                        'mt_account_id': result[0],
                        'bot_user_id': result[1],
                        'mt_account_number': result[2],
                        'created_at': result[3],
                        'updated_at': result[4]
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get MT account by telegram chat ID: {e}")
            return None
    
    def remove_mt_account(self, bot_user_id: int) -> bool:
        """
        Remove MT5 account for a bot user (logout)
        
        Args:
            bot_user_id: Bot user ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    "DELETE FROM mt_account WHERE bot_user_id = ?",
                    (bot_user_id,)
                )
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logger.info(f"Removed MT account for bot_user_id {bot_user_id}")
                    return True
                else:
                    logger.info(f"No MT account found for bot_user_id {bot_user_id}")
                    return False
                
        except Exception as e:
            logger.error(f"Failed to remove MT account: {e}")
            return False
    
    def get_all_bot_users(self) -> List[Dict]:
        """
        Get all bot users
        
        Returns:
            List of dictionaries with user info
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT bot_user_id, telegram_chat_id, is_admin, created_at, updated_at FROM bot_user ORDER BY created_at"
                )
                
                results = cursor.fetchall()
                
                users = []
                for result in results:
                    users.append({
                        'bot_user_id': result[0],
                        'telegram_chat_id': result[1],
                        'is_admin': bool(result[2]),
                        'created_at': result[3],
                        'updated_at': result[4]
                    })
                
                return users
                
        except Exception as e:
            logger.error(f"Failed to get all bot users: {e}")
            return []
    
    def get_all_mt_accounts(self) -> List[Dict]:
        """
        Get all MT5 accounts with user info
        
        Returns:
            List of dictionaries with account and user info
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT ma.mt_account_id, ma.bot_user_id, ma.mt_account_number,
                           bu.telegram_chat_id, ma.created_at, ma.updated_at
                    FROM mt_account ma
                    JOIN bot_user bu ON ma.bot_user_id = bu.bot_user_id
                    ORDER BY ma.created_at
                """)
                
                results = cursor.fetchall()
                
                accounts = []
                for result in results:
                    accounts.append({
                        'mt_account_id': result[0],
                        'bot_user_id': result[1],
                        'mt_account_number': result[2],
                        'telegram_chat_id': result[3],
                        'created_at': result[4],
                        'updated_at': result[5]
                    })
                
                return accounts
                
        except Exception as e:
            logger.error(f"Failed to get all MT accounts: {e}")
            return []
    
    def get_database_stats(self) -> Dict:
        """
        Get database statistics
        
        Returns:
            Dictionary with database stats
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count bot users
                cursor.execute("SELECT COUNT(*) FROM bot_user")
                bot_users_count = cursor.fetchone()[0]
                
                # Count MT accounts
                cursor.execute("SELECT COUNT(*) FROM mt_account")
                mt_accounts_count = cursor.fetchone()[0]
                
                # Count active sessions (users with MT accounts)
                cursor.execute("""
                    SELECT COUNT(*) FROM bot_user bu
                    INNER JOIN mt_account ma ON bu.bot_user_id = ma.bot_user_id
                """)
                active_sessions_count = cursor.fetchone()[0]
                
                return {
                    'total_bot_users': bot_users_count,
                    'total_mt_accounts': mt_accounts_count,
                    'active_sessions': active_sessions_count,
                    'database_path': self.db_path
                }
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {
                'total_bot_users': 0,
                'total_mt_accounts': 0,
                'active_sessions': 0,
                'database_path': self.db_path,
                'error': str(e)
            }
    
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """
        Clean up old MT account sessions (optional maintenance function)
        
        Args:
            days_old: Number of days to consider as old
            
        Returns:
            Number of sessions cleaned up
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM mt_account 
                    WHERE updated_at < datetime('now', '-{} days')
                """.format(days_old))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old MT account sessions")
                
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            return 0


# Global database instance
db_manager = DatabaseManager()
