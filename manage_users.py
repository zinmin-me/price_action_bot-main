#!/usr/bin/env python3
"""
User Management Script for Price Action Trading Bot
Allows administrators to manage bot users and view database statistics
"""

import sys
import argparse
from database import db_manager

def add_user(telegram_chat_id, is_admin=False):
    """Add a new bot user"""
    try:
        bot_user_id = db_manager.add_bot_user(telegram_chat_id, is_admin)
        if bot_user_id:
            role = "Admin" if is_admin else "Regular User"
            print(f"✅ Added bot user successfully!")
            print(f"Bot User ID: {bot_user_id}")
            print(f"Telegram Chat ID: {telegram_chat_id}")
            print(f"Role: {role}")
        else:
            print("❌ Failed to add bot user. User might already exist.")
    except Exception as e:
        print(f"❌ Error adding user: {e}")

def list_users():
    """List all bot users"""
    try:
        users = db_manager.get_all_bot_users()
        
        if not users:
            print("📋 No users found.")
            return
        
        print("📋 Bot Users:")
        print("-" * 50)
        
        for user in users:
            # Check if user has an MT account
            mt_account = db_manager.get_mt_account_by_bot_user_id(user['bot_user_id'])
            account_status = f"MT Account: {mt_account['mt_account_number']}" if mt_account else "No MT Account"
            role_status = "👑 Admin" if user['is_admin'] else "👤 User"
            
            print(f"User {user['bot_user_id']}:")
            print(f"  Telegram Chat ID: {user['telegram_chat_id']}")
            print(f"  Role: {role_status}")
            print(f"  {account_status}")
            print(f"  Created: {user['created_at']}")
            print()
            
    except Exception as e:
        print(f"❌ Error listing users: {e}")

def show_stats():
    """Show database statistics"""
    try:
        stats = db_manager.get_database_stats()
        
        print("📊 Database Statistics:")
        print("-" * 30)
        print(f"Total Bot Users: {stats['total_bot_users']}")
        print(f"Total MT Accounts: {stats['total_mt_accounts']}")
        print(f"Active Sessions: {stats['active_sessions']}")
        print(f"Database Path: {stats['database_path']}")
        
        if 'error' in stats:
            print(f"Error: {stats['error']}")
            
    except Exception as e:
        print(f"❌ Error getting stats: {e}")

def list_mt_accounts():
    """List all MT accounts"""
    try:
        accounts = db_manager.get_all_mt_accounts()
        
        if not accounts:
            print("📋 No MT accounts found.")
            return
        
        print("📋 MT Accounts:")
        print("-" * 50)
        
        for account in accounts:
            print(f"MT Account {account['mt_account_id']}:")
            print(f"  Bot User ID: {account['bot_user_id']}")
            print(f"  Telegram Chat ID: {account['telegram_chat_id']}")
            print(f"  MT Account Number: {account['mt_account_number']}")
            print(f"  Created: {account['created_at']}")
            print()
            
    except Exception as e:
        print(f"❌ Error listing MT accounts: {e}")

def remove_user(telegram_chat_id):
    """Remove a bot user and their MT account"""
    try:
        # Get user info
        user = db_manager.get_bot_user_by_telegram_chat_id(telegram_chat_id)
        if not user:
            print(f"❌ User with telegram_chat_id {telegram_chat_id} not found.")
            return
        
        # Remove MT account if exists
        mt_account = db_manager.get_mt_account_by_bot_user_id(user['bot_user_id'])
        if mt_account:
            db_manager.remove_mt_account(user['bot_user_id'])
            print(f"✅ Removed MT account {mt_account['mt_account_number']}")
        
        # Note: We don't have a remove_bot_user method in the database manager
        # This would need to be added if you want to completely remove users
        print(f"⚠️  Bot user {user['bot_user_id']} still exists in database.")
        print("   To completely remove, you would need to delete from database manually.")
        
    except Exception as e:
        print(f"❌ Error removing user: {e}")

def main():
    parser = argparse.ArgumentParser(description='Manage bot users for Price Action Trading Bot')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add user command
    add_parser = subparsers.add_parser('add', help='Add a new bot user')
    add_parser.add_argument('telegram_chat_id', type=int, help='Telegram chat ID')
    add_parser.add_argument('--admin', action='store_true', help='Make this user an admin')
    
    # List users command
    subparsers.add_parser('list', help='List all bot users')
    
    # Show stats command
    subparsers.add_parser('stats', help='Show database statistics')
    
    # List MT accounts command
    subparsers.add_parser('mt_accounts', help='List all MT accounts')
    
    # Remove user command
    remove_parser = subparsers.add_parser('remove', help='Remove a bot user')
    remove_parser.add_argument('telegram_chat_id', type=int, help='Telegram chat ID')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'add':
        add_user(args.telegram_chat_id, args.admin)
    elif args.command == 'list':
        list_users()
    elif args.command == 'stats':
        show_stats()
    elif args.command == 'mt_accounts':
        list_mt_accounts()
    elif args.command == 'remove':
        remove_user(args.telegram_chat_id)

if __name__ == '__main__':
    main()
