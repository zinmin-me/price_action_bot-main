#!/usr/bin/env python3
"""
Script to create terminal for existing admin/user who is already in database
This is useful when you have an admin user but no terminal configured yet
"""

import sys
import logging
from database import db_manager
from auto_terminal_manager import auto_terminal_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_terminal_for_existing_user(telegram_chat_id: int, mt_account_number: int):
    """
    Create terminal for an existing user who is already in database
    
    Args:
        telegram_chat_id: Telegram chat ID of existing user
        mt_account_number: MT5 account number for the user
    """
    print(f"🔧 Creating terminal for existing user...")
    print("=" * 50)
    
    try:
        # Check if user exists in database
        user = db_manager.get_bot_user_by_telegram_chat_id(telegram_chat_id)
        if not user:
            print(f"❌ User with Telegram Chat ID {telegram_chat_id} not found in database!")
            print("💡 Use 'python manage_users.py add {telegram_chat_id}' to add the user first.")
            return False
        
        print(f"✅ Found user in database:")
        print(f"   Bot User ID: {user['bot_user_id']}")
        print(f"   Telegram Chat ID: {user['telegram_chat_id']}")
        print(f"   Role: {'👑 Admin' if user['is_admin'] else '👤 User'}")
        
        # Check if user already has an MT account
        existing_account = db_manager.get_mt_account_by_bot_user_id(user['bot_user_id'])
        if existing_account:
            print(f"\n⚠️  User already has an MT account:")
            print(f"   Account Number: {existing_account['mt_account_number']}")
            print(f"   Terminal Name: {existing_account['terminal_name']}")
            
            response = input("\nDo you want to update the account number? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("❌ Operation cancelled.")
                return False
        
        # Initialize auto terminal manager
        if not auto_terminal_manager.initialize():
            print("❌ Failed to initialize Auto Terminal Manager")
            return False
        
        # Create terminal for the user
        print(f"\n🖥️ Creating terminal for user...")
        success = auto_terminal_manager.create_terminal_for_user(
            user['bot_user_id'], 
            mt_account_number
        )
        
        if success:
            print(f"✅ Terminal created successfully!")
            print(f"   Terminal Name: user_{mt_account_number}")
            print(f"   Account Number: {mt_account_number}")
            print(f"\n🎉 User can now login with:")
            print(f"   /login {mt_account_number} <password> <server>")
            return True
        else:
            print(f"❌ Failed to create terminal for user")
            return False
            
    except Exception as e:
        print(f"❌ Error creating terminal: {e}")
        return False

def list_existing_users():
    """List all existing users in database"""
    print("📋 Existing users in database:")
    print("=" * 40)
    
    try:
        users = db_manager.get_all_bot_users()
        if not users:
            print("No users found in database.")
            return
        
        for user in users:
            role = "👑 Admin" if user['is_admin'] else "👤 User"
            print(f"• User {user['bot_user_id']}: {user['telegram_chat_id']} ({role})")
            
            # Check if user has MT account
            account = db_manager.get_mt_account_by_bot_user_id(user['bot_user_id'])
            if account:
                print(f"  └─ MT Account: {account['mt_account_number']} (Terminal: {account['terminal_name']})")
            else:
                print(f"  └─ No MT account configured")
        
    except Exception as e:
        print(f"❌ Error listing users: {e}")

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python create_terminal_for_existing_user.py <telegram_chat_id> <mt_account_number>")
        print("  python create_terminal_for_existing_user.py list")
        print()
        print("Examples:")
        print("  python create_terminal_for_existing_user.py 123456789 11045991")
        print("  python create_terminal_for_existing_user.py list")
        return
    
    if sys.argv[1] == "list":
        list_existing_users()
        return
    
    try:
        telegram_chat_id = int(sys.argv[1])
        mt_account_number = int(sys.argv[2])
    except ValueError:
        print("❌ Invalid arguments. telegram_chat_id and mt_account_number must be numbers.")
        return
    
    create_terminal_for_existing_user(telegram_chat_id, mt_account_number)

if __name__ == "__main__":
    main()
