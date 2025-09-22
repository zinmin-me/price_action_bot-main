#!/usr/bin/env python3
"""
Script to delete a specific user from the database
Allows deletion of individual users by Telegram Chat ID
"""

import sys
import logging
from database import db_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def delete_specific_user(telegram_chat_id: int, confirm=True):
    """
    Delete a specific user from the database
    
    Args:
        telegram_chat_id: Telegram Chat ID of user to delete
        confirm: Whether to ask for confirmation before deletion
    """
    print(f"🗑️ Deleting specific user: {telegram_chat_id}")
    print("=" * 50)
    
    try:
        # Check if user exists
        user = db_manager.get_bot_user_by_telegram_chat_id(telegram_chat_id)
        if not user:
            print(f"❌ User with Telegram Chat ID {telegram_chat_id} not found in database!")
            return False
        
        # Check if user is admin
        if user['is_admin']:
            print(f"⚠️  WARNING: User {telegram_chat_id} is an ADMIN!")
            print("❌ Admin users cannot be deleted for security reasons.")
            print("💡 Only regular users can be deleted.")
            return False
        
        print(f"✅ Found user to delete:")
        print(f"   Bot User ID: {user['bot_user_id']}")
        print(f"   Telegram Chat ID: {user['telegram_chat_id']}")
        print(f"   Role: Regular User")
        
        # Check if user has MT account
        account = db_manager.get_mt_account_by_bot_user_id(user['bot_user_id'])
        if account:
            print(f"   MT Account: {account['mt_account_number']}")
            print(f"   Terminal: {account['terminal_name']}")
        else:
            print(f"   MT Account: None")
        
        if confirm:
            response = input(f"\nAre you sure you want to delete user {telegram_chat_id}? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("❌ Deletion cancelled.")
                return False
        
        # Delete user
        print(f"\n🗑️ Deleting user {telegram_chat_id}...")
        
        # Remove MT account first (if exists)
        if account:
            if db_manager.remove_mt_account(user['bot_user_id']):
                print(f"  ✅ Removed MT account: {account['mt_account_number']}")
            else:
                print(f"  ❌ Failed to remove MT account")
        
        # Remove user
        if db_manager.remove_bot_user(telegram_chat_id):
            print(f"  ✅ Deleted user: {telegram_chat_id}")
            print(f"\n🎉 User deletion complete!")
            return True
        else:
            print(f"  ❌ Failed to delete user: {telegram_chat_id}")
            return False
        
    except Exception as e:
        print(f"❌ Error deleting user: {e}")
        return False

def list_users_for_deletion():
    """List all users that can be deleted (non-admin users)"""
    print("📋 Users available for deletion (non-admin users):")
    print("=" * 60)
    
    try:
        users = db_manager.get_all_bot_users()
        if not users:
            print("No users found in database.")
            return []
        
        admin_users = []
        regular_users = []
        
        for user in users:
            if user['is_admin']:
                admin_users.append(user)
            else:
                regular_users.append(user)
        
        print(f"👑 Admin users ({len(admin_users)}) - CANNOT BE DELETED:")
        for user in admin_users:
            print(f"  • User {user['bot_user_id']}: {user['telegram_chat_id']} (Admin)")
        
        print(f"\n👤 Regular users ({len(regular_users)}) - CAN BE DELETED:")
        if not regular_users:
            print("  No regular users found.")
        else:
            for user in regular_users:
                account = db_manager.get_mt_account_by_bot_user_id(user['bot_user_id'])
                if account:
                    print(f"  • User {user['bot_user_id']}: {user['telegram_chat_id']} (MT: {account['mt_account_number']})")
                else:
                    print(f"  • User {user['bot_user_id']}: {user['telegram_chat_id']} (No MT account)")
        
        return regular_users
        
    except Exception as e:
        print(f"❌ Error listing users: {e}")
        return []

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python delete_specific_user.py <telegram_chat_id>")
        print("  python delete_specific_user.py list")
        print("  python delete_specific_user.py <telegram_chat_id> --force")
        print()
        print("Examples:")
        print("  python delete_specific_user.py 987654321")
        print("  python delete_specific_user.py list")
        print("  python delete_specific_user.py 987654321 --force")
        return
    
    if sys.argv[1] == "list":
        list_users_for_deletion()
        return
    
    try:
        telegram_chat_id = int(sys.argv[1])
    except ValueError:
        print("❌ Invalid telegram_chat_id. Must be a number.")
        return
    
    # Check for force flag
    force = len(sys.argv) > 2 and sys.argv[2] == "--force"
    
    if force:
        print("🚨 FORCE MODE: No confirmation required")
        print("=" * 40)
    
    delete_specific_user(telegram_chat_id, confirm=not force)

if __name__ == "__main__":
    main()
