#!/usr/bin/env python3
"""
Setup script to create the first admin user
This should be run once to create the initial administrator
"""

import sys
from database import db_manager

def setup_first_admin(telegram_chat_id):
    """Setup the first admin user"""
    print("🔧 Setting up first admin user...")
    print("=" * 40)
    
    try:
        # Check if any users exist
        users = db_manager.get_all_bot_users()
        
        if users:
            print("⚠️  Users already exist in the database.")
            print("📋 Current users:")
            for user in users:
                role = "👑 Admin" if user['is_admin'] else "👤 User"
                print(f"  • User {user['bot_user_id']}: {user['telegram_chat_id']} ({role})")
            
            response = input("\nDo you want to add another admin? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("❌ Setup cancelled.")
                return False
        
        # Add the admin user
        print(f"\n➕ Adding admin user with Telegram Chat ID: {telegram_chat_id}")
        bot_user_id = db_manager.add_bot_user(telegram_chat_id, is_admin=True)
        
        if bot_user_id:
            print(f"✅ Admin user created successfully!")
            print(f"   Bot User ID: {bot_user_id}")
            print(f"   Telegram Chat ID: {telegram_chat_id}")
            print(f"   Role: 👑 Admin")
            print(f"\n🎉 Setup complete! This user can now:")
            print(f"   • Use all bot features")
            print(f"   • Add other users with /add_user")
            print(f"   • View user lists with /list_users")
            print(f"   • Access database stats with /db_stats")
            return True
        else:
            print("❌ Failed to create admin user. User might already exist.")
            return False
            
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python setup_admin.py <telegram_chat_id>")
        print("\nExample:")
        print("  python setup_admin.py 123456789")
        print("\nTo find your Telegram Chat ID:")
        print("  1. Start a chat with your bot")
        print("  2. Send any message")
        print("  3. Check the bot logs for your chat ID")
        sys.exit(1)
    
    try:
        telegram_chat_id = int(sys.argv[1])
    except ValueError:
        print("❌ Invalid telegram_chat_id. Must be a number.")
        sys.exit(1)
    
    success = setup_first_admin(telegram_chat_id)
    
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    main()
