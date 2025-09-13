# Database System for Price Action Trading Bot

This document describes the database system implemented for the Price Action Trading Bot to manage user access and MT5 account associations.

## Overview

The database system provides:
- **User Authorization**: Only authorized Telegram users can access the bot
- **Account Management**: One-to-one relationship between Telegram users and MT5 accounts
- **Session Tracking**: Track which users are logged into which MT5 accounts
- **Admin Tools**: Commands to manage users and view statistics

## Database Schema

### bot_user Table
Stores authorized Telegram users who can access the bot.

| Column | Type | Description |
|--------|------|-------------|
| bot_user_id | INTEGER PRIMARY KEY | Auto-incrementing unique identifier |
| telegram_chat_id | INTEGER UNIQUE | Telegram chat ID (must be unique) |
| created_at | TIMESTAMP | When the user was added |
| updated_at | TIMESTAMP | Last update time |

### mt_account Table
Stores MT5 account associations for bot users.

| Column | Type | Description |
|--------|------|-------------|
| mt_account_id | INTEGER PRIMARY KEY | Auto-incrementing unique identifier |
| bot_user_id | INTEGER | Foreign key to bot_user table |
| mt_account_number | INTEGER | MT5 account number |
| created_at | TIMESTAMP | When the account was associated |
| updated_at | TIMESTAMP | Last update time |

## Key Features

### 1. User Authorization
- Only Telegram chat IDs in the `bot_user` table can use the bot
- Unauthorized users receive an access denied message
- User data must be added manually by administrators

### 2. One-to-One Account Relationship
- Each Telegram user can only be associated with one MT5 account at a time
- If a user wants to switch accounts, they must logout first
- The system prevents multiple account associations per user

### 3. Login Process
1. Check if Telegram chat ID is authorized (exists in `bot_user` table)
2. If authorized, allow login process to proceed
3. On successful MT5 login, store the association in `mt_account` table
4. Update existing association if user logs in with a different account

### 4. Logout Process
1. Disconnect from MT5
2. Remove the association from `mt_account` table
3. Clear the session from memory

## Usage

### For Administrators

#### Adding Users
```bash
# Using the management script
python manage_users.py add 123456789

# Using Telegram bot command (if authorized)
/add_user 123456789
```

#### Viewing Statistics
```bash
# Using the management script
python manage_users.py stats

# Using Telegram bot command
/db_stats
```

#### Listing Users
```bash
# Using the management script
python manage_users.py list

# Using Telegram bot command
/list_users
```

### For Users

#### First Time Setup
1. Administrator adds your Telegram chat ID to the database
2. You can now use `/start` to begin using the bot
3. Use `/login` to connect your MT5 account

#### Daily Usage
1. Use `/login` to connect to your MT5 account
2. Use bot commands and features as normal
3. Use `/logout` when done to disconnect

## Database File

The database is stored as `trading_bot.db` in the project root directory. This is a SQLite database that is created automatically when the bot starts.

## Security Considerations

1. **Manual User Addition**: Users must be manually added by administrators, preventing unauthorized access
2. **Account Isolation**: Each user can only access their own MT5 account
3. **Session Management**: Proper cleanup of sessions and database entries on logout
4. **Access Control**: Authorization checks on all major bot commands

## Admin Commands

The following Telegram commands are available for administrators:

- `/add_user <telegram_chat_id>` - Add a new authorized user
- `/list_users` - List all authorized users and their MT account status
- `/db_stats` - Show database statistics

## Management Script

The `manage_users.py` script provides command-line access to user management:

```bash
# Add a user
python manage_users.py add 123456789

# List all users
python manage_users.py list

# Show database statistics
python manage_users.py stats

# List MT accounts
python manage_users.py mt_accounts

# Remove a user (removes MT account association)
python manage_users.py remove 123456789
```

## Error Handling

The system includes comprehensive error handling:
- Database connection errors
- Invalid user operations
- MT5 connection failures
- Authorization failures

All errors are logged and appropriate messages are sent to users.

## Future Enhancements

Potential future improvements:
1. User roles and permissions
2. Account usage tracking and limits
3. Automatic user cleanup based on inactivity
4. Backup and restore functionality
5. Multi-account support per user (with proper switching)
6. User activity logging and analytics
