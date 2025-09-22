"""
Setup script for multi-terminal MT5 configuration
Helps users configure multiple MT5 terminals for different accounts
"""

import os
import json
import sys
from pathlib import Path
from terminal_manager import TerminalManager, TerminalConfig

def find_mt5_installations():
    """Find all available MT5 installations"""
    manager = TerminalManager()
    return manager.find_available_terminals()

def create_terminal_config():
    """Interactive terminal configuration creation"""
    print("=== Multi-Terminal MT5 Setup ===")
    print("This script will help you configure multiple MT5 terminals for different accounts.\n")
    
    # Find available terminals
    print("Searching for MT5 installations...")
    available_terminals = find_mt5_installations()
    
    if not available_terminals:
        print("❌ No MT5 installations found!")
        print("Please install MetaTrader 5 first.")
        return False
    
    print(f"✅ Found {len(available_terminals)} MT5 installation(s):")
    for i, path in enumerate(available_terminals, 1):
        print(f"  {i}. {path}")
    
    terminals = []
    
    while True:
        print(f"\n--- Terminal Configuration {len(terminals) + 1} ---")
        
        # Terminal name
        name = input("Enter terminal name (e.g., 'demo_account_1'): ").strip()
        if not name:
            print("❌ Terminal name is required!")
            continue
        
        # Check for duplicate names
        if any(t['name'] == name for t in terminals):
            print("❌ Terminal name already exists!")
            continue
        
        # Terminal path
        if len(available_terminals) == 1:
            terminal_path = available_terminals[0]
            print(f"Using terminal: {terminal_path}")
        else:
            print("Select terminal path:")
            for i, path in enumerate(available_terminals, 1):
                print(f"  {i}. {path}")
            
            try:
                choice = int(input("Enter choice (1-{}): ".format(len(available_terminals))))
                if 1 <= choice <= len(available_terminals):
                    terminal_path = available_terminals[choice - 1]
                else:
                    print("❌ Invalid choice!")
                    continue
            except ValueError:
                print("❌ Please enter a valid number!")
                continue
        
        # Account credentials
        try:
            login = int(input("Enter account login number: "))
        except ValueError:
            print("❌ Login must be a number!")
            continue
        
        password = input("Enter account password: ").strip()
        if not password:
            print("❌ Password is required!")
            continue
        
        server = input("Enter server name: ").strip()
        if not server:
            print("❌ Server name is required!")
            continue
        
        # Trading settings
        symbol = input("Enter trading symbol (default: EURUSD): ").strip() or "EURUSD"
        timeframe = input("Enter timeframe (default: M15): ").strip() or "M15"
        
        # Auto-start setting
        auto_start_input = input("Auto-start this terminal? (y/n, default: y): ").strip().lower()
        auto_start = auto_start_input != 'n'
        
        # Create terminal config
        terminal_config = {
            "name": name,
            "terminal_path": terminal_path,
            "login": login,
            "password": password,
            "server": server,
            "symbol": symbol,
            "timeframe": timeframe,
            "auto_start": auto_start,
            "port_offset": 0
        }
        
        terminals.append(terminal_config)
        
        print(f"✅ Terminal '{name}' configured successfully!")
        
        # Ask if user wants to add more terminals
        add_more = input("\nAdd another terminal? (y/n): ").strip().lower()
        if add_more != 'y':
            break
    
    if not terminals:
        print("❌ No terminals configured!")
        return False
    
    # Save configuration
    config = {
        "terminals": terminals,
        "settings": {
            "monitoring_enabled": True,
            "auto_restart_failed_terminals": True,
            "max_restart_attempts": 3,
            "restart_delay_seconds": 30,
            "health_check_interval_seconds": 60
        }
    }
    
    config_file = "terminals_config.json"
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Configuration saved to: {config_file}")
        print(f"✅ Configured {len(terminals)} terminal(s)")
        
        # Show summary
        print("\n--- Configuration Summary ---")
        for terminal in terminals:
            print(f"• {terminal['name']}: {terminal['login']} @ {terminal['server']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return False

def test_terminal_configuration():
    """Test the terminal configuration"""
    print("\n=== Testing Terminal Configuration ===")
    
    config_file = "terminals_config.json"
    if not os.path.exists(config_file):
        print(f"❌ Configuration file not found: {config_file}")
        return False
    
    try:
        # Load configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Initialize terminal manager
        manager = TerminalManager()
        
        # Load terminals
        if not manager.create_terminal_configs_from_file(config_file):
            print("❌ Failed to load terminal configurations!")
            return False
        
        print(f"✅ Loaded {len(config['terminals'])} terminal configuration(s)")
        
        # Test each terminal
        for terminal_config in config['terminals']:
            name = terminal_config['name']
            print(f"\nTesting terminal: {name}")
            
            # Start terminal
            if manager.start_terminal(name):
                print(f"  ✅ Terminal '{name}' started successfully")
                
                # Get status
                status = manager.get_terminal_status(name)
                if 'error' not in status:
                    print(f"  ✅ Status: {status['status']['status']}")
                    if status['status']['process_id']:
                        print(f"  ✅ Process ID: {status['status']['process_id']}")
                else:
                    print(f"  ❌ Status error: {status['error']}")
            else:
                print(f"  ❌ Failed to start terminal '{name}'")
        
        # Start monitoring
        manager.start_monitoring()
        print("\n✅ Terminal monitoring started")
        
        print("\n--- Terminal Status ---")
        all_status = manager.get_terminal_status()
        for name, info in all_status['terminals'].items():
            status = info['status']['status']
            process_id = info['status'].get('process_id', 'N/A')
            print(f"• {name}: {status} (PID: {process_id})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing configuration: {e}")
        return False

def main():
    """Main setup function"""
    print("MT5 Multi-Terminal Setup")
    print("=" * 30)
    
    while True:
        print("\nOptions:")
        print("1. Create new terminal configuration")
        print("2. Test existing configuration")
        print("3. View current configuration")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            create_terminal_config()
        elif choice == '2':
            test_terminal_configuration()
        elif choice == '3':
            view_configuration()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("❌ Invalid choice! Please enter 1-4.")

def view_configuration():
    """View current terminal configuration"""
    config_file = "terminals_config.json"
    
    if not os.path.exists(config_file):
        print(f"❌ Configuration file not found: {config_file}")
        return
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print("\n--- Current Configuration ---")
        print(f"Terminals configured: {len(config['terminals'])}")
        
        for i, terminal in enumerate(config['terminals'], 1):
            print(f"\n{i}. {terminal['name']}")
            print(f"   Path: {terminal['terminal_path']}")
            print(f"   Account: {terminal['login']} @ {terminal['server']}")
            print(f"   Symbol: {terminal['symbol']} ({terminal['timeframe']})")
            print(f"   Auto-start: {terminal['auto_start']}")
        
        # Show settings
        settings = config.get('settings', {})
        print(f"\n--- Settings ---")
        print(f"Monitoring: {settings.get('monitoring_enabled', False)}")
        print(f"Auto-restart: {settings.get('auto_restart_failed_terminals', False)}")
        print(f"Max restart attempts: {settings.get('max_restart_attempts', 3)}")
        
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")

if __name__ == "__main__":
    main()
