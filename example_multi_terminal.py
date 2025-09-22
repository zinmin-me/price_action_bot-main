"""
Example script demonstrating multi-terminal MT5 usage
"""

import time
import logging
from terminal_manager import TerminalManager, TerminalConfig
from mt5_connector import MT5Connector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_basic_usage():
    """Basic example of using multiple terminals"""
    print("=== Basic Multi-Terminal Example ===")
    
    # Initialize terminal manager
    manager = TerminalManager()
    
    # Create terminal configurations
    demo_config = TerminalConfig(
        name="demo_account",
        terminal_path="C:\\Program Files\\MetaTrader 5\\terminal64.exe",
        login=12345,
        password="demo_password",
        server="DemoServer",
        symbol="EURUSD",
        timeframe="M15",
        auto_start=True
    )
    
    live_config = TerminalConfig(
        name="live_account",
        terminal_path="C:\\Program Files\\MetaTrader 5\\terminal64.exe",
        login=67890,
        password="live_password",
        server="LiveServer",
        symbol="GBPUSD",
        timeframe="M15",
        auto_start=False
    )
    
    # Add terminals to manager
    manager.add_terminal(demo_config)
    manager.add_terminal(live_config)
    
    # Start terminals
    print("Starting terminals...")
    manager.start_terminal("demo_account")
    manager.start_terminal("live_account")
    
    # Wait for terminals to initialize
    time.sleep(5)
    
    # Check status
    status = manager.get_terminal_status()
    print(f"Terminal Status: {status}")
    
    # Create connectors
    demo_connector = MT5Connector(
        login=12345,
        password="demo_password",
        server="DemoServer",
        terminal_name="demo_account",
        dedicated_terminal=True
    )
    
    live_connector = MT5Connector(
        login=67890,
        password="live_password",
        server="LiveServer",
        terminal_name="live_account",
        dedicated_terminal=True
    )
    
    # Connect to accounts
    print("Connecting to demo account...")
    if demo_connector.connect():
        print("✅ Demo account connected successfully")
        account_info = demo_connector.get_account_info()
        if account_info:
            print(f"   Balance: {account_info['balance']} {account_info['currency']}")
    else:
        print("❌ Demo account connection failed")
    
    print("Connecting to live account...")
    if live_connector.connect():
        print("✅ Live account connected successfully")
        account_info = live_connector.get_account_info()
        if account_info:
            print(f"   Balance: {account_info['balance']} {account_info['currency']}")
    else:
        print("❌ Live account connection failed")
    
    # Get terminal information
    demo_terminal_info = demo_connector.get_terminal_info()
    live_terminal_info = live_connector.get_terminal_info()
    
    print(f"\nDemo Terminal Info: {demo_terminal_info}")
    print(f"Live Terminal Info: {live_terminal_info}")
    
    # Cleanup
    print("\nCleaning up...")
    demo_connector.disconnect()
    live_connector.disconnect()
    manager.stop_all_terminals()
    manager.cleanup()

def example_config_file_usage():
    """Example using configuration file"""
    print("\n=== Configuration File Example ===")
    
    # Initialize terminal manager
    manager = TerminalManager()
    
    # Load configuration from file
    config_file = "terminals_config.json"
    if manager.create_terminal_configs_from_file(config_file):
        print(f"✅ Loaded configuration from {config_file}")
        
        # Start all terminals
        results = manager.start_all_terminals()
        print(f"Start results: {results}")
        
        # Start monitoring
        manager.start_monitoring()
        print("✅ Monitoring started")
        
        # Wait a bit
        time.sleep(10)
        
        # Check status
        status = manager.get_terminal_status()
        print(f"All terminals status: {status}")
        
        # Cleanup
        manager.stop_monitoring()
        manager.stop_all_terminals()
        manager.cleanup()
    else:
        print(f"❌ Failed to load configuration from {config_file}")

def example_terminal_management():
    """Example of terminal management operations"""
    print("\n=== Terminal Management Example ===")
    
    manager = TerminalManager()
    
    # Find available terminals
    available_terminals = manager.find_available_terminals()
    print(f"Available MT5 installations: {available_terminals}")
    
    if not available_terminals:
        print("❌ No MT5 installations found!")
        return
    
    # Create a test terminal
    test_config = TerminalConfig(
        name="test_terminal",
        terminal_path=available_terminals[0],
        login=99999,
        password="test_password",
        server="TestServer",
        auto_start=False
    )
    
    # Add and manage terminal
    manager.add_terminal(test_config)
    print("✅ Test terminal added")
    
    # Start terminal
    if manager.start_terminal("test_terminal"):
        print("✅ Test terminal started")
        
        # Check status
        status = manager.get_terminal_status("test_terminal")
        print(f"Terminal status: {status}")
        
        # Restart terminal
        if manager.restart_terminal("test_terminal"):
            print("✅ Test terminal restarted")
        
        # Stop terminal
        if manager.stop_terminal("test_terminal"):
            print("✅ Test terminal stopped")
    
    # Remove terminal
    manager.remove_terminal("test_terminal")
    print("✅ Test terminal removed")

def main():
    """Main example function"""
    print("Multi-Terminal MT5 Examples")
    print("=" * 40)
    
    try:
        # Run examples
        example_basic_usage()
        example_config_file_usage()
        example_terminal_management()
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"❌ Example failed: {e}")

if __name__ == "__main__":
    main()
