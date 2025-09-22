"""
Multi-Terminal Manager for MT5 Trading Bot
Handles multiple MT5 terminal instances for different accounts
"""

import os
import time
import subprocess
import threading
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    _PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class TerminalConfig:
    """Configuration for a single MT5 terminal instance"""
    name: str
    terminal_path: str
    login: int
    password: str
    server: str
    symbol: str = "EURUSD"
    timeframe: str = "M15"
    auto_start: bool = True
    port_offset: int = 0  # For different data ports if needed

class TerminalManager:
    """
    Manages multiple MT5 terminal instances for different accounts.
    Each terminal runs independently with its own account.
    """
    
    def __init__(self):
        self.terminals: Dict[str, TerminalConfig] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self.terminal_status: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._monitoring_thread = None
        self._monitoring_active = False
        
    def add_terminal(self, config: TerminalConfig) -> bool:
        """
        Add a new terminal configuration
        
        Args:
            config: TerminalConfig object with terminal settings
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        with self._lock:
            if config.name in self.terminals:
                logger.warning(f"Terminal '{config.name}' already exists")
                return False
                
            # Validate terminal path
            if not os.path.exists(config.terminal_path):
                logger.error(f"Terminal path does not exist: {config.terminal_path}")
                return False
                
            self.terminals[config.name] = config
            self.terminal_status[config.name] = {
                'status': 'configured',
                'last_check': datetime.now(),
                'process_id': None,
                'account_connected': False,
                'error_count': 0
            }
            
            logger.info(f"Added terminal configuration: {config.name}")
            
            # Auto-start if enabled
            if config.auto_start:
                return self.start_terminal(config.name)
                
            return True
    
    def remove_terminal(self, name: str) -> bool:
        """
        Remove a terminal configuration and stop its process
        
        Args:
            name: Terminal name to remove
            
        Returns:
            bool: True if removed successfully, False otherwise
        """
        with self._lock:
            if name not in self.terminals:
                logger.warning(f"Terminal '{name}' not found")
                return False
                
            # Stop the terminal if running
            self.stop_terminal(name)
            
            # Remove from configuration
            del self.terminals[name]
            del self.terminal_status[name]
            
            logger.info(f"Removed terminal: {name}")
            return True
    
    def start_terminal(self, name: str) -> bool:
        """
        Start a specific terminal instance
        
        Args:
            name: Terminal name to start
            
        Returns:
            bool: True if started successfully, False otherwise
        """
        with self._lock:
            if name not in self.terminals:
                logger.error(f"Terminal '{name}' not configured")
                return False
                
            config = self.terminals[name]
            
            # Check if already running
            if name in self.processes and self.processes[name].poll() is None:
                logger.info(f"Terminal '{name}' is already running")
                return True
                
            try:
                # Resolve a writable launch path: if in Program Files, copy to user-local dir and run from there
                launch_path = config.terminal_path
                try:
                    path_l = str(launch_path).lower()
                    pf_markers = [
                        os.path.join("C:\\", "Program Files").lower(),
                        os.path.join("C:\\", "Program Files (x86)").lower(),
                    ]
                    is_program_files = any(path_l.startswith(m) for m in pf_markers)
                except Exception:
                    is_program_files = False

                if is_program_files:
                    try:
                        # Build per-terminal user dir
                        local_base = os.path.join(str(Path.home()), "AppData", "Local", "PriceActionBot", "mt5", name)
                        os.makedirs(local_base, exist_ok=True)
                        # Determine filename (terminal64.exe or terminal.exe)
                        exe_name = os.path.basename(launch_path)
                        user_exe = os.path.join(local_base, exe_name)
                        # Copy binary if missing or outdated
                        try:
                            import shutil
                            if not os.path.exists(user_exe) or (
                                os.path.getmtime(user_exe) < os.path.getmtime(launch_path)
                            ):
                                shutil.copy2(launch_path, user_exe)
                                logger.info(f"Copied MT5 executable to writable path: {user_exe}")
                            launch_path = user_exe
                            # Persist updated path back into config for future starts
                            config.terminal_path = launch_path
                            logger.info(f"Using writable MT5 path for '{name}': {launch_path}")
                        except Exception as e:
                            logger.warning(f"Failed to copy MT5 executable to user dir, attempting to run original: {e}")
                    except Exception as e:
                        logger.warning(f"Writable launch path setup failed: {e}")

                # Start the terminal process
                logger.info(f"Starting terminal '{name}': {launch_path}")
                
                # Use per-terminal working directory (prefer user-local dir if used)
                work_dir = os.path.dirname(launch_path)
                
                # Launch MT5 in portable mode so it stores data next to the executable (now writable)
                process = subprocess.Popen(
                    [launch_path, "/portable"],
                    cwd=work_dir,
                    shell=False,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
                )
                
                self.processes[name] = process
                self.terminal_status[name].update({
                    'status': 'starting',
                    'process_id': process.pid,
                    'start_time': datetime.now(),
                    'error_count': 0
                })
                
                # Wait a moment for terminal to initialize (MT5 can take a few seconds)
                time.sleep(8)
                
                # Check if process is still running
                if process.poll() is None:
                    self.terminal_status[name]['status'] = 'running'
                    logger.info(f"Terminal '{name}' started successfully (PID: {process.pid})")
                    return True
                else:
                    self.terminal_status[name]['status'] = 'failed'
                    logger.error(f"Terminal '{name}' failed to start")
                    return False
                    
            except Exception as e:
                logger.error(f"Error starting terminal '{name}': {e}")
                self.terminal_status[name]['status'] = 'error'
                self.terminal_status[name]['error_count'] += 1
                return False
    
    def stop_terminal(self, name: str) -> bool:
        """
        Stop a specific terminal instance
        
        Args:
            name: Terminal name to stop
            
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        with self._lock:
            if name not in self.processes:
                logger.warning(f"Terminal '{name}' is not running")
                return True
                
            process = self.processes[name]
            
            try:
                if process.poll() is None:  # Process is still running
                    logger.info(f"Stopping terminal '{name}' (PID: {process.pid})")
                    
                    # Attempt polite close via WM_CLOSE on Windows
                    if os.name == 'nt':
                        try:
                            import ctypes
                            ctypes.windll.user32.PostMessageW(process.pid, 0x0010, 0, 0)  # WM_CLOSE
                            time.sleep(2)
                        except Exception:
                            pass
                    
                    # Try graceful termination
                    process.terminate()
                    
                    # Wait for graceful shutdown
                    try:
                        process.wait(timeout=12)
                    except subprocess.TimeoutExpired:
                        # Force kill if graceful shutdown fails
                        logger.warning(f"Force killing terminal '{name}'")
                        process.kill()
                        process.wait()
                
                # Clean up
                del self.processes[name]
                self.terminal_status[name]['status'] = 'stopped'
                self.terminal_status[name]['process_id'] = None
                
                logger.info(f"Terminal '{name}' stopped successfully")
                return True
                
            except Exception as e:
                logger.error(f"Error stopping terminal '{name}': {e}")
                return False
    
    def restart_terminal(self, name: str) -> bool:
        """
        Restart a specific terminal instance
        
        Args:
            name: Terminal name to restart
            
        Returns:
            bool: True if restarted successfully, False otherwise
        """
        logger.info(f"Restarting terminal '{name}'")
        self.stop_terminal(name)
        time.sleep(2)  # Brief pause between stop and start
        return self.start_terminal(name)
    
    def get_terminal_status(self, name: str = None) -> Dict:
        """
        Get status of terminal(s)
        
        Args:
            name: Specific terminal name, or None for all terminals
            
        Returns:
            Dict: Terminal status information
        """
        with self._lock:
            if name:
                if name not in self.terminals:
                    return {'error': f"Terminal '{name}' not found"}
                return {
                    'name': name,
                    'config': self.terminals[name],
                    'status': self.terminal_status[name]
                }
            else:
                return {
                    'terminals': {
                        name: {
                            'config': config,
                            'status': self.terminal_status[name]
                        }
                        for name, config in self.terminals.items()
                    },
                    'total_terminals': len(self.terminals),
                    'running_terminals': len([p for p in self.processes.values() if p.poll() is None])
                }
    
    def start_all_terminals(self) -> Dict[str, bool]:
        """
        Start all configured terminals
        
        Returns:
            Dict[str, bool]: Results for each terminal
        """
        results = {}
        for name in self.terminals:
            results[name] = self.start_terminal(name)
        return results
    
    def stop_all_terminals(self) -> Dict[str, bool]:
        """
        Stop all running terminals
        
        Returns:
            Dict[str, bool]: Results for each terminal
        """
        results = {}
        for name in list(self.processes.keys()):
            results[name] = self.stop_terminal(name)
        return results
    
    def start_monitoring(self):
        """Start background monitoring of terminal processes"""
        if self._monitoring_active:
            return
            
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitor_terminals, daemon=True)
        self._monitoring_thread.start()
        logger.info("Terminal monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Terminal monitoring stopped")
    
    def _monitor_terminals(self):
        """Background monitoring loop for terminal processes"""
        while self._monitoring_active:
            try:
                with self._lock:
                    for name, process in list(self.processes.items()):
                        if process.poll() is not None:  # Process has terminated
                            logger.warning(f"Terminal '{name}' process terminated unexpectedly")
                            self.terminal_status[name]['status'] = 'crashed'
                            self.terminal_status[name]['error_count'] += 1
                            
                            # Auto-restart if configured
                            config = self.terminals.get(name)
                            if config and config.auto_start and self.terminal_status[name]['error_count'] < 3:
                                logger.info(f"Auto-restarting terminal '{name}'")
                                time.sleep(5)  # Brief delay before restart
                                self.start_terminal(name)
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in terminal monitoring: {e}")
                time.sleep(30)  # Longer delay on error
    
    def find_available_terminals(self) -> List[str]:
        """
        Find all available MT5 terminal installations
        
        Returns:
            List[str]: List of available terminal paths
        """
        possible_paths = [
            r"C:\Program Files\MetaTrader 5\terminal64.exe",
            r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
            r"C:\Program Files\MetaTrader 5\terminal.exe",
            r"C:\Program Files (x86)\MetaTrader 5\terminal.exe"
        ]
        
        # Add user-specific paths
        username = os.getenv('USERNAME', '')
        if username:
            user_paths = [
                rf"C:\Users\{username}\AppData\Roaming\MetaQuotes\Terminal\*\terminal64.exe",
                rf"C:\Users\{username}\AppData\Roaming\MetaQuotes\Terminal\*\terminal.exe"
            ]
            possible_paths.extend(user_paths)
        
        valid_paths = []
        for path in possible_paths:
            if '*' in path:
                # Handle wildcard paths
                import glob
                matches = glob.glob(path)
                valid_paths.extend(matches)
            elif os.path.exists(path):
                valid_paths.append(path)
        
        return valid_paths
    
    def create_terminal_configs_from_file(self, config_file: str) -> bool:
        """
        Load terminal configurations from a JSON file
        
        Args:
            config_file: Path to JSON configuration file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            import json
            
            if not os.path.exists(config_file):
                logger.error(f"Configuration file not found: {config_file}")
                return False
            
            with open(config_file, 'r') as f:
                configs = json.load(f)
            
            for config_data in configs.get('terminals', []):
                config = TerminalConfig(**config_data)
                self.add_terminal(config)
            
            logger.info(f"Loaded {len(configs.get('terminals', []))} terminal configurations")
            return True
            
        except Exception as e:
            logger.error(f"Error loading terminal configurations: {e}")
            return False
    
    def save_terminal_configs_to_file(self, config_file: str) -> bool:
        """
        Save current terminal configurations to a JSON file
        
        Args:
            config_file: Path to save configuration file
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            import json
            
            configs = {
                'terminals': [
                    {
                        'name': config.name,
                        'terminal_path': config.terminal_path,
                        'login': config.login,
                        'password': config.password,
                        'server': config.server,
                        'symbol': config.symbol,
                        'timeframe': config.timeframe,
                        'auto_start': config.auto_start,
                        'port_offset': config.port_offset
                    }
                    for config in self.terminals.values()
                ]
            }
            
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(configs, f, indent=2)
            
            logger.info(f"Saved terminal configurations to: {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving terminal configurations: {e}")
            return False
    
    def get_terminal_for_account(self, login: int) -> Optional[str]:
        """
        Find which terminal is configured for a specific account
        
        Args:
            login: Account login number
            
        Returns:
            Optional[str]: Terminal name if found, None otherwise
        """
        for name, config in self.terminals.items():
            if config.login == login:
                return name
        return None
    
    def cleanup(self):
        """Clean up all resources"""
        logger.info("Cleaning up terminal manager...")
        self.stop_monitoring()
        self.stop_all_terminals()
        logger.info("Terminal manager cleanup complete")

# Global terminal manager instance
terminal_manager = TerminalManager()
