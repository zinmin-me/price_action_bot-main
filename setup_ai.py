#!/usr/bin/env python3
"""
AI Trading Bot Setup Script
Installs dependencies and sets up the AI system
"""

import subprocess
import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and handle errors"""
    try:
        logger.info(f"🔄 {description}...")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("❌ Python 3.8 or higher is required")
        return False
    
    logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    logger.info("📦 Installing AI Trading Bot dependencies...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "ai/models",
        "ai/data",
        "logs"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"✅ Created directory: {directory}")
        except Exception as e:
            logger.error(f"❌ Failed to create directory {directory}: {e}")
            return False
    
    return True

def test_imports():
    """Test if all required modules can be imported"""
    logger.info("🧪 Testing imports...")
    
    test_modules = [
        "numpy",
        "pandas", 
        "sklearn",
        "xgboost",
        "lightgbm",
        "tensorflow",
        "telegram",
        "MetaTrader5"
    ]
    
    failed_imports = []
    
    for module in test_modules:
        try:
            __import__(module)
            logger.info(f"✅ {module} imported successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        logger.error(f"❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    return True

def test_ai_components():
    """Test AI components"""
    logger.info("🤖 Testing AI components...")
    
    try:
        from ai import AIStrategy, AutoTrainer, DataProcessor, ModelManager
        logger.info("✅ AI components imported successfully")
        
        # Test basic functionality
        data_processor = DataProcessor()
        model_manager = ModelManager()
        
        logger.info("✅ AI components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ AI components test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🤖 AI Trading Bot Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        logger.error("❌ Failed to create directories")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        logger.error("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        logger.error("❌ Some modules failed to import")
        logger.info("💡 Try running: pip install -r requirements.txt")
        sys.exit(1)
    
    # Test AI components
    if not test_ai_components():
        logger.error("❌ AI components test failed")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Configure your MT5 credentials in config.py")
    print("2. Set up your Telegram bot token")
    print("3. Run: python main.py")
    print("4. Use /ai_train in Telegram to train AI models")
    print("5. Use /ai_status to check AI status")
    
    print("\n🚀 Your AI Trading Bot is ready!")

if __name__ == "__main__":
    main()
