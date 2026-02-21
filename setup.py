#! /usr/bin/env python3
"""
Setup script for STL-10 Classification Project
Run this to check dependencies and setup environment
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n🔧 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name} (Count: {gpu_count})")
        else:
            print("⚠️  No GPU detected - will use CPU (slower training)")
    except ImportError:
        print("⚠️  PyTorch not installed yet")

def create_directories():
    """Create necessary directories"""
    dirs = ['models', 'results']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"📁 Created directory: {dir_name}")

def verify_wandb_setup():
    """Check Wandb configuration"""
    from config import Config
    config = Config()
    if config.WANDB_API_KEY and config.WANDB_API_KEY.startswith("wandb_"):
        print("✅ Wandb API key configured")
        print(f"📊 Project: {config.WANDB_PROJECT}")
    else:
        print("⚠️  Wandb API key not properly configured in config.py")

def main():
    print("STL-10 Classification Project Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install dependencies
    if not install_dependencies():
        return
    
    # Check GPU
    check_gpu()
    
    # Create directories
    create_directories()
    
    # Verify Wandb
    try:
        verify_wandb_setup()
    except ImportError as e:
        print(f"⚠️  Config verification failed: {e}")
    
    print("\n" + "="*50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run: python run_experiment.py")
    print("2. Monitor training on Wandb dashboard")
    print("3. Check results in 'results' folder")
    print("4. Upload to GitHub when complete")
    print("="*50)

if __name__ == "__main__":
    main()