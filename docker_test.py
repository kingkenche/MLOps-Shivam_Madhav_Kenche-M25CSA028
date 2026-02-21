#!/usr/bin/env python3
"""
Docker environment test script for STL-10 Classification
Verifies that the Docker environment is properly configured
"""

import sys
import os
import torch
import subprocess
from pathlib import Path

def print_header(title):
    print("\n" + "=" * 60)
    print(f"🐳 {title}")
    print("=" * 60)

def print_success(msg):
    print(f"✅ {msg}")

def print_warning(msg):
    print(f"⚠️  {msg}")

def print_error(msg):
    print(f"❌ {msg}")

def test_docker_environment():
    """Test basic Docker environment setup"""
    print_header("Docker Environment Test")
    
    # Check if running in Docker
    is_docker = os.path.exists('/.dockerenv') or os.getenv('DOCKER_CONTAINER') == 'true'
    if is_docker:
        print_success("Running inside Docker container")
    else:
        print_warning("Not running in Docker container")
    
    # Check Python version
    python_version = sys.version
    print(f"Python version: {python_version}")
    
    # Check working directory
    cwd = os.getcwd()
    print(f"Current directory: {cwd}")
    
    # Check if required directories exist
    required_dirs = ['models', 'results', 'wandb']
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print_success(f"Directory exists: {dir_name}")
        else:
            os.makedirs(dir_name, exist_ok=True)
            print_success(f"Created directory: {dir_name}")

def test_gpu_availability():
    """Test GPU and CUDA availability"""
    print_header("GPU and CUDA Test")
    
    print(f"PyTorch version: {torch.__version__}")
    
    # Test CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print_success(f"CUDA is available")
        device_count = torch.cuda.device_count()
        print(f"CUDA device count: {device_count}")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print_success(f"Device {i}: {device_name} ({memory_gb:.1f}GB)")
            
        # Test tensor operations on GPU
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.matmul(x, x)
            print_success("GPU tensor operations working")
        except Exception as e:
            print_error(f"GPU tensor operations failed: {e}")
    else:
        print_warning("CUDA not available, will use CPU")
    
    return cuda_available

def test_dependencies():
    """Test required Python packages"""
    print_header("Dependencies Test")
    
    required_packages = [
        'torch', 'torchvision', 'transformers', 'datasets',
        'wandb', 'huggingface_hub', 'matplotlib', 'seaborn',
        'sklearn', 'numpy', 'PIL'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print_success(f"Package available: {package}")
        except ImportError:
            print_error(f"Package missing: {package}")

def test_wandb_connection():
    """Test Wandb API configuration"""
    print_header("Wandb Connection Test")
    
    wandb_key = os.getenv('WANDB_API_KEY')
    if wandb_key:
        print_success("Wandb API key found in environment")
        # Truncate key for security
        truncated_key = wandb_key[:10] + "..." + wandb_key[-10:] if len(wandb_key) > 20 else wandb_key
        print(f"API Key: {truncated_key}")
    else:
        print_warning("Wandb API key not found in environment")
    
    try:
        import wandb
        # Test wandb initialization (without actually logging)
        print_success("Wandb package imported successfully")
    except Exception as e:
        print_error(f"Wandb test failed: {e}")

def test_huggingface():
    """Test HuggingFace Hub connection"""
    print_header("HuggingFace Hub Test")
    
    try:
        from datasets import list_datasets
        from huggingface_hub import HfApi
        
        print_success("HuggingFace packages imported successfully")
        
        # Test dataset availability (without downloading)
        print("Testing dataset access...")
        api = HfApi()
        try:
            dataset_info = api.dataset_info("Chiranjeev007/STL-10_Subset")
            print_success("STL-10 dataset found on HuggingFace Hub")
        except Exception as e:
            print_warning(f"Could not verify dataset: {e}")
            
    except Exception as e:
        print_error(f"HuggingFace test failed: {e}")

def test_file_permissions():
    """Test file system permissions"""
    print_header("File Permissions Test")
    
    try:
        # Test write permissions
        test_file = "docker_test.tmp"
        with open(test_file, 'w') as f:
            f.write("Docker test")
        
        if os.path.exists(test_file):
            print_success("File write permissions OK")
            os.remove(test_file)
        else:
            print_error("File write failed")
            
    except Exception as e:
        print_error(f"File permission test failed: {e}")

def test_memory():
    """Test available memory"""
    print_header("Memory Test")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        print(f"Total memory: {memory_gb:.1f} GB")
        print(f"Available memory: {available_gb:.1f} GB")
        
        if available_gb > 2.0:
            print_success("Sufficient memory available")
        else:
            print_warning("Low memory - consider increasing container memory limit")
            
    except ImportError:
        print_warning("psutil not available, skipping memory check")
    except Exception as e:
        print_warning(f"Memory check failed: {e}")

def run_quick_training_test():
    """Run a quick training test with minimal data"""
    print_header("Quick Training Test")
    
    try:
        from docker_config import DockerConfig
        DockerConfig.print_config()
        print_success("Docker configuration loaded successfully")
        
        # Test model creation
        from model import create_model
        model = create_model(num_classes=10)
        print_success("Model created successfully")
        
        # Test with dummy data
        device = torch.device(DockerConfig.DEVICE)
        model = model.to(device)
        
        # Create dummy batch
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
            
        if output.shape == (batch_size, 10):
            print_success("Model forward pass successful")
        else:
            print_error(f"Unexpected output shape: {output.shape}")
            
    except Exception as e:
        print_error(f"Training test failed: {e}")

def main():
    """Run all tests"""
    print_header("STL-10 Classification Docker Environment Test")
    print("Running comprehensive environment verification...")
    
    test_docker_environment()
    gpu_available = test_gpu_availability()
    test_dependencies()
    test_wandb_connection()
    test_huggingface()
    test_file_permissions()
    test_memory()
    run_quick_training_test()
    
    print_header("Test Summary")
    print("🎉 Environment test completed!")
    print("\nNext steps:")
    print("1. If all tests passed, run: python run_experiment.py")
    print("2. For Jupyter notebook: jupyter notebook --ip=0.0.0.0 --allow-root")
    print("3. For interactive shell: python -i")
    
    if gpu_available:
        print("\n🚀 GPU detected - ready for fast training!")
    else:
        print("\n💻 Using CPU - training will be slower but functional")

if __name__ == "__main__":
    main()