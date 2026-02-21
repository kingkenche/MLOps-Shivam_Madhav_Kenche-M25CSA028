#!/bin/bash

# Docker entrypoint script for STL-10 Classification
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🐳 Starting STL-10 Classification Docker Container${NC}"

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check GPU availability
echo -e "${BLUE}🔍 Checking system resources...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU Status:${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    print_status "NVIDIA GPU detected"
else
    print_warning "No NVIDIA GPU detected, using CPU"
fi

# Check Python and torch
echo -e "${BLUE}🐍 Python Environment:${NC}"
python --version
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "CUDA device count: $(python -c 'import torch; print(torch.cuda.device_count())')"
    print_status "CUDA environment ready"
fi

# Create necessary directories
echo -e "${BLUE}📁 Setting up directories...${NC}"
mkdir -p models results wandb sample_data
print_status "Directories created"

# Set permissions
echo -e "${BLUE}🔒 Setting permissions...${NC}"
chmod +x *.py 2>/dev/null || true
print_status "Permissions set"

# Validate Wandb API key
echo -e "${BLUE}📊 Checking Wandb configuration...${NC}"
if [ ! -z "$WANDB_API_KEY" ]; then
    print_status "Wandb API key configured"
else
    print_warning "Wandb API key not found in environment"
fi

# Display available commands
echo -e "${BLUE}🚀 Available commands:${NC}"
echo "  python run_experiment.py        - Run full training pipeline"
echo "  python train.py                 - Run training only"
echo "  python -c 'import torch; print(torch.cuda.is_available())'  - Test CUDA"
echo "  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root  - Start Jupyter"

# Execute the command passed as arguments
if [ $# -eq 0 ]; then
    print_status "No command specified, running default experiment..."
    exec python run_experiment.py
else
    print_status "Executing: $@"
    exec "$@"
fi