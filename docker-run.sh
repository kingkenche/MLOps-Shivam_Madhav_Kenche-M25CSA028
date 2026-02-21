#!/bin/bash

# Docker management script for STL-10 Classification
# Usage: ./docker-run.sh [build|run|stop|logs|shell|jupyter|gpu-test]

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_NAME="stl10-classification"
IMAGE_NAME="stl10-classifier"
CONTAINER_NAME="stl10-classification"

# Functions
print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    if ! docker system info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi
    
    print_status "Docker is ready"
}

check_nvidia_docker() {
    if command -v nvidia-docker &> /dev/null || docker info | grep -q "nvidia"; then
        print_status "NVIDIA Docker support detected"
        return 0
    else
        print_warning "NVIDIA Docker support not found, will use CPU only"
        return 1
    fi
}

build_image() {
    print_info "Building Docker image..."
    docker build -t $IMAGE_NAME . || {
        print_error "Failed to build Docker image"
        exit 1
    }
    print_status "Docker image built successfully"
}

run_training() {
    print_info "Starting STL-10 training in Docker..."
    
    # Check if NVIDIA runtime is available
    if check_nvidia_docker; then
        RUNTIME_FLAG="--runtime=nvidia"
        GPU_ENV="-e NVIDIA_VISIBLE_DEVICES=all"
    else
        RUNTIME_FLAG=""
        GPU_ENV=""
    fi
    
    docker run -it --rm \
        $RUNTIME_FLAG \
        $GPU_ENV \
        -e WANDB_API_KEY="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" \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/results:/app/results" \
        -v "$(pwd)/wandb:/app/wandb" \
        -v "$HOME/.cache/huggingface:/home/appuser/.cache/huggingface" \
        -v "$HOME/.cache/torch:/home/appuser/.cache/torch" \
        --name $CONTAINER_NAME \
        $IMAGE_NAME
}

run_evaluation() {
    print_info "Evaluating pre-trained STL-10 model..."
    print_info "Using existing model from Colab training"
    
    # Check if NVIDIA runtime is available
    if check_nvidia_docker; then
        RUNTIME_FLAG="--runtime=nvidia"
        GPU_ENV="-e NVIDIA_VISIBLE_DEVICES=all"
    else
        RUNTIME_FLAG=""
        GPU_ENV=""
    fi
    
    docker run -it --rm \
        $RUNTIME_FLAG \
        $GPU_ENV \
        -e WANDB_API_KEY="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" \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/results:/app/results" \
        -v "$(pwd)/wandb:/app/wandb" \
        -v "$HOME/.cache/huggingface:/home/appuser/.cache/huggingface" \
        -v "$HOME/.cache/torch:/home/appuser/.cache/torch" \
        --name "${CONTAINER_NAME}-eval" \
        $IMAGE_NAME \
        python evaluate_pretrained.py
}

run_jupyter() {
    print_info "Starting Jupyter notebook server..."
    
    if check_nvidia_docker; then
        RUNTIME_FLAG="--runtime=nvidia"
        GPU_ENV="-e NVIDIA_VISIBLE_DEVICES=all"
    else
        RUNTIME_FLAG=""
        GPU_ENV=""
    fi
    
    docker run -it --rm \
        $RUNTIME_FLAG \
        $GPU_ENV \
        -p 8888:8888 \
        -e WANDB_API_KEY="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" \
        -v "$(pwd):/app" \
        -v "$HOME/.cache/huggingface:/home/appuser/.cache/huggingface" \
        -v "$HOME/.cache/torch:/home/appuser/.cache/torch" \
        --name "${CONTAINER_NAME}-jupyter" \
        $IMAGE_NAME \
        jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
    
    print_info "Jupyter notebook available at: http://localhost:8888"
}

run_shell() {
    print_info "Starting interactive shell..."
    
    if check_nvidia_docker; then
        RUNTIME_FLAG="--runtime=nvidia"
        GPU_ENV="-e NVIDIA_VISIBLE_DEVICES=all"
    else
        RUNTIME_FLAG=""
        GPU_ENV=""
    fi
    
    docker run -it --rm \
        $RUNTIME_FLAG \
        $GPU_ENV \
        -e WANDB_API_KEY="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" \
        -v "$(pwd):/app" \
        -v "$HOME/.cache/huggingface:/home/appuser/.cache/huggingface" \
        -v "$HOME/.cache/torch:/home/appuser/.cache/torch" \
        --name "${CONTAINER_NAME}-shell" \
        $IMAGE_NAME \
        /bin/bash
}

stop_containers() {
    print_info "Stopping all containers..."
    docker stop $(docker ps -q --filter "name=$PROJECT_NAME") 2>/dev/null || true
    print_status "Containers stopped"
}

show_logs() {
    print_info "Showing container logs..."
    docker logs -f $CONTAINER_NAME
}

gpu_test() {
    print_info "Testing GPU availability in container..."
    
    if check_nvidia_docker; then
        RUNTIME_FLAG="--runtime=nvidia"
        GPU_ENV="-e NVIDIA_VISIBLE_DEVICES=all"
    else
        RUNTIME_FLAG=""
        GPU_ENV=""
    fi
    
    docker run --rm \
        $RUNTIME_FLAG \
        $GPU_ENV \
        $IMAGE_NAME \
        python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA devices:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'Device {i}: {torch.cuda.get_device_name(i)}')
        print(f'Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB')
"
}

show_usage() {
    echo -e "${BLUE}🐳 STL-10 Classification Docker Manager${NC}"
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  build      Build the Docker image"
    echo "  run        Run training (default)"
    echo "  evaluate   Evaluate pre-trained model from Colab"
    echo "  jupyter    Start Jupyter notebook server"
    echo "  shell      Open interactive shell"
    echo "  stop       Stop all running containers"
    echo "  logs       Show container logs"
    echo "  gpu-test   Test GPU availability"
    echo "  help       Show this help message"
    echo
    echo "Examples:"
    echo "  $0 build && $0 run"
    echo "  $0 jupyter"
    echo "  $0 shell"
}

# Main script
check_docker

case "${1:-run}" in
    build)
        build_image
        ;;
    run)
        build_image
        run_training
        ;;
    evaluate)
        build_image
        run_evaluation
        ;;
    jupyter)
        build_image
        run_jupyter
        ;;
    shell)
        build_image
        run_shell
        ;;
    stop)
        stop_containers
        ;;
    logs)
        show_logs
        ;;
    gpu-test)
        build_image
        gpu_test
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac