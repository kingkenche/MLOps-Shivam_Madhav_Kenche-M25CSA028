"""
Docker-specific configuration for STL-10 Classification
Override settings for containerized deployment
"""
import os
import torch

class DockerConfig:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration  
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration (optimized for Docker)
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '16'))  # Reduced for memory efficiency
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.001'))
    NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', '15'))  # Reduced for faster training
    PATIENCE = int(os.getenv('PATIENCE', '5'))
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = os.getenv('WANDB_PROJECT', 'stl10-classification-docker')
    WANDB_ENTITY = os.getenv('WANDB_ENTITY', None)
    WANDB_API_KEY = os.getenv(
        'WANDB_API_KEY', 
        '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'
    )
    
    # HuggingFace configuration
    HF_MODEL_NAME = os.getenv('HF_MODEL_NAME', 'kingkenche/stl10-resnet18-docker')
    
    # Docker-optimized paths
    MODEL_SAVE_PATH = "/app/models"
    RESULTS_PATH = "/app/results"
    CACHE_DIR = "/home/appuser/.cache"
    
    # Docker environment detection
    IS_DOCKER = os.path.exists('/.dockerenv') or os.getenv('DOCKER_CONTAINER') == 'true'
    
    # Device configuration with Docker considerations
    if torch.cuda.is_available() and os.getenv('CUDA_VISIBLE_DEVICES') != '-1':
        DEVICE = "cuda"
        # Memory optimization for Docker
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        DEVICE = "cpu"
    
    # Docker-specific memory settings
    if IS_DOCKER:
        # Optimize for containerized environment
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid warnings
        
    # Multiprocessing configuration for Docker
    NUM_WORKERS = int(os.getenv('NUM_WORKERS', '2'))  # Reduced for containers
    PIN_MEMORY = os.getenv('PIN_MEMORY', 'true').lower() == 'true' and DEVICE == "cuda"
    
    # Docker health check settings
    HEALTH_CHECK_INTERVAL = int(os.getenv('HEALTH_CHECK_INTERVAL', '30'))
    
    # Logging configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Network configuration for Docker
    HUGGINGFACE_HUB_VERBOSITY = os.getenv('HUGGINGFACE_HUB_VERBOSITY', 'warning')
    
    @classmethod
    def print_config(cls):
        """Print current configuration for debugging"""
        print("=" * 50)
        print("🐳 Docker Configuration")
        print("=" * 50)
        print(f"Docker Environment: {cls.IS_DOCKER}")
        print(f"Device: {cls.DEVICE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Workers: {cls.NUM_WORKERS}")
        print(f"Pin Memory: {cls.PIN_MEMORY}")
        print(f"Model Path: {cls.MODEL_SAVE_PATH}")
        print(f"Results Path: {cls.RESULTS_PATH}")
        print(f"Wandb Project: {cls.WANDB_PROJECT}")
        if cls.DEVICE == "cuda":
            print(f"CUDA Devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  Device {i}: {props.name} ({props.total_memory // 1024**3}GB)")
        print("=" * 50)