# Google Colab Configuration for STL-10 Classification
class ColabConfig:
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
    
    # Training configuration (Optimized for Colab)
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 15  # Reduced for Colab time limits
    PATIENCE = 3     # Early stopping patience
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification-colab"
    WANDB_ENTITY = None
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18-colab"
    
    # Colab paths
    MODEL_SAVE_PATH = "/content/models"
    RESULTS_PATH = "/content/results"
    
    # Device (will be set automatically in notebook)
    DEVICE = "cuda"  # Default, will be detected