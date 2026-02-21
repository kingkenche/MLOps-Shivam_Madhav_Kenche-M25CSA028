import torch

class Config:
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
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"