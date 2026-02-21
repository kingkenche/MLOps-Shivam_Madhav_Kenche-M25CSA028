import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']