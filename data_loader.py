import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader