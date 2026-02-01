"""
Custom CIFAR-10 DataLoader Implementation
"""
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import numpy as np
from PIL import Image


class CustomCIFAR10Dataset(Dataset):
    """
    Custom Dataset wrapper for CIFAR-10 with custom transformations
    """
    def __init__(self, root='./data', train=True, transform=None, download=True):
        """
        Args:
            root (str): Root directory for data storage
            train (bool): If True, creates dataset from training set, otherwise test set
            transform (callable, optional): Optional transform to be applied on a sample
            download (bool): If True, downloads the dataset
        """
        # Load CIFAR-10 dataset
        self.cifar10 = CIFAR10(root=root, train=train, download=download, transform=None)
        self.transform = transform
        self.train = train
        
        # CIFAR-10 class names
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
    def __len__(self):
        return len(self.cifar10)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        image, label = self.cifar10[idx]
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(train=True):
    """
    Get data transformations for CIFAR-10
    
    Args:
        train (bool): If True, returns training transforms with augmentation
    
    Returns:
        transforms.Compose: Composed transformations
    """
    if train:
        # Training transforms with data augmentation
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                               std=[0.2470, 0.2435, 0.2616])
        ])
    else:
        # Test transforms without augmentation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                               std=[0.2470, 0.2435, 0.2616])
        ])
    
    return transform


def get_dataloaders(batch_size=128, num_workers=2, root='./data'):
    """
    Create train and test dataloaders for CIFAR-10
    
    Args:
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of worker processes for data loading
        root (str): Root directory for data storage
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Get transforms
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)
    
    # Create datasets
    train_dataset = CustomCIFAR10Dataset(
        root=root,
        train=True,
        transform=train_transform,
        download=True
    )
    
    test_dataset = CustomCIFAR10Dataset(
        root=root,
        train=False,
        transform=test_transform,
        download=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Classes: {train_dataset.classes}")
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test the dataloader
    print("Testing Custom CIFAR-10 DataLoader...")
    train_loader, test_loader = get_dataloaders(batch_size=4)
    
    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    print("\nDataLoader test successful!")
