"""
dataset.py - CIFAR-100 dataset loading and preprocessing for ViT
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def get_cifar100_transforms(img_size=224):
    """
    Returns train and val/test transforms suitable for ViT-S pre-trained on ImageNet.
    Resize to 224x224 and normalize with ImageNet stats.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(img_size, padding=int(img_size * 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_transform, val_transform


def get_dataloaders(data_dir="./data", batch_size=64, num_workers=2, img_size=224):
    """
    Download CIFAR-100 and return train, val, test DataLoaders.
    We use 45000/5000 split from the official 50000 train set,
    keeping the 10000 test set intact.

    NOTE: Inside Docker, /dev/shm is limited to 64 MB by default.
    If you hit a 'Bus error' with workers > 0, either:
      (a) pass --workers 0  to this script, or
      (b) restart the container with --shm-size=8g
    """
    train_transform, val_transform = get_cifar100_transforms(img_size)

    full_train = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    full_train_val = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=val_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=val_transform
    )

    # 90/10 train-val split
    n_total  = len(full_train)   # 50 000
    n_train  = int(0.9 * n_total)
    n_val    = n_total - n_train

    indices = list(range(n_total))
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_dataset = torch.utils.data.Subset(full_train,     train_idx)
    val_dataset   = torch.utils.data.Subset(full_train_val, val_idx)

    # persistent_workers=True avoids re-spawning workers each epoch.
    # Set num_workers=0 inside Docker with limited /dev/shm.
    pw = num_workers > 0
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=pw, prefetch_factor=2 if pw else None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=pw, prefetch_factor=2 if pw else None,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=pw, prefetch_factor=2 if pw else None,
    )

    return train_loader, val_loader, test_loader


CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree',
    'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]
