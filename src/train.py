"""
Training Script for CIFAR-10 with Gradient and Weight Visualization
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import os

from .dataloader import get_dataloaders
from .model import get_model
from .utils import (
    GradientFlowTracker, 
    WeightUpdateTracker, 
    calculate_accuracy,
    AverageMeter,
    save_checkpoint
)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, 
                grad_tracker, weight_tracker, global_step):
    """
    Train for one epoch
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        grad_tracker: Gradient flow tracker
        weight_tracker: Weight update tracker
        global_step: Global training step counter
    
    Returns:
        tuple: (average_loss, average_accuracy, updated_global_step)
    """
    model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Track gradients (every 100 steps)
        if global_step % 100 == 0:
            grad_tracker.log_to_wandb(epoch, global_step)
        
        # Update weights
        optimizer.step()
        
        # Track weight updates (every 100 steps)
        if global_step % 100 == 0:
            weight_tracker.log_to_wandb(epoch, global_step)
        
        # Calculate metrics
        accuracy = calculate_accuracy(outputs, labels)
        losses.update(loss.item(), images.size(0))
        accuracies.update(accuracy, images.size(0))
        
        # Log to WandB
        wandb.log({
            'train/loss_step': loss.item(),
            'train/accuracy_step': accuracy,
            'train/learning_rate': optimizer.param_groups[0]['lr']
        }, step=global_step)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracies.avg:.2f}%'
        })
        
        global_step += 1
    
    return losses.avg, accuracies.avg, global_step


def validate(model, test_loader, criterion, device, epoch, global_step):
    """
    Validate the model
    
    Args:
        model: Neural network model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        global_step: Global training step counter
    
    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.eval()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f'Epoch {epoch} [Val]')
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate metrics
            accuracy = calculate_accuracy(outputs, labels)
            losses.update(loss.item(), images.size(0))
            accuracies.update(accuracy, images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accuracies.avg:.2f}%'
            })
    
    # Log to WandB
    wandb.log({
        'val/loss': losses.avg,
        'val/accuracy': accuracies.avg,
        'epoch': epoch
    }, step=global_step)
    
    return losses.avg, accuracies.avg


def train_model(config):
    """
    Main training function
    
    Args:
        config (dict): Training configuration
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Get data loaders
    print("\nLoading data...")
    train_loader, test_loader = get_dataloaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Get model
    print("\nInitializing model...")
    model = get_model(num_classes=10, device=device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs']
    )
    
    # Initialize trackers
    grad_tracker = GradientFlowTracker(model)
    weight_tracker = WeightUpdateTracker(model)
    
    # Training loop
    print(f"\nStarting training for {config['epochs']} epochs...")
    best_accuracy = 0.0
    global_step = 0
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc, global_step = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, grad_tracker, weight_tracker, global_step
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, test_loader, criterion, device, epoch, global_step
        )
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                'models/best_model.pth'
            )
            print(f"  ✓ New best accuracy: {best_accuracy:.2f}%")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                f'checkpoints/checkpoint_epoch_{epoch}.pth'
            )
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    print(f"{'='*60}")
    
    return model, best_accuracy


if __name__ == "__main__":
    # Training configuration
    config = {
        'batch_size': 128,
        'epochs': 30,
        'learning_rate': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'num_workers': 2
    }
    
    # Train model
    model, best_acc = train_model(config)
