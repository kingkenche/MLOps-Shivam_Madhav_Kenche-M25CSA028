"""
Utility Functions for Training and Visualization
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import wandb


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """
    Save model checkpoint
    
    Args:
        model (nn.Module): Model to save
        optimizer: Optimizer state
        epoch (int): Current epoch
        loss (float): Current loss
        accuracy (float): Current accuracy
        filepath (str): Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath, device='cuda'):
    """
    Load model checkpoint
    
    Args:
        model (nn.Module): Model to load weights into
        optimizer: Optimizer to load state into
        filepath (str): Path to checkpoint file
        device (str): Device to load model on
    
    Returns:
        tuple: (epoch, loss, accuracy)
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']


class GradientFlowTracker:
    """Track gradient flow through the network"""
    
    def __init__(self, model):
        self.model = model
        self.gradient_data = defaultdict(list)
    
    def track(self):
        """Collect gradient statistics for each layer"""
        gradient_stats = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None and param.requires_grad:
                grad = param.grad.cpu().detach()
                
                gradient_stats[name] = {
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'max': grad.max().item(),
                    'min': grad.min().item(),
                    'norm': grad.norm().item()
                }
        
        return gradient_stats
    
    def log_to_wandb(self, epoch, step):
        """Log gradient statistics to WandB"""
        gradient_stats = self.track()
        
        # Prepare data for logging
        log_dict = {}
        layer_names = []
        mean_grads = []
        std_grads = []
        grad_norms = []
        
        for name, stats in gradient_stats.items():
            # Log individual layer stats
            log_dict[f"gradients/{name}/mean"] = stats['mean']
            log_dict[f"gradients/{name}/std"] = stats['std']
            log_dict[f"gradients/{name}/norm"] = stats['norm']
            
            # Collect for summary plot
            layer_names.append(name.split('.')[-2] + '.' + name.split('.')[-1])
            mean_grads.append(abs(stats['mean']))
            std_grads.append(stats['std'])
            grad_norms.append(stats['norm'])
        
        # Create gradient flow visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Mean gradients
        axes[0].bar(range(len(mean_grads)), mean_grads)
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Mean |Gradient|')
        axes[0].set_title(f'Gradient Flow - Mean (Epoch {epoch})')
        axes[0].set_xticks(range(len(layer_names)))
        axes[0].set_xticklabels(layer_names, rotation=90, fontsize=6)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Gradient std
        axes[1].bar(range(len(std_grads)), std_grads, color='orange')
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('Gradient Std')
        axes[1].set_title(f'Gradient Flow - Std (Epoch {epoch})')
        axes[1].set_xticks(range(len(layer_names)))
        axes[1].set_xticklabels(layer_names, rotation=90, fontsize=6)
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Gradient norms
        axes[2].bar(range(len(grad_norms)), grad_norms, color='green')
        axes[2].set_xlabel('Layer')
        axes[2].set_ylabel('Gradient Norm')
        axes[2].set_title(f'Gradient Flow - Norm (Epoch {epoch})')
        axes[2].set_xticks(range(len(layer_names)))
        axes[2].set_xticklabels(layer_names, rotation=90, fontsize=6)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Log to WandB
        log_dict['gradient_flow_visualization'] = wandb.Image(fig)
        wandb.log(log_dict, step=step)
        
        plt.close(fig)


class WeightUpdateTracker:
    """Track weight updates during training"""
    
    def __init__(self, model):
        self.model = model
        self.previous_weights = {}
        self._save_weights()
    
    def _save_weights(self):
        """Save current weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.previous_weights[name] = param.data.clone().cpu()
    
    def track_and_update(self):
        """Track weight changes and update saved weights"""
        weight_changes = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.previous_weights:
                current_weight = param.data.cpu()
                previous_weight = self.previous_weights[name]
                
                # Calculate change
                change = current_weight - previous_weight
                
                weight_changes[name] = {
                    'mean_change': change.mean().item(),
                    'std_change': change.std().item(),
                    'max_change': change.abs().max().item(),
                    'norm_change': change.norm().item(),
                    'relative_change': (change.norm() / (previous_weight.norm() + 1e-8)).item()
                }
        
        # Update saved weights
        self._save_weights()
        
        return weight_changes
    
    def log_to_wandb(self, epoch, step):
        """Log weight update statistics to WandB"""
        weight_changes = self.track_and_update()
        
        # Prepare data for logging
        log_dict = {}
        layer_names = []
        mean_changes = []
        norm_changes = []
        relative_changes = []
        
        for name, stats in weight_changes.items():
            # Log individual layer stats
            log_dict[f"weight_updates/{name}/mean_change"] = stats['mean_change']
            log_dict[f"weight_updates/{name}/norm_change"] = stats['norm_change']
            log_dict[f"weight_updates/{name}/relative_change"] = stats['relative_change']
            
            # Collect for summary plot
            layer_names.append(name.split('.')[-2] + '.' + name.split('.')[-1])
            mean_changes.append(abs(stats['mean_change']))
            norm_changes.append(stats['norm_change'])
            relative_changes.append(stats['relative_change'])
        
        # Create weight update visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Mean weight changes
        axes[0].bar(range(len(mean_changes)), mean_changes)
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Mean |Weight Change|')
        axes[0].set_title(f'Weight Updates - Mean (Epoch {epoch})')
        axes[0].set_xticks(range(len(layer_names)))
        axes[0].set_xticklabels(layer_names, rotation=90, fontsize=6)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Weight change norms
        axes[1].bar(range(len(norm_changes)), norm_changes, color='orange')
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('Weight Change Norm')
        axes[1].set_title(f'Weight Updates - Norm (Epoch {epoch})')
        axes[1].set_xticks(range(len(layer_names)))
        axes[1].set_xticklabels(layer_names, rotation=90, fontsize=6)
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Relative weight changes
        axes[2].bar(range(len(relative_changes)), relative_changes, color='green')
        axes[2].set_xlabel('Layer')
        axes[2].set_ylabel('Relative Weight Change')
        axes[2].set_title(f'Weight Updates - Relative (Epoch {epoch})')
        axes[2].set_xticks(range(len(layer_names)))
        axes[2].set_xticklabels(layer_names, rotation=90, fontsize=6)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Log to WandB
        log_dict['weight_update_visualization'] = wandb.Image(fig)
        wandb.log(log_dict, step=step)
        
        plt.close(fig)


def calculate_accuracy(outputs, labels):
    """
    Calculate classification accuracy
    
    Args:
        outputs (torch.Tensor): Model outputs
        labels (torch.Tensor): Ground truth labels
    
    Returns:
        float: Accuracy percentage
    """
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = 100 * correct / total
    return accuracy


class AverageMeter:
    """Compute and store the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
