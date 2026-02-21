import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })