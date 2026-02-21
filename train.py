import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': correct_predictions / total_samples
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Log metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            })
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save checkpoint
                best_model_path = os.path.join(self.config.MODEL_SAVE_PATH, "best_model.pth")
                save_model_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, val_acc, best_model_path
                )
                print(f"New best model saved with validation accuracy: {val_acc:.4f}")
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        print(f"\nTraining completed! Best validation accuracy: {self.best_val_acc:.4f}")
    
    def evaluate_test_set(self):
        """Evaluate on test set and generate visualizations"""
        print("\nEvaluating on test set...")
        
        # Load best model
        best_model_path = os.path.join(self.config.MODEL_SAVE_PATH, "best_model.pth")
        checkpoint = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Testing"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_accuracy = correct_predictions / total_samples
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Calculate class-wise accuracy
        class_accuracies = calculate_class_wise_accuracy(
            all_labels, all_predictions, self.config.CLASS_NAMES
        )
        
        # Log test accuracy
        wandb.log({"test_accuracy": test_accuracy})
        
        # Create and log confusion matrix
        cm_plot = create_confusion_matrix(all_labels, all_predictions, self.config.CLASS_NAMES)
        wandb.log({"confusion_matrix": wandb.Image(cm_plot)})
        cm_plot.savefig(os.path.join(self.config.RESULTS_PATH, "confusion_matrix.png"))
        cm_plot.close()
        
        # Create and log class-wise accuracy plot
        acc_plot = plot_class_wise_accuracy(class_accuracies)
        wandb.log({"class_wise_accuracy": wandb.Image(acc_plot)})
        acc_plot.savefig(os.path.join(self.config.RESULTS_PATH, "class_wise_accuracy.png"))
        acc_plot.close()
        
        # Log class-wise accuracies to wandb
        for class_name, accuracy in class_accuracies.items():
            wandb.log({f"accuracy_{class_name}": accuracy})
        
        # Log sample predictions
        log_sample_predictions(
            self.model, self.test_loader, self.config.CLASS_NAMES, 
            self.device, num_correct=10, num_incorrect=10
        )
        
        return test_accuracy, class_accuracies

if __name__ == "__main__":
    config = Config()
    trainer = STL10Trainer(config)
    trainer.train()
    test_accuracy, class_accuracies = trainer.evaluate_test_set()
    
    # Print exam sheet answers
    print("\n" + "="*50)
    print("EXAM SHEET ANSWERS")
    print("="*50)
    print(f"10. Test Accuracy: {test_accuracy:.4f}")
    print("\nClass-wise accuracy for each class:")
    for class_name, accuracy in class_accuracies.items():
        print(f"{class_name}: {accuracy:.4f}")
    print("="*50)