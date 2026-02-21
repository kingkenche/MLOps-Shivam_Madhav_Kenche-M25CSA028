#!/usr/bin/env python3
"""
Evaluate pre-trained STL-10 model from Colab training
Works with existing model checkpoints and generates evaluation results
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from pathlib import Path

# Import your existing modules
from model import create_model
from data_loader import STL10Dataset, get_transforms
from docker_config import DockerConfig
import wandb

class ModelEvaluator:
    def __init__(self, model_path="models/best_model.pth", device="auto"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else device)
        self.model = None
        self.test_loader = None
        
        # Create results directory
        os.makedirs("results", exist_ok=True)
        
    def load_model(self):
        """Load the pre-trained model"""
        print(f"🔄 Loading model from {self.model_path}...")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
            
        # Create model architecture
        self.model = create_model(num_classes=DockerConfig.NUM_CLASSES)
        
        # Load trained weights
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✅ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
                    print(f"📊 Best accuracy: {checkpoint.get('best_accuracy', 'unknown'):.4f}")
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def setup_data_loader(self):
        """Setup test data loader"""
        print("🔄 Setting up data loader...")
        
        try:
            from datasets import load_dataset
            
            # Load dataset
            dataset = load_dataset(DockerConfig.DATASET_NAME)
            
            # Get transforms
            _, _, test_transform = get_transforms()
            
            # Create test dataset
            test_dataset = STL10Dataset(
                dataset['test'], 
                transform=test_transform
            )
            
            # Create test loader
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=DockerConfig.BATCH_SIZE,
                shuffle=False,
                num_workers=DockerConfig.NUM_WORKERS,
                pin_memory=DockerConfig.PIN_MEMORY
            )
            
            print(f"✅ Test dataset loaded: {len(test_dataset)} samples")
            
        except Exception as e:
            print(f"❌ Error setting up data loader: {e}")
            raise
    
    def evaluate_model(self):
        """Run comprehensive model evaluation"""
        print("🔄 Starting model evaluation...")
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"Processed {batch_idx + 1}/{len(self.test_loader)} batches")
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        return all_preds, all_labels, all_probs
    
    def calculate_metrics(self, predictions, labels):
        """Calculate comprehensive evaluation metrics"""
        print("📊 Calculating metrics...")
        
        # Overall accuracy
        accuracy = np.mean(predictions == labels)
        
        # Class-wise accuracy
        class_accuracies = {}
        for i, class_name in enumerate(DockerConfig.CLASS_NAMES):
            class_mask = labels == i
            if np.sum(class_mask) > 0:
                class_acc = np.mean(predictions[class_mask] == labels[class_mask])
                class_accuracies[class_name] = class_acc
        
        # Classification report
        report = classification_report(
            labels, predictions, 
            target_names=DockerConfig.CLASS_NAMES,
            output_dict=True
        )
        
        return accuracy, class_accuracies, report
    
    def generate_confusion_matrix(self, predictions, labels):
        """Generate and save confusion matrix"""
        print("🎨 Generating confusion matrix...")
        
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=DockerConfig.CLASS_NAMES,
            yticklabels=DockerConfig.CLASS_NAMES
        )
        plt.title('STL-10 Classification Confusion Matrix\n(Pre-trained Model Evaluation)', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        save_path = "results/confusion_matrix_evaluation.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to {save_path}")
        plt.close()
    
    def generate_class_accuracy_plot(self, class_accuracies):
        """Generate class-wise accuracy plot"""
        print("📊 Generating class accuracy plot...")
        
        classes = list(class_accuracies.keys())
        accuracies = [class_accuracies[cls] * 100 for cls in classes]
        
        plt.figure(figsize=(14, 8))
        bars = plt.bar(classes, accuracies, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
                             '#F7DC6F', '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA'])
        
        plt.title('STL-10 Class-wise Accuracy\n(Pre-trained Model Evaluation)', fontsize=14, fontweight='bold')
        plt.xlabel('Classes')
        plt.ylabel('Accuracy (%)')
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1, 
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = "results/class_wise_accuracy_evaluation.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Class accuracy plot saved to {save_path}")
        plt.close()
    
    def generate_sample_predictions(self, probabilities, predictions, labels, num_samples=16):
        """Generate sample prediction visualizations"""
        print("🖼️ Generating sample predictions...")
        
        # Get a few batches for visualization
        sample_images = []
        sample_labels = []
        sample_preds = []
        sample_probs = []
        
        with torch.no_grad():
            for batch_idx, (images, batch_labels) in enumerate(self.test_loader):
                if len(sample_images) >= num_samples:
                    break
                    
                images = images.to(self.device)
                outputs = self.model(images)
                batch_probs = torch.softmax(outputs, dim=1)
                batch_preds = torch.argmax(outputs, dim=1)
                
                # Take first few samples from this batch
                batch_size = min(num_samples - len(sample_images), images.size(0))
                
                sample_images.extend(images[:batch_size].cpu())
                sample_labels.extend(batch_labels[:batch_size].cpu())
                sample_preds.extend(batch_preds[:batch_size].cpu())
                sample_probs.extend(batch_probs[:batch_size].cpu())
        
        # Create visualization
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(sample_images))):
            img = sample_images[i]
            true_label = sample_labels[i].item()
            pred_label = sample_preds[i].item()
            confidence = sample_probs[i][pred_label].item()
            
            # Denormalize image
            img = img.clone()
            for t, m, s in zip(img, DockerConfig.MEAN, DockerConfig.STD):
                t.mul_(s).add_(m)
            img = torch.clamp(img, 0, 1)
            
            # Display image
            axes[i].imshow(img.permute(1, 2, 0))
            axes[i].axis('off')
            
            # Set title with prediction info
            true_class = DockerConfig.CLASS_NAMES[true_label]
            pred_class = DockerConfig.CLASS_NAMES[pred_label]
            color = 'green' if true_label == pred_label else 'red'
            
            axes[i].set_title(
                f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2f}',
                fontsize=10, color=color, fontweight='bold'
            )
        
        plt.suptitle('STL-10 Sample Predictions\n(Pre-trained Model)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = "results/sample_predictions_evaluation.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sample predictions saved to {save_path}")
        plt.close()
    
    def save_results_summary(self, accuracy, class_accuracies, report):
        """Save comprehensive results summary"""
        print("📝 Saving results summary...")
        
        summary_path = "results/EVALUATION_SUMMARY.md"
        
        with open(summary_path, 'w') as f:
            f.write("# STL-10 Pre-trained Model Evaluation Results\n\n")
            f.write(f"**Evaluation Date:** {torch.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Model Path:** {self.model_path}\n")
            f.write(f"**Device:** {self.device}\n\n")
            
            f.write(f"## 📊 Overall Performance\n\n")
            f.write(f"**Test Accuracy:** {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
            
            f.write("## 🎯 Class-wise Accuracy\n\n")
            f.write("| Class | Accuracy | Performance |\n")
            f.write("|-------|----------|-------------|\n")
            
            for class_name, acc in class_accuracies.items():
                if acc >= 0.90:
                    performance = "Excellent ⭐"
                elif acc >= 0.80:
                    performance = "Good ✅"
                elif acc >= 0.70:
                    performance = "Fair 🟡"
                else:
                    performance = "Needs Improvement 🔴"
                    
                f.write(f"| {class_name} | {acc:.4f} ({acc*100:.1f}%) | {performance} |\n")
            
            f.write("\n## 📈 Detailed Classification Report\n\n")
            f.write("```\n")
            f.write(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
            f.write("-" * 60 + "\n")
            
            for class_name in DockerConfig.CLASS_NAMES:
                if class_name in report:
                    metrics = report[class_name]
                    f.write(f"{class_name:<12} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                           f"{metrics['f1-score']:<10.3f} {metrics['support']:<10.0f}\n")
            
            f.write("-" * 60 + "\n")
            macro_avg = report['macro avg']
            f.write(f"{'Macro Avg':<12} {macro_avg['precision']:<10.3f} {macro_avg['recall']:<10.3f} "
                   f"{macro_avg['f1-score']:<10.3f} {macro_avg['support']:<10.0f}\n")
            
            weighted_avg = report['weighted avg']
            f.write(f"{'Weighted Avg':<12} {weighted_avg['precision']:<10.3f} {weighted_avg['recall']:<10.3f} "
                   f"{weighted_avg['f1-score']:<10.3f} {weighted_avg['support']:<10.0f}\n")
            f.write("```\n\n")
            
            f.write("## 🎨 Generated Files\n\n")
            f.write("- `confusion_matrix_evaluation.png` - Confusion matrix heatmap\n")
            f.write("- `class_wise_accuracy_evaluation.png` - Class accuracy bar chart\n")
            f.write("- `sample_predictions_evaluation.png` - Sample prediction examples\n")
            
        print(f"✅ Results summary saved to {summary_path}")
    
    def run_complete_evaluation(self):
        """Run complete evaluation pipeline"""
        print("🚀 Starting complete model evaluation...")
        print("=" * 60)
        
        try:
            # Load model and data
            self.load_model()
            self.setup_data_loader()
            
            # Run evaluation
            predictions, labels, probabilities = self.evaluate_model()
            
            # Calculate metrics
            accuracy, class_accuracies, report = self.calculate_metrics(predictions, labels)
            
            # Generate visualizations
            self.generate_confusion_matrix(predictions, labels)
            self.generate_class_accuracy_plot(class_accuracies)
            self.generate_sample_predictions(probabilities, predictions, labels)
            
            # Save summary
            self.save_results_summary(accuracy, class_accuracies, report)
            
            # Print results
            print("=" * 60)
            print("🎉 EVALUATION COMPLETE!")
            print("=" * 60)
            print(f"📊 Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"📁 Results saved in: results/ directory")
            print("=" * 60)
            
            return accuracy, class_accuracies
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            raise

def main():
    """Main evaluation function"""
    print("🐳 STL-10 Pre-trained Model Evaluator")
    print("=" * 60)
    
    # Check if model exists
    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Available models:")
        models_dir = Path("models")
        if models_dir.exists():
            for model_file in models_dir.glob("*.pth"):
                print(f"  - {model_file}")
        else:
            print("  - No models directory found")
        return
    
    # Run evaluation
    evaluator = ModelEvaluator(model_path=model_path)
    accuracy, class_accuracies = evaluator.run_complete_evaluation()
    
    print("\n🎯 Quick Summary:")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print("Top 3 performing classes:")
    sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)
    for i, (class_name, acc) in enumerate(sorted_classes[:3]):
        print(f"{i+1}. {class_name}: {acc*100:.1f}%")

if __name__ == "__main__":
    main()