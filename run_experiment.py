"""
Complete STL-10 Classification Experiment
Run this script to execute the entire pipeline
"""

import torch
import os
import sys
from config import Config
from train import STL10Trainer
from huggingface_upload import upload_model_to_huggingface, load_model_from_huggingface

def main():
    print("STL-10 Image Classification with MLOps Pipeline")
    print("="*60)
    
    # Initialize configuration
    config = Config()
    
    print(f"Device: {config.DEVICE}")
    print(f"Dataset: {config.DATASET_NAME}")
    print(f"Model: ResNet-18")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Max Epochs: {config.NUM_EPOCHS}")
    print("="*60)
    
    try:
        # Step 1-6: Training Pipeline
        print("\n🚀 Starting training pipeline...")
        trainer = STL10Trainer(config)
        trainer.train()
        
        # Step 7-8: Evaluation
        print("\n📊 Evaluating model...")
        test_accuracy, class_accuracies = trainer.evaluate_test_set()
        
        # Step 6: Upload model to HuggingFace
        print("\n📤 Preparing model for HuggingFace upload...")
        upload_model_to_huggingface(config)
        
        # Step 6b: Load model from HuggingFace (simulation)
        print("\n📥 Loading model from HuggingFace format...")
        loaded_model = load_model_from_huggingface(config)
        
        # Print final exam answers
        print("\n" + "="*60)
        print("🎯 FINAL EXAM ANSWERS")
        print("="*60)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("\nClass-wise Accuracy:")
        for i, (class_name, accuracy) in enumerate(class_accuracies.items(), 1):
            print(f"{i:2d}. {class_name:8s}: {accuracy:.4f}")
        print("="*60)
        
        print("\n✅ Experiment completed successfully!")
        print("📊 Check your Wandb dashboard for detailed results")
        print("📁 Check the 'results' folder for saved plots")
        print("💾 Check the 'models' folder for saved model checkpoint")
        
        # Generate summary for GitHub upload
        create_results_summary(test_accuracy, class_accuracies, config)
        
    except Exception as e:
        print(f"\n❌ Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def create_results_summary(test_accuracy, class_accuracies, config):
    """Create a summary of results for GitHub upload"""
    
    summary_content = f"""# STL-10 Classification Results

## Experiment Summary
- **Model**: ResNet-18 (pretrained)
- **Dataset**: STL-10 Subset from HuggingFace
- **Training Configuration**:
  - Batch Size: {config.BATCH_SIZE}
  - Learning Rate: {config.LEARNING_RATE}
  - Max Epochs: {config.NUM_EPOCHS}
  - Device: {config.DEVICE}

## Results

### Overall Test Accuracy: {test_accuracy:.4f}

### Class-wise Accuracy:
"""
    
    for i, (class_name, accuracy) in enumerate(class_accuracies.items(), 1):
        summary_content += f"{i:2d}. **{class_name}**: {accuracy:.4f}\n"
    
    summary_content += f"""
## Files Generated:
- Best model checkpoint: `models/best_model.pth`
- Confusion matrix: `results/confusion_matrix.png`
- Class-wise accuracy plot: `results/class_wise_accuracy.png`
- Model for HuggingFace: `models/pytorch_model.bin`

## Wandb Dashboard:
All training metrics, loss curves, and visualizations are logged to Wandb.

## Next Steps:
1. Push code and results to GitHub
2. Upload model to HuggingFace Hub
3. Share Wandb dashboard link

---
*Generated automatically by STL-10 Classification Pipeline*
"""
    
    # Save summary
    with open("RESULTS_SUMMARY.md", "w") as f:
        f.write(summary_content)
    
    print(f"\n📄 Results summary saved to RESULTS_SUMMARY.md")

if __name__ == "__main__":
    main()