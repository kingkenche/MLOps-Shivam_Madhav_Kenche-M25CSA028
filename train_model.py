"""
Main Training Script for CIFAR-10 CNN
Run this script to train the model with WandB tracking
"""
import torch
import wandb
import argparse
from src.train import train_model
from src.flops_counter import print_flops_analysis
from src.model import get_model


def main():
    """Main execution function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train CNN on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--project-name', type=str, default='cifar10-cnn-training', 
                       help='WandB project name')
    parser.add_argument('--run-name', type=str, default='resnet18-baseline',
                       help='WandB run name')
    args = parser.parse_args()
    
    # Configuration
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'num_workers': 2,
        'model': 'ResNet-18',
        'dataset': 'CIFAR-10'
    }
    
    # Initialize WandB
    print("Initializing WandB...")
    wandb.init(
        project=args.project_name,
        name=args.run_name,
        config=config,
        save_code=True
    )
    
    # Print configuration
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    for key, value in config.items():
        print(f"{key:20s}: {value}")
    print("="*60 + "\n")
    
    # Count FLOPs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nAnalyzing model complexity...")
    temp_model = get_model(device=device)
    flops_results = print_flops_analysis(temp_model, device=device)
    
    # Log FLOPs to WandB
    wandb.config.update({
        'total_flops': flops_results['total_flops'],
        'total_params': flops_results['total_params'],
        'trainable_params': flops_results['trainable_params']
    })
    
    del temp_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Train model
    model, best_accuracy = train_model(config)
    
    # Log final results
    wandb.summary['best_accuracy'] = best_accuracy
    
    print("\n✓ Training completed successfully!")
    print(f"✓ Check your results at: {wandb.run.get_url()}")
    
    # Finish WandB run
    wandb.finish()


if __name__ == "__main__":
    main()
