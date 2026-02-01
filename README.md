# CIFAR-10 CNN Training with Visualization

This project implements a CNN (ResNet-18) for CIFAR-10 classification with comprehensive visualization of gradient flow and weight updates using Weights & Biases (WandB).

## 📋 Project Overview

- **Model**: ResNet-18 (adapted for 32x32 CIFAR-10 images)
- **Dataset**: CIFAR-10 (50,000 training + 10,000 test images)
- **Custom DataLoader**: Implemented with data augmentation
- **FLOPs Analysis**: Model complexity measurement
- **Visualization**: Gradient flow and weight update tracking via WandB

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- WandB account ([sign up here](https://wandb.ai/))

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Login to WandB:
```bash
wandb login
```

## 📊 Usage

### Training the Model

Run the training script with default parameters (30 epochs):

```bash
python train_model.py
```

### Custom Training Parameters

```bash
python train_model.py --epochs 25 --batch-size 128 --lr 0.1 --project-name "my-cifar10-project" --run-name "my-experiment"
```

### Available Arguments

- `--epochs`: Number of training epochs (default: 30)
- `--batch-size`: Batch size for training (default: 128)
- `--lr`: Initial learning rate (default: 0.1)
- `--project-name`: WandB project name (default: 'cifar10-cnn-training')
- `--run-name`: WandB run name (default: 'resnet18-baseline')

## 📁 Project Structure

```
Assignment-2/
├── src/
│   ├── __init__.py
│   ├── dataloader.py       # Custom CIFAR-10 dataloader
│   ├── model.py            # ResNet-18 implementation
│   ├── flops_counter.py    # FLOPs counting utility
│   ├── train.py            # Training loop with visualization
│   └── utils.py            # Helper functions and trackers
├── data/                   # CIFAR-10 dataset (auto-downloaded)
├── models/                 # Saved model checkpoints
├── checkpoints/            # Training checkpoints
├── train_model.py          # Main execution script
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## 🔬 Model Architecture

**ResNet-18 for CIFAR-10**

- Modified first convolution layer for 32x32 input
- 4 residual blocks with increasing channels (64 → 128 → 256 → 512)
- Global average pooling
- Fully connected layer for 10 classes

### Model Complexity

*Results will be populated after first training run*

- **Total Parameters**: TBD
- **Trainable Parameters**: TBD
- **FLOPs**: TBD

## 📈 Training Details

### Hyperparameters

- **Optimizer**: SGD with momentum (0.9)
- **Learning Rate**: 0.1 (cosine annealing schedule)
- **Weight Decay**: 5e-4
- **Batch Size**: 128
- **Epochs**: 30

### Data Augmentation

- Random Crop (32x32 with padding=4)
- Random Horizontal Flip (p=0.5)
- Color Jitter (brightness, contrast, saturation=0.2)
- Normalization (CIFAR-10 mean/std)

## 📊 Visualizations in WandB

The following metrics and visualizations are tracked:

### Training Metrics
- Training loss (per step and per epoch)
- Training accuracy (per step and per epoch)
- Validation loss (per epoch)
- Validation accuracy (per epoch)
- Learning rate schedule

### Gradient Flow Visualization
- Mean gradient magnitude per layer
- Gradient standard deviation per layer
- Gradient norm per layer
- Updated every 100 training steps

### Weight Update Visualization
- Mean weight change per layer
- Weight change norm per layer
- Relative weight change per layer
- Updated every 100 training steps

## 🔍 Key Findings and Observations

*This section will be updated after training completion*

### Training Performance
- **Best Validation Accuracy**: TBD
- **Final Training Loss**: TBD
- **Training Time**: TBD

### Gradient Flow Analysis
- Observations on gradient magnitudes across layers
- Identification of vanishing/exploding gradients
- Layer-wise gradient behavior

### Weight Update Analysis
- Weight update patterns across epochs
- Layer-wise learning dynamics
- Convergence behavior

## 📝 Results

### Training Curves

*Training and validation curves will be available in WandB dashboard*

### Confusion Matrix

*Will be added after training completion*

## 🔗 Links

- **WandB Dashboard**: [Link will be added after training]
- **GitHub Repository**: [Link will be added]

## 📚 References

- ResNet Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- CIFAR-10 Dataset: [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/cifar.html)

## 👤 Author

**Name**: [Your Name]  
**Roll Number**: [Your Roll Number]  
**Assignment**: Lab 2 Worksheet

## 📄 License

This project is for educational purposes as part of ML/DL Ops coursework.

---

**Note**: Model weights and checkpoints are excluded from version control as per assignment requirements.
