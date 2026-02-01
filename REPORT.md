# Lab 2 Worksheet Report
## CIFAR-10 CNN Training with Gradient Flow and Weight Update Visualization

---

**Student Name**: Shivam Madhav Kenche  
**Roll Number**: M25CSA028  
**Assignment**: Lab 2 Worksheet  
**Date**: 1 February 2026

---

## 1. Introduction

This report presents the implementation and analysis of a ResNet-18 Convolutional Neural Network trained on the CIFAR-10 dataset. The project focuses on understanding model complexity through FLOPs analysis, monitoring gradient flow, and tracking weight updates during training using Weights & Biases (WandB) for visualization.

---

## 2. Model Architecture

### 2.1 ResNet-18 Overview

The implemented model is a ResNet-18 architecture adapted for CIFAR-10's 32×32 input images.

| Component | Description |
|-----------|-------------|
| **Input Layer** | 3×32×32 (RGB images) |
| **Initial Conv** | 3×3 conv, 64 filters, stride=1, padding=1 |
| **Layer 1** | 2 BasicBlocks, 64 channels |
| **Layer 2** | 2 BasicBlocks, 128 channels, stride=2 |
| **Layer 3** | 2 BasicBlocks, 256 channels, stride=2 |
| **Layer 4** | 2 BasicBlocks, 512 channels, stride=2 |
| **Pooling** | Global Average Pooling |
| **Output** | Fully Connected, 10 classes |

### 2.2 BasicBlock Structure

Each BasicBlock contains:
- Conv2d (3×3) → BatchNorm2d → ReLU
- Conv2d (3×3) → BatchNorm2d
- Shortcut connection (identity or 1×1 conv for dimension matching)
- ReLU activation

### 2.3 Key Architectural Decisions

1. **Modified Initial Convolution**: Changed from 7×7 (stride=2) to 3×3 (stride=1) to preserve spatial resolution for smaller 32×32 inputs
2. **No MaxPooling after initial conv**: Removed to maintain spatial dimensions
3. **Kaiming Initialization**: Used for better gradient flow in ReLU networks

---

## 3. Model Complexity Analysis (FLOPs)

### 3.1 Complexity Metrics

| Metric | Value |
|--------|-------|
| **Total Parameters** | 11.17M (11,173,962) |
| **Trainable Parameters** | 11.17M (11,173,962) |
| **FLOPs** | 556.66M (556,659,712) |

### 3.2 Parameter Distribution

The majority of parameters are concentrated in:
- **Layer 4**: 512 channels with the largest feature maps in channel dimension
- **Fully Connected Layer**: 512 × 10 = 5,120 parameters
- **Convolutional Layers**: Account for ~99% of total parameters

### 3.3 FLOPs Breakdown

- Earlier layers (Layer 1, 2) have higher spatial dimensions but fewer channels
- Later layers (Layer 3, 4) have more channels but smaller spatial dimensions
- The computational cost is balanced across the network due to this design

---

## 4. Training Configuration

### 4.1 Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Optimizer** | SGD with Momentum |
| **Momentum** | 0.9 |
| **Initial Learning Rate** | 0.1 |
| **Learning Rate Schedule** | Cosine Annealing |
| **Weight Decay** | 5e-4 |
| **Batch Size** | 128 |
| **Epochs** | 30 |

### 4.2 Data Augmentation

| Augmentation | Configuration |
|--------------|---------------|
| Random Crop | 32×32 with padding=4 |
| Random Horizontal Flip | p=0.5 |
| Color Jitter | brightness=0.2, contrast=0.2, saturation=0.2 |
| Normalization | Mean=[0.4914, 0.4822, 0.4465], Std=[0.2470, 0.2435, 0.2616] |

### 4.3 Training Environment

| Component | Details |
|-----------|---------|
| **GPU** | NVIDIA A30 |
| **CUDA Version** | 12.2 |
| **Python Version** | 3.10.12 |
| **Framework** | PyTorch |
| **Cluster** | SLURM (csedpu, node cn02) |

---

## 5. Training Results

### 5.1 Final Metrics

| Metric | Value |
|--------|-------|
| **Best Validation Accuracy** | **93.22%** |
| **Final Validation Accuracy** | 93.17% |
| **Final Validation Loss** | 0.226 |
| **Final Training Accuracy** | 99.22% |
| **Final Training Loss** | 0.030 |
| **Final Learning Rate** | 0.000274 |
| **Total Training Time** | ~14 minutes (836 seconds) |
| **Total Steps** | 11,700 |

### 5.2 Training Progression

The model showed consistent improvement throughout training:

| Epoch Range | Observation |
|-------------|-------------|
| 1-5 | Rapid initial learning, accuracy reaches ~80% |
| 6-15 | Steady improvement, accuracy reaches ~90% |
| 16-25 | Fine-tuning phase, accuracy stabilizes ~92-93% |
| 26-30 | Final convergence, best accuracy achieved |

### 5.3 Checkpoints Saved

| Checkpoint | Epoch |
|------------|-------|
| checkpoint_epoch_5.pth | 5 |
| checkpoint_epoch_10.pth | 10 |
| checkpoint_epoch_15.pth | 15 |
| checkpoint_epoch_20.pth | 20 |
| checkpoint_epoch_25.pth | 25 |
| checkpoint_epoch_30.pth | 30 |

---

## 6. Gradient Flow Analysis

### 6.1 Overview

Gradient flow was monitored every 100 training steps to understand how gradients propagate through the network during backpropagation.

### 6.2 Layer-wise Gradient Statistics (Final Epoch)

| Layer | Gradient Norm | Gradient Mean | Gradient Std |
|-------|---------------|---------------|--------------|
| conv1 | 0.076 | -0.00019 | 0.0018 |
| layer1.0.conv1 | 0.096 | 1.52e-05 | 0.0005 |
| layer1.0.conv2 | 0.102 | 4.91e-06 | 0.0005 |
| layer2.0.conv1 | 0.112 | -1.56e-05 | 0.0004 |
| layer2.0.conv2 | 0.132 | 1.30e-06 | 0.0003 |
| layer3.0.conv1 | 0.175 | 5.99e-06 | 0.0003 |
| layer3.0.conv2 | 0.241 | -5.76e-07 | 0.0003 |
| layer4.0.conv1 | 0.175 | 2.44e-07 | 0.0002 |
| layer4.0.conv2 | 0.131 | -1.02e-06 | 8.53e-05 |
| fc | 0.032 | 1.69e-10 | 0.0004 |

### 6.3 Key Observations

1. **No Vanishing Gradients**: All layers receive meaningful gradients (norms > 0.03)

2. **No Exploding Gradients**: Gradient norms remain bounded (< 0.3 for all layers)

3. **Healthy Gradient Flow**: 
   - The gradient norms show a bell-shaped pattern across layers
   - Middle layers (layer3) show the highest gradient magnitudes
   - This is expected behavior for ResNets with skip connections

4. **Residual Connections Effectiveness**:
   - Skip connections ensure gradients can flow directly to earlier layers
   - layer1 receives comparable gradients to later layers despite being further from the loss

5. **BatchNorm Contribution**:
   - BatchNorm layers help normalize gradient magnitudes
   - Prevents gradient statistics from varying wildly across layers

---

## 7. Weight Update Analysis

### 7.1 Overview

Weight updates were tracked every 100 steps to understand the learning dynamics and convergence behavior.

### 7.2 Layer-wise Weight Update Statistics (Final Epoch)

| Layer | Norm Change | Relative Change | Mean Change |
|-------|-------------|-----------------|-------------|
| conv1 | 0.00336 | 0.00073 | -6.83e-06 |
| layer1.0.conv1 | 0.00403 | 0.00090 | 1.61e-07 |
| layer1.0.conv2 | 0.00437 | 0.00091 | -2.84e-07 |
| layer2.0.conv1 | 0.00550 | 0.00099 | 9.48e-07 |
| layer2.0.conv2 | 0.00715 | 0.00104 | -2.65e-07 |
| layer3.0.conv1 | 0.00897 | 0.00104 | 4.12e-07 |
| layer3.0.conv2 | 0.01261 | 0.00117 | -3.24e-08 |
| layer4.0.conv1 | 0.00842 | 0.00106 | -4.59e-08 |
| layer4.0.conv2 | 0.00609 | 0.00094 | 5.18e-08 |
| fc | 0.00124 | 0.00017 | -8.39e-10 |

### 7.3 Key Observations

1. **Consistent Updates Across Layers**:
   - All layers receive meaningful weight updates (no frozen layers)
   - Relative change ranges from 0.0001 to 0.002 per step

2. **Learning Rate Schedule Effect**:
   - Cosine annealing gradually reduces learning rate
   - Weight update magnitudes decrease smoothly over training
   - Final learning rate (0.000274) produces fine-grained updates

3. **Layer-wise Learning Patterns**:
   - Middle layers (layer3) show larger absolute weight changes
   - FC layer shows smallest relative change (already well-learned features)
   - Earlier layers adapt more slowly (more stable low-level features)

4. **Convergence Behavior**:
   - Weight updates become smaller as training progresses
   - No oscillations or instability observed
   - Smooth convergence to local minimum

5. **No Weight Collapse**:
   - All layers maintain diverse weight updates
   - No layer shows zero or near-zero updates

---

## 8. WandB Visualizations

### 8.1 Logged Metrics

The following visualizations are available in the WandB dashboard:

1. **Training Curves**
   - Training loss (per step and per epoch)
   - Training accuracy (per step and per epoch)
   - Validation loss (per epoch)
   - Validation accuracy (per epoch)
   - Learning rate schedule

2. **Gradient Flow Visualization**
   - Bar chart showing gradient norms per layer
   - Updated every 100 training steps
   - Allows tracking gradient health throughout training

3. **Weight Update Visualization**
   - Bar chart showing weight update norms per layer
   - Relative weight change tracking
   - Updated every 100 training steps

4. **Model Metrics**
   - Total parameters
   - Trainable parameters
   - FLOPs count

### 8.2 Dashboard Access

- **WandB Project**: cifar10-cnn-training
- **Run Name**: resnet18-baseline
- **Run ID**: lj0kbe7z
- **Dashboard Link**: [https://api.wandb.ai/links/shivamkenche-indian-institute-of-technology-jodhpur/9c0kqdrh](https://api.wandb.ai/links/shivamkenche-indian-institute-of-technology-jodhpur/9c0kqdrh)
- **GitHub Repository**: [https://github.com/kingkenche/MLOps-Shivam_Madhav_Kenche-M25CSA028](https://github.com/kingkenche/MLOps-Shivam_Madhav_Kenche-M25CSA028)

---

## 9. Conclusions

### 9.1 Summary of Achievements

1. **Successfully implemented ResNet-18** adapted for CIFAR-10 classification
2. **Achieved 93.22% validation accuracy** in 30 epochs (~14 minutes training)
3. **Model complexity**: 11.17M parameters, 556.66M FLOPs
4. **Comprehensive monitoring** of gradient flow and weight updates via WandB

### 9.2 Key Insights

1. **ResNet Architecture Effectiveness**:
   - Skip connections successfully prevent vanishing gradients
   - All layers learn effectively throughout training
   - The architecture is well-suited for CIFAR-10

2. **Training Dynamics**:
   - Cosine annealing provides smooth learning rate decay
   - Data augmentation helps prevent overfitting
   - Model converges stably without oscillations

3. **Gradient Flow Health**:
   - No evidence of vanishing or exploding gradients
   - BatchNorm + ResNet connections ensure healthy gradient propagation
   - All layers contribute to learning

4. **Weight Update Patterns**:
   - Consistent learning across all layers
   - No frozen or collapsed layers
   - Smooth convergence behavior

### 9.3 Potential Improvements

1. **Longer Training**: More epochs could potentially improve accuracy further
2. **Advanced Augmentation**: AutoAugment or CutOut could help generalization
3. **Learning Rate Warmup**: Could help initial training stability
4. **Label Smoothing**: Could improve calibration and generalization

---

## 10. References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.

2. Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. Technical Report.

3. Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training. ICML 2015.

4. Loshchilov, I., & Hutter, F. (2016). SGDR: Stochastic Gradient Descent with Warm Restarts. ICLR 2017.

---

## Appendix A: Project Structure

```
Assignment-2/
├── src/
│   ├── __init__.py
│   ├── dataloader.py       # Custom CIFAR-10 dataloader
│   ├── model.py            # ResNet-18 implementation
│   ├── flops_counter.py    # FLOPs counting utility
│   ├── train.py            # Training loop with visualization
│   └── utils.py            # Helper functions and trackers
├── data/                   # CIFAR-10 dataset
├── checkpoints/            # Training checkpoints (not in Git)
├── train_model.py          # Main execution script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Appendix B: Submission Details

```
Student Name: Shivam Madhav Kenche
Roll Number: M25CSA028
Assignment: Lab 2 Worksheet

Key Results:
- Model: ResNet-18
- Best Validation Accuracy: 93.22%
- Total Epochs: 30
- Total FLOPs: 556.66M
- Total Parameters: 11.17M
- Training Time: ~14 minutes
```

---

*Report generated on 1 February 2026*
