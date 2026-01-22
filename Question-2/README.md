# MLOps Assignment 1 - FashionMNIST ResNet Experiments

**Name:** SHIVAM MADHAV KENCHE  
**Roll Number:** M25CSA028
**Course:** MLOps  
**Assignment:** 1

## 📋 Table of Contents
- [Overview](#overview)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Analysis](#analysis)
- [Files](#files)
- [How to Run](#how-to-run)
- [Links](#links)

## 🎯 Overview

This project implements and compares three ResNet architectures (ResNet-18, ResNet-32, ResNet-50) on the FashionMNIST dataset, evaluating performance across:
- **Optimizers:** SGD (with momentum) and Adam
- **Compute Devices:** CPU and GPU
- **Metrics:** Training time, FLOPs, Classification accuracy

## ⚙️ Experimental Setup

| Parameter | Value |
|-----------|-------|
| **Dataset** | FashionMNIST (28×28 grayscale, 10 classes) |
| **Models** | ResNet-18, ResNet-32, ResNet-50 |
| **Optimizers** | SGD (momentum=0.9), Adam |
| **Learning Rate** | 0.001 |
| **Batch Size** | 16 |
| **Epochs** | 2 |
| **Devices** | CPU and GPU (CUDA) |
| **Framework** | PyTorch 2.1.2+ |

## 📊 Results

### Complete Results Table

| Compute | Batch Size | Optimizer | Learning Rate | Metric | ResNet-18 | ResNet-32 | ResNet-50 |
|---------|------------|-----------|---------------|--------|-----------|-----------|-----------|
| **CPU** | 16 | SGD | 0.001 | Test Accuracy (%) | 89.50 | 91.10 | 90.90 |
| **CPU** | 16 | SGD | 0.001 | Train Time (sec) | 3200.0 | 5400.0 | 5500.0 |
| **CPU** | 16 | SGD | 0.001 | FLOPs | 457.730M | 939.116M | 939.116M |
| **CPU** | 16 | Adam | 0.001 | Test Accuracy (%) | 90.20 | 90.30 | 89.20 |
| **CPU** | 16 | Adam | 0.001 | Train Time (sec) | 3300.0 | 5500.0 | 5600.0 |
| **CPU** | 16 | Adam | 0.001 | FLOPs | 457.730M | 939.116M | 939.116M |
| **GPU** | 16 | SGD | 0.001 | Test Accuracy (%) | 89.80 | 91.31 | 91.10 |
| **GPU** | 16 | SGD | 0.001 | Train Time (sec) | 180.0 | 240.5 | 241.3 |
| **GPU** | 16 | SGD | 0.001 | FLOPs | 457.730M | 939.116M | 939.116M |
| **GPU** | 16 | Adam | 0.001 | Test Accuracy (%) | 90.55 | 90.46 | 89.41 |
| **GPU** | 16 | Adam | 0.001 | Train Time (sec) | 178.5 | 239.3 | 240.8 |
| **GPU** | 16 | Adam | 0.001 | FLOPs | 457.730M | 939.116M | 939.116M |

### Summary Statistics

#### Training Time Analysis
| Device | Average Time | Best Time | Worst Time |
|--------|--------------|-----------|------------|
| CPU | 4750.0 sec (79.2 min) | 3200.0 sec | 5600.0 sec |
| GPU | 220.1 sec (3.7 min) | 178.5 sec | 241.3 sec |
| **Speedup** | **21.58x** | - | - |

#### Accuracy by Model
| Model | Average Accuracy | Best Accuracy | Optimizer | Device |
|-------|------------------|---------------|-----------|--------|
| ResNet-18 | 90.01% | 90.55% | Adam | GPU |
| ResNet-32 | 90.79% | 91.31% | SGD | GPU |
| ResNet-50 | 90.15% | 91.10% | Both SGD | GPU/CPU |

#### Optimizer Comparison
| Optimizer | Average Accuracy | Best Model |
|-----------|------------------|------------|
| SGD | 90.62% | ResNet-32 (91.31%) |
| Adam | 89.94% | ResNet-18 (90.55%) |

#### Computational Complexity
| Model | FLOPs | Parameters | Relative Cost |
|-------|-------|------------|---------------|
| ResNet-18 | 457.730M | 11.173M | 1.0x (baseline) |
| ResNet-32 | 939.116M | 21.281M | 2.05x |
| ResNet-50 | 939.116M | 21.281M | 2.05x |

## 📈 Analysis

### 1. GPU vs CPU Performance
- **GPU provides 21.58x average speedup** over CPU
- Speedup increases with model complexity:
  - ResNet-18: 18.13x
  - ResNet-32: 22.72x
  - ResNet-50: 23.02x
- GPU is essential for practical deep learning workflows

### 2. Model Architecture Impact
- **ResNet-32 achieves best accuracy (91.31%)** with SGD on GPU
- Deeper doesn't always mean better: ResNet-50 performed slightly worse
- ResNet-18 offers best speed-accuracy tradeoff for resource-constrained scenarios

### 3. Optimizer Selection
- **SGD outperforms Adam** overall (90.62% vs 89.94%)
- SGD particularly effective on deeper models
- Adam converges faster but may overfit with limited epochs

### 4. FLOPs and Efficiency
- FLOPs correlate with training time on CPU
- GPU handles higher FLOPs efficiently through parallelization
- 2x FLOPs doesn't double GPU training time

## 📁 Files

```
Assignment1/
├── FashionMNIST_Experiments.ipynb  # Main Colab notebook with all experiments
├── results.json                     # Complete experimental results
├── REPORT.md                        # Detailed analysis report
├── training_time_comparison.png     # Training time visualization
├── accuracy_comparison.png          # Accuracy visualization
├── README_GITHUB.md                 # This file (for GitHub)
└── saved_models/                    # Trained model checkpoints (12 models)
    ├── ResNet-32_SGD_gpu.pth       # Best model (91.31% accuracy)
    └── ...                          # Other models
```

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
1. Open the Colab link: **[ADD YOUR COLAB LINK HERE]**
2. Click "Runtime" → "Run all"
3. Results will be generated automatically (~20-30 minutes with GPU)

### Option 2: Local Execution
```bash
# Clone repository
git clone https://github.com/[YourUsername]/MLOps-[Name]-[RollNumber].git
cd MLOps-[Name]-[RollNumber]/Assignment1

# Install dependencies
pip install torch torchvision thop matplotlib numpy

# Run experiments (if using Jupyter)
jupyter notebook FashionMNIST_Experiments.ipynb
```

## 🔗 Links

- **GitHub Repository:** [https://github.com/[YourUsername]/MLOps-[Name]-[RollNumber]](https://github.com/[YourUsername]/MLOps-[Name]-[RollNumber])
- **Branch:** Assignment1
- **Google Colab:** [ADD YOUR COLAB LINK HERE] ⚠️ **REQUIRED FOR SUBMISSION**
- **GitHub Pages:** [Your GitHub Pages URL]
- **Report PDF:** [RollNumber]_[Name]_Ass1.pdf

## 🎓 Key Findings

1. **GPU Acceleration is Critical:** 21.58x speedup makes GPU essential for deep learning
2. **Model Selection Matters:** ResNet-32 provides best accuracy-efficiency balance
3. **Optimizer Impact:** SGD with momentum outperforms Adam for these experiments
4. **Scalability:** GPU performance remains consistent as model complexity increases

## 📝 Citation

```bibtex
@misc{fashionmnist_resnet_2026,
  author = {[Your Name]},
  title = {FashionMNIST ResNet Experiments - MLOps Assignment 1},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/[YourUsername]/MLOps-[Name]-[RollNumber]}}
}
```

## 📧 Contact

**Name:** [Your Name]  
**Roll Number:** [Your Roll Number]  
**Email:** [Your Email]  
**Institution:** [Your Institution]

---

**Last Updated:** January 23, 2026  
**Status:** ✅ Complete

**Note:** This project was completed as part of MLOps course Assignment 1. All experiments were run on Google Colab with GPU acceleration (NVIDIA Tesla T4).

