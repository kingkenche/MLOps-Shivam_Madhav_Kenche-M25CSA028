# MLOps Assignment 1

**Student:** SHIVAM MADHAV KENCHE  
**Roll Number:** M25CSA028  
**Branch:** Assignment1

---

## Q1(a): Deep Learning Classification

### MNIST - Test Classification Accuracy (%)

| Batch Size | Optimizer | Learning Rate | ResNet-18 | ResNet-50 |
|:----------:|:---------:|:-------------:|:---------:|:---------:|
| 16 | SGD | 0.001 | **99.81** | **99.79** |
| 16 | SGD | 0.0001 | 99.75 | 99.35 |
| 16 | Adam | 0.001 | 99.23 | 96.78 |
| 16 | Adam | 0.0001 | 99.57 | 97.83 |
| 32 | SGD | 0.001 | 99.81 | 98.82 |
| 32 | SGD | 0.0001 | 99.65 | 99.01 |
| 32 | Adam | 0.001 | 99.39 | 99.09 |
| 32 | Adam | 0.0001 | 99.64 | 99.42 |

### FashionMNIST - Test Classification Accuracy (%)

| Batch Size | Optimizer | Learning Rate | ResNet-18 | ResNet-50 |
|:----------:|:---------:|:-------------:|:---------:|:---------:|
| 16 | SGD | 0.001 | 96.59 | 96.56 |
| 16 | SGD | 0.0001 | 94.41 | 95.46 |
| 16 | Adam | 0.001 | 93.34 | 91.48 |
| 16 | Adam | 0.0001 | **97.47** | 96.40 |
| 32 | SGD | 0.001 | 95.64 | **97.20** |
| 32 | SGD | 0.0001 | 96.26 | 96.44 |
| 32 | Adam | 0.001 | 93.84 | 91.51 |
| 32 | Adam | 0.0001 | 97.28 | 96.09 |

### Best Models

| Dataset | Configuration | Accuracy |
|---------|---------------|:--------:|
| MNIST | ResNet-18, BS=16, SGD, LR=0.001 | **99.81%** |
| FashionMNIST | ResNet-18, BS=16, Adam, LR=0.0001 | **97.47%** |

### Hyperparameter Variations

| Configuration | MNIST Avg | FashionMNIST Avg |
|---------------|:---------:|:----------------:|
| pin_memory=False, epochs=10 | 99.18% | 95.37% |
| pin_memory=False, epochs=2 | 98.11% | 89.16% |
| pin_memory=True, epochs=2 | 98.03% | 87.82% |
| pin_memory=True, epochs=5 | 98.77% | 91.08% |

**Total Experiments:** 128 models trained

---

## Key Findings

1. ✅ All models achieve >80% accuracy
2. ResNet-18 outperforms ResNet-50 for 28×28 images
3. SGD with LR=0.001 works best for MNIST
4. Adam with LR=0.0001 works best for FashionMNIST
5. More epochs significantly improve accuracy

---

## Training Curves

### MNIST Best Model
![MNIST Training Curves](results/mnist_training_curves.png)

### FashionMNIST Best Model
![FashionMNIST Training Curves](results/fashion_training_curves.png)

---

## Repository Structure

```
Assignment1/
├── README.md                     # This file
├── M25CSA028_SHIVAM_MADHAV_KENCHE_Ass1_Q1a.md  # Detailed report
├── notebooks/
│   └── Q1a_Submission_Colab.ipynb    # Main experiment notebook
├── models/
│   └── resnet.py                     # ResNet implementations
└── results/
    ├── mnist_q1a_results_final.csv   # MNIST results
    ├── fashion_q1a_results_final.csv # FashionMNIST results
    ├── mnist_resnet18_bs16_SGD_lr0.001_best.pth      # Best MNIST model
    ├── fashion_resnet18_bs16_Adam_lr0.0001_best.pth  # Best FashionMNIST model
    ├── mnist_training_curves.png     # MNIST training curves
    ├── fashion_training_curves.png   # FashionMNIST training curves
    └── combined_training_curves.png  # Combined comparison
```

---

## Colab Link

**[Open in Colab](https://colab.research.google.com/drive/1WIHMM9SxlzkmhT0BA6VkplWVDUBTjs7K)**

---

## How to Run

```python
# Clone repository
git clone https://github.com/kingkenche/MLOps-Shivam_Madhav_Kenche-M25CSA028.git
cd MLOps-Shivam_Madhav_Kenche-M25CSA028
git checkout Assignment1

# Or open directly in Colab
# Upload Q1a_Submission_Colab.ipynb to Google Colab
# Set Runtime > GPU
# Run all cells
```

---

**Report:** [M25CSA028_SHIVAM_MADHAV_KENCHE_Ass1.pdf](./M25CSA028_SHIVAM_MADHAV_KENCHE_Ass1.pdf)
