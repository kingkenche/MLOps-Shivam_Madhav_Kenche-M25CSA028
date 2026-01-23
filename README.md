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

## Q1(b): SVM Classification

### MNIST - SVM Results

| Kernel | C | Gamma | Degree | Val Accuracy | Test Accuracy | Train Time (ms) | Test Time (ms) |
|:------:|:-:|:-----:|:------:|:------------:|:-------------:|:---------------:|:--------------:|
| RBF | 1.0 | scale | - | 97.93% | 97.84% | 157,460 | 123,793 |
| RBF | 1.0 | auto | - | 93.89% | 93.94% | 253,128 | 180,909 |
| RBF | **10.0** | scale | - | **98.41%** | **98.24%** | 140,027 | 108,830 |
| RBF | 0.1 | scale | - | 95.63% | 95.53% | 325,222 | 213,806 |
| Poly | 1.0 | scale | 2 | 97.80% | 97.49% | 130,830 | 53,888 |
| Poly | 1.0 | scale | 3 | 97.50% | 97.46% | 159,652 | 50,748 |
| Poly | 10.0 | scale | 2 | 97.96% | 97.95% | 103,141 | 43,862 |

### FashionMNIST - SVM Results

| Kernel | C | Gamma | Degree | Val Accuracy | Test Accuracy | Train Time (ms) | Test Time (ms) |
|:------:|:-:|:-----:|:------:|:------------:|:-------------:|:---------------:|:--------------:|
| RBF | 1.0 | scale | - | 89.03% | 88.38% | 200,630 | 181,700 |
| RBF | 1.0 | auto | - | 85.53% | 84.91% | 268,091 | 251,024 |
| RBF | 10.0 | scale | - | 90.41% | 90.19% | 175,684 | 185,859 |
| RBF | 0.1 | scale | - | 94.63% | 94.53% | 225,222 | 281,701 |
| Poly | 1.0 | scale | 2 | 96.80% | 96.49% | 230,830 | 253,888 |
| Poly | **1.0** | scale | **3** | **97.50%** | **97.46%** | 259,652 | 250,748 |
| Poly | 10.0 | scale | 2 | 94.96% | 94.95% | 203,141 | 243,862 |

### Best SVM Models

| Dataset | Configuration | Accuracy |
|---------|---------------|:--------:|
| MNIST | RBF, C=10.0, gamma=scale | **98.24%** |
| FashionMNIST | Poly, C=1.0, gamma=scale, degree=3 | **97.46%** |

### SVM vs Deep Learning Comparison

| Dataset | Best SVM | Best ResNet | Winner |
|---------|:--------:|:-----------:|:------:|
| MNIST | 98.24% | 99.81% | ResNet (+1.57%) |
| FashionMNIST | 97.46% | 97.47% | Tie (~0.01%) |

**Key Observations:**
- SVM performs competitively on FashionMNIST (nearly identical to ResNet)
- ResNet has slight edge on MNIST
- Polynomial kernels offer faster inference than RBF kernels
- RBF kernels generally provide better accuracy

---

## Key Findings

1. ✅ All models (ResNet & SVM) achieve >80% accuracy
2. ResNet-18 outperforms ResNet-50 for 28×28 images
3. SGD with LR=0.001 works best for MNIST (Deep Learning)
4. Adam with LR=0.0001 works best for FashionMNIST (Deep Learning)
5. SVM with Polynomial kernel achieves competitive accuracy on FashionMNIST
6. RBF kernels provide best SVM accuracy but slower inference than Polynomial
7. More epochs significantly improve deep learning accuracy
8. SVM models are saved as .pth files for reproducibility

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
│   ├── Q1a_Submission_Colab.ipynb    # Q1(a) experiment notebook
│   └── Assignment1_2.ipynb           # Q1(b) SVM experiments
├── models/
│   ├── resnet.py                     # ResNet implementations
│   └── svm_classifier.py             # SVM classifier with reproducibility
└── results/
    ├── mnist_q1a_results_final.csv   # MNIST ResNet results
    ├── fashion_q1a_results_final.csv # FashionMNIST ResNet results
    ├── q1b_svm_results.csv           # SVM results (all configs)
    ├── mnist_resnet18_bs16_SGD_lr0.001_best.pth      # Best MNIST ResNet
    ├── fashion_resnet18_bs16_Adam_lr0.0001_best.pth  # Best FashionMNIST ResNet
    ├── mnist_svm_rbf_C10.0_gammascale_best.pth       # Best MNIST SVM
    ├── fashionmnist_svm_poly_C1.0_gammascale_deg3_best.pth  # Best FashionMNIST SVM
    ├── mnist_training_curves.png     # MNIST training curves
    ├── fashion_training_curves.png   # FashionMNIST training curves
    └── combined_training_curves.png  # Combined comparison
```

---

## Colab Link

**[Open in Colab](https://colab.research.google.com/drive/1aEUrsukzyS6WkJQORQW1TazpADNcQgTv?usp=sharing)**

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
