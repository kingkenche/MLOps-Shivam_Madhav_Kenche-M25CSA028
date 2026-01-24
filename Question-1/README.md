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
![MNIST Training Curves](../results/mnist_training_curves.png)

### FashionMNIST Best Model
![FashionMNIST Training Curves](../results/fashion_training_curves.png)

---

## Confusion Matrix Analysis

### MNIST - Best Model Confusion Matrix
**Model:** ResNet-18, Batch Size=16, SGD, LR=0.001 (99.81% accuracy)

![MNIST Confusion Matrix](../results/mnist_confusion_matrix.png)

#### MNIST Classification Report

| Class | Precision | Recall | F1-Score | Support |
|:-----:|:---------:|:------:|:--------:|:-------:|
| 0 | 1.00 | 1.00 | 1.00 | 1,409 |
| 1 | 0.99 | 1.00 | 1.00 | 1,492 |
| 2 | 1.00 | 1.00 | 1.00 | 1,433 |
| 3 | 1.00 | 1.00 | 1.00 | 1,420 |
| 4 | 1.00 | 1.00 | 1.00 | 1,412 |
| 5 | 1.00 | 1.00 | 1.00 | 1,204 |
| 6 | 1.00 | 1.00 | 1.00 | 1,328 |
| 7 | 1.00 | 1.00 | 1.00 | 1,508 |
| 8 | 1.00 | 1.00 | 1.00 | 1,347 |
| 9 | 1.00 | 1.00 | 1.00 | 1,447 |
| **Overall** | **1.00** | **1.00** | **1.00** | **14,000** |

#### MNIST Per-Class Accuracy
- 0: 99.93% | 1: 99.93% | 2: 99.86% | 3: 99.86% | 4: 99.93%
- 5: 99.92% | 6: 99.92% | 7: 99.93% | 8: 99.93% | 9: 99.93%

---

### FashionMNIST - Best Model Confusion Matrix
**Model:** ResNet-18, Batch Size=16, Adam, LR=0.0001 (97.47% accuracy)

![FashionMNIST Confusion Matrix](../results/fashionmnist_confusion_matrix.png)

#### FashionMNIST Classification Report

| Class | Precision | Recall | F1-Score | Support |
|:-----:|:---------:|:------:|:--------:|:-------:|
| T-shirt/top | 0.96 | 0.94 | 0.95 | 1,408 |
| Trouser | 1.00 | 0.99 | 1.00 | 1,371 |
| Pullover | 0.95 | 0.97 | 0.96 | 1,430 |
| Dress | 0.96 | 0.99 | 0.97 | 1,402 |
| Coat | 0.98 | 0.96 | 0.97 | 1,403 |
| Sandal | 1.00 | 1.00 | 1.00 | 1,368 |
| Shirt | 0.92 | 0.93 | 0.93 | 1,348 |
| Sneaker | 0.98 | 1.00 | 0.99 | 1,408 |
| Bag | 1.00 | 1.00 | 1.00 | 1,430 |
| Ankle boot | 1.00 | 0.99 | 0.99 | 1,432 |
| **Overall** | **0.97** | **0.97** | **0.97** | **14,000** |

#### FashionMNIST Per-Class Accuracy
- T-shirt/top: 93.75% | Trouser: 99.12% | Pullover: 96.78% | Dress: 98.57% | Coat: 95.65%
- Sandal: 99.93% | Shirt: 93.25% | Sneaker: 99.50% | Bag: 99.72% | Ankle boot: 98.53%

**Key Observations:**
- **Most Challenging Classes:** Shirt (93.25%) and T-shirt/top (93.75%) - similar visual features cause confusion
- **Best Performing Classes:** Sandal (99.93%), Bag (99.72%), Sneaker (99.50%) - distinct shapes are easier to classify
- **Class Confusion:** Shirt ↔ T-shirt/top and Pullover ↔ Coat show highest confusion rates

---

### Side-by-Side Comparison
![Confusion Matrices Comparison](../results/confusion_matrices_comparison.png)

---

## Repository Structure

```
Question-1/
├── README.md                     # This file
├── M25CSA028_SHIVAM_MADHAV_KENCHE_Ass1.pdf  # Detailed report
├── Q1_Submission_Colab.ipynb     # Q1(a) + Q1(b) experiment notebook
├── Assignment1_2.ipynb           # Q1(b) SVM experiments
└── index.html                    # GitHub Pages

Parent Directory:
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
    ├── combined_training_curves.png  # Combined comparison
    ├── mnist_confusion_matrix.png    # MNIST confusion matrix
    ├── fashionmnist_confusion_matrix.png  # FashionMNIST confusion matrix
    └── confusion_matrices_comparison.png  # Side-by-side comparison
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
git checkout Assignment-1

# Or open directly in Colab
# Upload Q1_Submission_Colab.ipynb to Google Colab
# Set Runtime > GPU
# Run all cells (includes both Q1(a) ResNet and Q1(b) SVM)
```

---

**Report:** [M25CSA028_SHIVAM_MADHAV_KENCHE_Ass1.pdf](./M25CSA028_SHIVAM_MADHAV_KENCHE_Ass1.pdf)
