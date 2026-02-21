# M25CSA028 — ResNet-18 Image Classification

End-to-end ResNet-18 training and evaluation pipeline for a 10-class image classification task.

---

## 📁 Project Structure

```
M25CSA028/
├── train.py                        # Training script (ResNet-18, Adam, CrossEntropy)
├── evaluate.py                     # Evaluation script (accuracy, F1, confusion matrix)
├── M25CSA028_evaluate_colab.ipynb  # Google Colab notebook (GPU evaluation)
├── Dockerfile                      # Docker container for reproducible environment
├── requirements.txt                # Python dependencies
├── setA.pth                        # Trained model weights
├── results/
│   ├── classification_report.txt   # Saved classification report
│   ├── confusion_matrix.png        # Confusion matrix heatmap
│   └── single_image_result.png     # Single image inference output
└── data/
    ├── train/                      # Training images (class subfolders)
    └── test/                       # Test images (class subfolders)
```

---

## ⚙️ Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### requirements.txt
```
torch
torchvision
numpy
scikit-learn
Pillow
matplotlib
seaborn
```

---

## 🏋️ Training

```bash
python train.py
```

| Parameter   | Value        |
|-------------|--------------|
| Model       | ResNet-18    |
| Optimizer   | Adam (lr=1e-3) |
| Loss        | CrossEntropyLoss |
| Batch Size  | 32           |
| Epochs      | 3            |
| Input Size  | 224 × 224    |
| Classes     | 10           |

---

## 📊 Evaluation Results

Evaluated on **5,000 test images** using pretrained weights `setA.pth`.

```bash
python evaluate.py
```

### Overall Metrics

| Metric            | Value  |
|-------------------|--------|
| Overall Accuracy  | **89.00%** |
| Macro F1 Score    | **0.8413** |
| Weighted F1 Score | **0.85**   |

---

### Per-Class Accuracy

| Class | Accuracy | Support |
|-------|----------|---------|
| 0     | 100.00%  | 490     |
| 1     | 99.12%   | 568     |
| 2     | 97.87%   | 516     |
| 3     | 97.82%   | 505     |
| 4     | 95.11%   | 491     |
| **5** | **0.00%** ⚠️ | 445 |
| 6     | 96.45%   | 479     |
| 7     | 99.22%   | 514     |
| 8     | 94.05%   | 487     |
| 9     | 98.81%   | 505     |

> ⚠️ **Class 5 Accuracy is 0%** — the model predicted zero samples as class 5. This is likely due to very few training epochs (3) causing bias toward other classes.

---

### Classification Report

```
              precision    recall  f1-score   support

           0       0.90      1.00      0.94       490
           1       0.98      0.99      0.99       568
           2       0.98      0.98      0.98       516
           3       0.71      0.98      0.82       505
           4       0.99      0.95      0.97       491
           5       0.00      0.00      0.00       445
           6       0.99      0.96      0.98       479
           7       0.97      0.99      0.98       514
           8       0.87      0.94      0.90       487
           9       0.74      0.99      0.84       505

    accuracy                           0.89      5000
   macro avg       0.81      0.88      0.84      5000
weighted avg       0.82      0.89      0.85      5000
```

---

### Single Image Inference

```
Image           : data/test/0/884.png
Predicted Class : 0
Confidence      : 99.99%
```

---

## 🐳 Docker

```bash
# Build image
docker build -t m25csa028-ml .

# Run training + evaluation
docker run m25csa028-ml

# Run only evaluation
docker run m25csa028-ml python evaluate.py
```

> **GPU Support:** Replace base image in Dockerfile with `pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime` and add `--gpus all` to `docker run`.

---

## ☁️ Google Colab (No GPU locally)

Open `M25CSA028_evaluate_colab.ipynb` in Colab with **T4 GPU** runtime:

1. Upload `setA.pth` and `data/` to `MyDrive/M25CSA028/`
2. Set Runtime → **T4 GPU**
3. Fill in GitHub credentials in Step 1
4. Run all cells — results are saved to `results/` and pushed to GitHub

---

## 📌 Notes

- Model weights (`setA.pth`) are excluded from git via `.gitignore`
- All evaluation outputs are saved to `results/`
- Class 5 issue: increase training epochs or apply class-weighted loss to fix
