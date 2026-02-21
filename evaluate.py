import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import random
import os

from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

# ==========================
# Config
# ==========================
DATA_DIR   = "data/test/"
MODEL_PATH = "setA.pth"
BATCH_SIZE = 32
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Transforms
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# Dataset & DataLoader
# ==========================
dataset    = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = dataset.classes
print("Classes:", class_names)

# ==========================
# Load Model
# ==========================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print("Model Loaded Successfully!\n")

# ==========================
# Evaluation on Full Test Set
# ==========================
all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ==========================
# Overall Accuracy & F1
# ==========================
overall_acc = accuracy_score(all_labels, all_preds)
macro_f1    = f1_score(all_labels, all_preds, average="macro", zero_division=0)

print(f"Overall Accuracy : {overall_acc * 100:.2f}%")
print(f"Macro F1 Score   : {macro_f1:.4f}\n")

# ==========================
# Per-Class Accuracy
# ==========================
print("=" * 50)
print("Per-Class Accuracy")
print("=" * 50)
for cls_idx, cls_name in enumerate(class_names):
    mask      = all_labels == cls_idx
    cls_acc   = np.sum(all_preds[mask] == cls_idx) / np.sum(mask) * 100
    print(f"  Class {cls_name:>3} : {cls_acc:6.2f}%  (support: {np.sum(mask)})")

print("=" * 50)

# ==========================
# Classification Report
# ==========================
print("\nClassification Report:")
print(classification_report(
    all_labels, all_preds,
    target_names=class_names,
    zero_division=0
))

# ==========================
# Confusion Matrix Plot
# ==========================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("Confusion Matrix — Test Set", fontsize=14)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Confusion matrix saved to confusion_matrix.png\n")

# ==========================
# Single Image Inference
# ==========================
def predict_single_image(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    print(f"--- Single Image Inference ---")
    print(f"Image            : {image_path}")
    print(f"Predicted Class  : {class_names[pred.item()]}")
    print(f"Confidence       : {confidence.item()*100:.2f}%")
    print(f"All Class Probs  :")
    for i, p in enumerate(probs[0].tolist()):
        marker = " <-- predicted" if i == pred.item() else ""
        print(f"  Class {class_names[i]:>3}: {p*100:6.2f}%{marker}")

# Run inference on the specified image
predict_single_image("data/test/5/340.png")