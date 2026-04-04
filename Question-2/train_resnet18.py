"""
train_resnet18.py
=================
Train a non-pretrained ResNet-18 on CIFAR-10 from scratch.
Target: ≥ 72 % test-set accuracy.

Run:
    python train_resnet18.py
"""

import os
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DEVICE, RESNET18_EPOCHS, RESNET18_BATCH_SIZE,
    RESNET18_LR, RESNET18_MOMENTUM, RESNET18_WEIGHT_DECAY,
    RESNET18_CKPT, WANDB_PROJECT, CIFAR10_CLASSES
)

# ── Data ──────────────────────────────────────────────────────────────────
def get_loaders(batch_size: int):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10("data", train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10("data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, test_loader


# ── Model ─────────────────────────────────────────────────────────────────
def build_model() -> nn.Module:
    model = models.resnet18(weights=None)        # non-pretrained
    # Adapt for CIFAR-10 (32×32 images)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(model.fc.in_features, 10)
    return model.to(DEVICE)


# ── Train / Eval loops ────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total * 100


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * labels.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total * 100


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    wandb.init(project=WANDB_PROJECT, name="resnet18_training", config={
        "epochs":       RESNET18_EPOCHS,
        "batch_size":   RESNET18_BATCH_SIZE,
        "lr":           RESNET18_LR,
        "momentum":     RESNET18_MOMENTUM,
        "weight_decay": RESNET18_WEIGHT_DECAY,
    })

    train_loader, test_loader = get_loaders(RESNET18_BATCH_SIZE)
    model     = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=RESNET18_LR,
                          momentum=RESNET18_MOMENTUM,
                          weight_decay=RESNET18_WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=RESNET18_EPOCHS)

    best_acc = 0.0
    print(f"Training ResNet-18 on CIFAR-10  |  device={DEVICE}")

    for epoch in tqdm(range(1, RESNET18_EPOCHS + 1), desc="Epoch"):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = evaluate(model, test_loader, criterion)
        scheduler.step()

        wandb.log({
            "epoch":      epoch,
            "train/loss": tr_loss, "train/acc": tr_acc,
            "val/loss":   va_loss, "val/acc":   va_acc,
            "lr":         scheduler.get_last_lr()[0],
        })

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "acc": best_acc}, RESNET18_CKPT)
            print(f"  ✓ epoch {epoch:3d}  val_acc={va_acc:.2f}%  (new best)")

    print(f"\nBest test accuracy: {best_acc:.2f}%")
    wandb.summary["best_test_acc"] = best_acc
    wandb.finish()


if __name__ == "__main__":
    main()
