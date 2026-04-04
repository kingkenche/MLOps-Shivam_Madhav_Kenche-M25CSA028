"""
utils.py - Training loop, evaluation, gradient logging, and plotting helpers
"""

import os
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from tqdm import tqdm


# ---------------------------------------------------------------------------
# One epoch: train
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, scheduler, device,
                    epoch, log_grads=False, wandb_run=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"Train E{epoch}")):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # timm models may return a dict when using features; handle both
        if isinstance(outputs, dict):
            outputs = outputs["logits"]

        loss = criterion(outputs, labels)
        loss.backward()

        # --- log LoRA gradient norms (optional, sampled every 50 steps) ---
        if log_grads and wandb_run and batch_idx % 50 == 0:
            grad_log = {}
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if "lora_" in name:
                        grad_log[f"grad_norm/{name}"] = param.grad.norm().item()
            if grad_log:
                wandb_run.log({**grad_log, "batch": batch_idx, "epoch": epoch},
                              commit=False)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, criterion, device, return_per_class=False,
             num_classes=100):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    class_correct = torch.zeros(num_classes)
    class_total   = torch.zeros(num_classes)

    for images, labels in tqdm(loader, desc="Eval"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if isinstance(outputs, dict):
            outputs = outputs["logits"]

        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

        if return_per_class:
            for c in range(num_classes):
                mask = (labels == c)
                class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                class_total[c]   += mask.sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    if return_per_class:
        per_class_acc = (class_correct / class_total.clamp(min=1)) * 100
        return avg_loss, accuracy, per_class_acc.numpy()

    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Training orchestrator
# ---------------------------------------------------------------------------

def run_training(
    model,
    train_loader,
    val_loader,
    config: dict,
    device,
    wandb_run=None,
    log_grads: bool = False,
    save_path: str  = "best_model.pth",
):
    """
    Full training loop.

    config keys expected:
      epochs, lr, weight_decay, warmup_epochs
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-4),
    )

    total_steps   = config["epochs"] * len(train_loader)
    warmup_steps  = config.get("warmup_epochs", 1) * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(1, config["epochs"] + 1):
        t_loss, t_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler,
            device, epoch, log_grads=log_grads, wandb_run=wandb_run
        )
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {t_loss:.4f}  Train Acc: {t_acc:.2f}%  |  "
              f"Val Loss: {v_loss:.4f}  Val Acc: {v_acc:.2f}%")

        if wandb_run:
            wandb_run.log({
                "epoch":      epoch,
                "train/loss": t_loss,
                "train/acc":  t_acc,
                "val/loss":   v_loss,
                "val/acc":    v_acc,
                "lr":         scheduler.get_last_lr()[0],
            })

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model  (val acc = {best_acc:.2f}%)")

    print(f"\nTraining complete. Best Val Acc: {best_acc:.2f}%")
    return history, best_acc


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_train_val_curves(history: dict, title: str, save_dir: str = "."):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   marker="s")
    axes[0].set_title(f"{title} – Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker="o")
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   marker="s")
    axes[1].set_title(f"{title} – Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{title.replace(' ', '_')}_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved curves → {path}")
    return path


def plot_classwise_histogram(per_class_acc: np.ndarray,
                             class_names: list,
                             title: str,
                             save_dir: str = "."):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(20, 5))
    indices = np.argsort(per_class_acc)
    sorted_acc   = per_class_acc[indices]
    sorted_names = [class_names[i] for i in indices]

    colors = ["#d62728" if a < 50 else "#2ca02c" for a in sorted_acc]
    ax.bar(range(len(sorted_acc)), sorted_acc, color=colors)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=90, fontsize=7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.axhline(y=sorted_acc.mean(), color="navy", linestyle="--",
               label=f"Mean: {sorted_acc.mean():.1f}%")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, f"{title.replace(' ', '_')}_classwise.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved class-wise histogram → {path}")
    return path
