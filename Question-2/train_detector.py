"""
train_detector.py
=================
Q2-(ii): Train ResNet-34-based binary detectors (clean vs adversarial).

Two separate detectors are trained:
  • Detector-PGD  – trained on PGD adversarial images
  • Detector-BIM  – trained on BIM adversarial images

Run:
    python train_detector.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms, models
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ART
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent, BasicIterativeMethod

from config import (
    DEVICE, RESNET18_CKPT,
    DET_PGD_EPS, DET_PGD_EPS_STEP, DET_PGD_MAX_ITER,
    DET_BIM_EPS, DET_BIM_EPS_STEP, DET_BIM_MAX_ITER,
    DET_EPOCHS, DET_BATCH_SIZE, DET_LR, DET_WEIGHT_DECAY, DET_N_SAMPLES,
    DET_PGD_CKPT, DET_BIM_CKPT,
    WANDB_PROJECT, CIFAR10_CLASSES, RESULTS_DIR
)

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)


# ── Load victim ResNet-18 ─────────────────────────────────────────────────
def load_victim() -> nn.Module:
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(model.fc.in_features, 10)
    ckpt = torch.load(RESNET18_CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.to(DEVICE)


# ── Build ART wrapper ─────────────────────────────────────────────────────
def make_art_classifier(victim: nn.Module) -> PyTorchClassifier:
    clip_min = ((0 - MEAN) / STD).min().item()
    clip_max = ((1 - MEAN) / STD).max().item()
    return PyTorchClassifier(
        model=victim,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(victim.parameters(), lr=0.01),
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(clip_min, clip_max),
        device_type="gpu" if DEVICE.type == "cuda" else "cpu",
    )


# ── Generate adversarial dataset ──────────────────────────────────────────
def generate_dataset(attack_name: str, n_samples: int = 10000):
    """
    Returns X (torch float32, N×3×32×32), y (torch long, N).
    y=0 → clean, y=1 → adversarial
    """
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    cifar = datasets.CIFAR10("data", train=True, download=True, transform=tf)
    loader = DataLoader(cifar, batch_size=256, shuffle=True,
                        num_workers=4, pin_memory=True)

    victim   = load_victim()
    art_clf  = make_art_classifier(victim)

    if attack_name == "pgd":
        attack = ProjectedGradientDescent(
            art_clf, eps=DET_PGD_EPS, eps_step=DET_PGD_EPS_STEP,
            max_iter=DET_PGD_MAX_ITER, targeted=False, batch_size=256,
        )
        atk_label = "PGD"
    elif attack_name == "bim":
        attack = BasicIterativeMethod(
            art_clf, eps=DET_BIM_EPS, eps_step=DET_BIM_EPS_STEP,
            max_iter=DET_BIM_MAX_ITER, targeted=False, batch_size=256,
        )
        atk_label = "BIM"
    else:
        raise ValueError(f"Unknown attack: {attack_name}")

    clean_list, adv_list = [], []
    collected = 0
    half = n_samples // 2

    print(f"Generating {atk_label} adversarial examples …")
    for imgs, _ in tqdm(loader, desc=atk_label):
        if collected >= half:
            break
        batch = min(imgs.size(0), half - collected)
        imgs_np = imgs[:batch].numpy()
        adv_np  = attack.generate(imgs_np)
        clean_list.append(imgs[:batch])
        adv_list.append(torch.from_numpy(adv_np))
        collected += batch

    X_clean = torch.cat(clean_list)         # label 0
    X_adv   = torch.cat(adv_list)           # label 1
    X = torch.cat([X_clean, X_adv])
    y = torch.cat([torch.zeros(len(X_clean), dtype=torch.long),
                   torch.ones( len(X_adv),   dtype=torch.long)])
    return X, y, X_clean, X_adv


# ── ResNet-34 detector ────────────────────────────────────────────────────
def build_detector() -> nn.Module:
    # Use pretrained weights to prevent the model from getting stuck at 50%
    # (The adversarial perturbation is subtle, so learning from scratch is hard)
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(model.fc.in_features, 2)   # binary
    return model.to(DEVICE)


# ── Train / eval ──────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            if train: optimizer.zero_grad()
            out  = model(X)
            loss = criterion(out, y)
            if train:
                loss.backward(); optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct    += out.argmax(1).eq(y).sum().item()
            total      += y.size(0)
    return total_loss / total, correct / total * 100


def train_detector(attack_name: str, run_name: str, ckpt_path: str,
                   X, y, wandb_run):
    """Train a ResNet-34 binary detector on pre-generated (X, y) tensors."""
    ds  = TensorDataset(X, y)
    n_val  = int(len(ds) * 0.15)
    n_test = int(len(ds) * 0.15)
    n_train= len(ds) - n_val - n_test
    tr_ds, va_ds, te_ds = random_split(
        ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    tr_loader = DataLoader(tr_ds, batch_size=DET_BATCH_SIZE, shuffle=True,  num_workers=4)
    va_loader = DataLoader(va_ds, batch_size=DET_BATCH_SIZE, shuffle=False, num_workers=4)
    te_loader = DataLoader(te_ds, batch_size=DET_BATCH_SIZE, shuffle=False, num_workers=4)

    model     = build_detector()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=DET_LR, weight_decay=DET_WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=DET_EPOCHS)

    best_val = 0.0
    print(f"\n=== Training {run_name} ===")
    for epoch in tqdm(range(1, DET_EPOCHS + 1), desc=run_name):
        tr_loss, tr_acc = run_epoch(model, tr_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, va_loader, criterion)
        scheduler.step()

        wandb_run.log({
            f"{attack_name}/epoch":     epoch,
            f"{attack_name}/train_loss": tr_loss,
            f"{attack_name}/train_acc":  tr_acc,
            f"{attack_name}/val_loss":   va_loss,
            f"{attack_name}/val_acc":    va_acc,
        })

        if va_acc > best_val:
            best_val = va_acc
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_acc": best_val}, ckpt_path)

    # Test
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    _, test_acc = run_epoch(model, te_loader, criterion)
    print(f"{run_name} → best_val={best_val:.2f}%  test_acc={test_acc:.2f}%")

    wandb_run.summary[f"{attack_name}_test_acc"] = test_acc
    return model, test_acc


# ── Unnormalize helper ────────────────────────────────────────────────────
def unnorm(t):
    img = t.cpu().numpy().transpose(1, 2, 0) * STD + MEAN
    return np.clip(img, 0, 1)


# ── Sample visualisation + WandB logging ─────────────────────────────────
def log_samples(attack_name, X_clean, X_adv, wandb_run, n=10):
    imgs = []
    for i in range(n):
        imgs.append(wandb.Image(unnorm(X_clean[i]),
                                caption=f"[{i}] Clean"))
        imgs.append(wandb.Image(unnorm(X_adv[i]),
                                caption=f"[{i}] Adv ({attack_name.upper()})"))
    wandb_run.log({f"{attack_name}_samples": imgs})

    # Matplotlib grid
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 5))
    for i in range(n):
        axes[0, i].imshow(unnorm(X_clean[i])); axes[0, i].axis("off")
        axes[1, i].imshow(unnorm(X_adv[i]));   axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Clean", fontsize=9)
    axes[1, 0].set_ylabel(f"Adv\n({attack_name.upper()})", fontsize=9)
    plt.suptitle(f"{attack_name.upper()} attack samples", fontsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{attack_name}_samples.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    wandb_run.log({f"{attack_name}_sample_grid": wandb.Image(path)})


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    run = wandb.init(project=WANDB_PROJECT, name="adversarial_detectors", config={
        "pgd_eps": DET_PGD_EPS, "pgd_eps_step": DET_PGD_EPS_STEP, "pgd_iter": DET_PGD_MAX_ITER,
        "bim_eps": DET_BIM_EPS, "bim_eps_step": DET_BIM_EPS_STEP, "bim_iter": DET_BIM_MAX_ITER,
        "det_epochs": DET_EPOCHS, "det_lr": DET_LR, "det_n_samples": DET_N_SAMPLES,
    })

    # --- PGD detector ---
    X_pgd, y_pgd, X_clean_pgd, X_adv_pgd = generate_dataset("pgd", DET_N_SAMPLES)
    log_samples("pgd", X_clean_pgd, X_adv_pgd, run)
    _, pgd_test_acc = train_detector(
        "pgd", "Detector-PGD", DET_PGD_CKPT,
        X_pgd, y_pgd, run
    )

    # --- BIM detector ---
    X_bim, y_bim, X_clean_bim, X_adv_bim = generate_dataset("bim", DET_N_SAMPLES)
    log_samples("bim", X_clean_bim, X_adv_bim, run)
    _, bim_test_acc = train_detector(
        "bim", "Detector-BIM", DET_BIM_CKPT,
        X_bim, y_bim, run
    )

    # --- Comparison table ---
    tbl = wandb.Table(columns=["Attack", "Test Detection Acc (%)"])
    tbl.add_data("PGD", round(pgd_test_acc, 2))
    tbl.add_data("BIM", round(bim_test_acc, 2))
    run.log({"detector_comparison": tbl})

    print("\n====== Final Detection Accuracy ======")
    print(f"  PGD detector : {pgd_test_acc:.2f}%")
    print(f"  BIM detector : {bim_test_acc:.2f}%")

    run.finish()


if __name__ == "__main__":
    main()
