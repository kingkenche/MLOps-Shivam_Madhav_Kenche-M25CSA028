"""
test_all.py
===========
Evaluation script – loads saved checkpoints and prints a full report.

Run:
    python test_all.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, TensorDataset, Subset

import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    BasicIterativeMethod,
)

from config import (
    DEVICE,
    RESNET18_CKPT, DET_PGD_CKPT, DET_BIM_CKPT,
    FGSM_EPS, PGD_EPS, PGD_EPS_STEP, PGD_MAX_ITER,
    BIM_EPS, BIM_EPS_STEP, BIM_MAX_ITER,
    WANDB_PROJECT, CIFAR10_CLASSES, RESULTS_DIR,
)

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)


# ── Model builders ────────────────────────────────────────────────────────
def build_resnet18_cifar() -> nn.Module:
    m = models.resnet18(weights=None)
    m.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(m.fc.in_features, 10)
    return m


def build_resnet34_detector() -> nn.Module:
    m = models.resnet34(weights=None)
    m.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(m.fc.in_features, 2)
    return m


def load(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.to(DEVICE)


# ── Data ──────────────────────────────────────────────────────────────────
def get_test_loader(n=2000, batch_size=200):
    tf = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    ds  = datasets.CIFAR10("data", train=False, download=True, transform=tf)
    sub = Subset(ds, list(range(n)))
    return DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=2)


def make_art_clf(victim):
    clip_min = ((0 - MEAN) / STD).min().item()
    clip_max = ((1 - MEAN) / STD).max().item()
    return PyTorchClassifier(
        model=victim,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(victim.parameters(), lr=0.01),
        input_shape=(3, 32, 32), nb_classes=10,
        clip_values=(clip_min, clip_max),
        device_type="gpu" if DEVICE.type == "cuda" else "cpu",
    )


# ── Helpers ───────────────────────────────────────────────────────────────
def unnorm(t):
    return np.clip(t.cpu().numpy().transpose(1, 2, 0) * STD + MEAN, 0, 1)


@torch.no_grad()
def clf_accuracy(model, imgs, labels):
    return model(imgs.to(DEVICE)).argmax(1).eq(labels.to(DEVICE)).float().mean().item() * 100


def det_accuracy(detector, imgs, labels):
    with torch.no_grad():
        return detector(imgs.to(DEVICE)).argmax(1).eq(labels.to(DEVICE)).float().mean().item() * 100


def fgsm_scratch(model, imgs, labels, eps):
    # NOTE: .to(DEVICE) must come BEFORE .requires_grad_(True)
    imgs = imgs.clone().detach().to(DEVICE)
    imgs.requires_grad_(True)
    labels = labels.to(DEVICE)
    nn.CrossEntropyLoss()(model(imgs), labels).backward()
    with torch.no_grad():
        adv = imgs + eps * imgs.grad.sign()
    return adv.detach()


# ── Main evaluation ───────────────────────────────────────────────────────
def main():
    run = wandb.init(project=WANDB_PROJECT, name="full_evaluation")

    loader = get_test_loader()
    all_imgs, all_lbls = [], []
    for imgs, lbls in loader:
        all_imgs.append(imgs); all_lbls.append(lbls)
    all_imgs = torch.cat(all_imgs)
    all_lbls = torch.cat(all_lbls)

    # --- ResNet-18 (classifier) ---
    victim = load(build_resnet18_cifar(), RESNET18_CKPT)
    art    = make_art_clf(victim)

    clean_acc = clf_accuracy(victim, all_imgs, all_lbls)

    # FGSM scratch
    adv_fgsm_s = fgsm_scratch(victim, all_imgs, all_lbls, FGSM_EPS)
    fgsm_s_acc = clf_accuracy(victim, adv_fgsm_s, all_lbls)

    # FGSM ART
    fgsm_art    = FastGradientMethod(art, eps=FGSM_EPS, eps_step=FGSM_EPS, targeted=False, batch_size=200)
    adv_fgsm_a  = torch.from_numpy(fgsm_art.generate(all_imgs.numpy()))
    fgsm_a_acc  = clf_accuracy(victim, adv_fgsm_a, all_lbls)

    # PGD
    pgd_att  = ProjectedGradientDescent(art, eps=PGD_EPS, eps_step=PGD_EPS_STEP,
                                         max_iter=PGD_MAX_ITER, targeted=False, batch_size=200)
    adv_pgd  = torch.from_numpy(pgd_att.generate(all_imgs.numpy()))
    pgd_acc  = clf_accuracy(victim, adv_pgd, all_lbls)

    # BIM
    bim_att  = BasicIterativeMethod(art, eps=BIM_EPS, eps_step=BIM_EPS_STEP,
                                     max_iter=BIM_MAX_ITER, targeted=False, batch_size=200)
    adv_bim  = torch.from_numpy(bim_att.generate(all_imgs.numpy()))
    bim_acc  = clf_accuracy(victim, adv_bim, all_lbls)

    # --- Detectors ---
    det_pgd = load(build_resnet34_detector(), DET_PGD_CKPT)
    det_bim = load(build_resnet34_detector(), DET_BIM_CKPT)

    n = len(all_imgs)
    # PGD detector: half clean (0), half adv (1)
    X_det_pgd = torch.cat([all_imgs[:n//2], adv_pgd[:n//2]])
    y_det_pgd = torch.cat([torch.zeros(n//2, dtype=torch.long),
                            torch.ones( n//2, dtype=torch.long)])
    pgd_det_acc = det_accuracy(det_pgd, X_det_pgd, y_det_pgd)

    X_det_bim = torch.cat([all_imgs[:n//2], adv_bim[:n//2]])
    y_det_bim = torch.cat([torch.zeros(n//2, dtype=torch.long),
                            torch.ones( n//2, dtype=torch.long)])
    bim_det_acc = det_accuracy(det_bim, X_det_bim, y_det_bim)

    # --- Print report ---
    print("\n" + "="*55)
    print("        ADVERSARIAL ROBUSTNESS EVALUATION REPORT")
    print("="*55)
    print(f"  Clean accuracy            : {clean_acc:.2f}%")
    print(f"  FGSM (scratch) ε={FGSM_EPS:.2f}    : {fgsm_s_acc:.2f}%")
    print(f"  FGSM (ART)     ε={FGSM_EPS:.2f}    : {fgsm_a_acc:.2f}%")
    print(f"  PGD            ε={PGD_EPS:.2f}    : {pgd_acc:.2f}%")
    print(f"  BIM            ε={BIM_EPS:.2f}    : {bim_acc:.2f}%")
    print("-"*55)
    print(f"  Detection (PGD detector)  : {pgd_det_acc:.2f}%")
    print(f"  Detection (BIM detector)  : {bim_det_acc:.2f}%")
    print("="*55)

    # --- Log to WandB ---
    summary_tbl = wandb.Table(
        columns=["Metric", "Value (%)"],
        data=[
            ["Clean accuracy",             round(clean_acc, 2)],
            [f"FGSM Scratch ε={FGSM_EPS}", round(fgsm_s_acc, 2)],
            [f"FGSM ART ε={FGSM_EPS}",     round(fgsm_a_acc, 2)],
            [f"PGD ε={PGD_EPS}",            round(pgd_acc, 2)],
            [f"BIM ε={BIM_EPS}",            round(bim_acc, 2)],
            ["PGD Detection Acc",           round(pgd_det_acc, 2)],
            ["BIM Detection Acc",           round(bim_det_acc, 2)],
        ]
    )
    run.log({"evaluation_summary": summary_tbl})

    # Log 10 samples per attack (required by assignment)
    def log_attack_samples(name, clean, adv, n=10):
        imgs = []
        for i in range(n):
            imgs.append(wandb.Image(unnorm(clean[i]), caption=f"[{i}] Clean"))
            imgs.append(wandb.Image(unnorm(adv[i]),   caption=f"[{i}] {name}"))
        run.log({f"samples_{name}": imgs})

    log_attack_samples("FGSM_Scratch", all_imgs, adv_fgsm_s)
    log_attack_samples("FGSM_ART",     all_imgs, adv_fgsm_a)
    log_attack_samples("PGD",          all_imgs, adv_pgd)
    log_attack_samples("BIM",          all_imgs, adv_bim)

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    labels  = ["Clean", "FGSM\nScratch", "FGSM\nART", "PGD", "BIM",
               "Det-PGD", "Det-BIM"]
    values  = [clean_acc, fgsm_s_acc, fgsm_a_acc, pgd_acc, bim_acc,
               pgd_det_acc, bim_det_acc]
    colors  = ["#2ecc71","#e74c3c","#e74c3c","#c0392b","#c0392b",
               "#3498db","#2980b9"]
    ax.bar(labels, values, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Evaluation Summary")
    ax.axhline(70, color="gray", linestyle="--", linewidth=1, label="70% threshold")
    ax.legend()
    for i, v in enumerate(values):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    path = os.path.join(RESULTS_DIR, "evaluation_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    run.log({"evaluation_bar_chart": wandb.Image(path)})

    run.finish()


if __name__ == "__main__":
    main()
