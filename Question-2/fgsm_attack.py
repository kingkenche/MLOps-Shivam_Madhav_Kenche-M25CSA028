"""
fgsm_attack.py
==============
Q2-(i): FGSM from scratch vs IBM ART.

Steps
-----
1. Load the best ResNet-18 checkpoint.
2. Craft adversarial examples with:
   a) FGSM implemented from scratch.
   b) FGSM via IBM ART (FastGradientMethod wrapper).
3. Sweep ε ∈ FGSM_EPSILONS and record accuracy.
4. Save side-by-side visual comparisons.
5. Log everything (metrics + images) to WandB.

Run:
    python fgsm_attack.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

# ART
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

from config import (
    DEVICE, RESNET18_CKPT, FGSM_EPSILONS, FGSM_EPS,
    WANDB_PROJECT, CIFAR10_CLASSES, RESULTS_DIR
)

# ── Helpers ───────────────────────────────────────────────────────────────
MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

def unnormalize(t: torch.Tensor) -> np.ndarray:
    """CHW tensor → HWC uint8 numpy, un-normalised."""
    img = t.cpu().numpy().transpose(1, 2, 0)
    img = img * STD + MEAN
    return np.clip(img, 0, 1)


def build_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc      = nn.Linear(model.fc.in_features, 10)
    return model.to(DEVICE)


def load_model() -> nn.Module:
    model = build_model()
    ckpt  = torch.load(RESNET18_CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint  (train acc={ckpt.get('acc', '?'):.2f}%)")
    return model


def get_test_loader(batch_size=200, n_samples=2000):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    ds = datasets.CIFAR10("data", train=False, download=True, transform=tf)
    sub = Subset(ds, list(range(n_samples)))
    return DataLoader(sub, batch_size=batch_size, shuffle=False,
                      num_workers=2, pin_memory=True)


# ── FGSM from scratch ──────────────────────────────────────────────────────
def fgsm_scratch(model: nn.Module, imgs: torch.Tensor,
                 labels: torch.Tensor, eps: float) -> torch.Tensor:
    """Single-step L∞ FGSM (no ART)."""
    # NOTE: .to(DEVICE) must come BEFORE .requires_grad_(True)
    # so the final tensor is a leaf node and receives .grad
    imgs = imgs.clone().detach().to(DEVICE)
    imgs.requires_grad_(True)
    labels = labels.to(DEVICE)
    loss = nn.CrossEntropyLoss()(model(imgs), labels)
    loss.backward()
    with torch.no_grad():
        adv = imgs + eps * imgs.grad.sign()
    return adv.detach()


@torch.no_grad()
def accuracy(model, imgs, labels):
    return model(imgs.to(DEVICE)).argmax(1).eq(labels.to(DEVICE)).float().mean().item() * 100


# ── Epsilon sweep ──────────────────────────────────────────────────────────
def epsilon_sweep(model, loader, art_classifier):
    results = {"eps": [], "clean": [], "scratch": [], "art": []}

    # Collect one full pass of clean data
    all_imgs, all_lbls = [], []
    for imgs, lbls in loader:
        all_imgs.append(imgs); all_lbls.append(lbls)
    all_imgs = torch.cat(all_imgs); all_lbls = torch.cat(all_lbls)

    clean_acc = accuracy(model, all_imgs, all_lbls)
    print(f"Clean accuracy: {clean_acc:.2f}%")

    for eps in FGSM_EPSILONS:
        # -- scratch --
        adv_scratch = fgsm_scratch(model, all_imgs, all_lbls, eps)
        sc_acc = accuracy(model, adv_scratch, all_lbls)

        # -- ART --
        art_attack = FastGradientMethod(art_classifier, eps=eps, eps_step=eps,
                                        targeted=False, batch_size=200)
        x_np  = all_imgs.numpy()
        adv_np = art_attack.generate(x_np)
        adv_art = torch.from_numpy(adv_np)
        art_acc = accuracy(model, adv_art, all_lbls)

        results["eps"].append(eps)
        results["clean"].append(clean_acc)
        results["scratch"].append(sc_acc)
        results["art"].append(art_acc)

        print(f"ε={eps:.3f}  clean={clean_acc:.1f}%  "
              f"scratch={sc_acc:.1f}%  art={art_acc:.1f}%")

    return results


# ── Visualisation ──────────────────────────────────────────────────────────
def make_comparison_figure(model, imgs, labels, art_classifier,
                           eps=FGSM_EPS, n=10):
    """Return (fig, adv_scratch_imgs, adv_art_imgs) for the first n samples."""
    imgs_sub = imgs[:n]; lbls_sub = labels[:n]

    adv_s = fgsm_scratch(model, imgs_sub, lbls_sub, eps)

    art_attack = FastGradientMethod(art_classifier, eps=eps, eps_step=eps,
                                    targeted=False, batch_size=n)
    adv_a = torch.from_numpy(art_attack.generate(imgs_sub.numpy()))

    model.eval()
    with torch.no_grad():
        pred_clean = model(imgs_sub.to(DEVICE)).argmax(1).cpu()
        pred_s     = model(adv_s.to(DEVICE)).argmax(1).cpu()
        pred_a     = model(adv_a.to(DEVICE)).argmax(1).cpu()

    fig, axes = plt.subplots(3, n, figsize=(n * 2, 7))
    row_titles = [
        "Original\n(pred: {})",
        f"FGSM Scratch ε={eps}\n(pred: {{}})",
        f"FGSM ART ε={eps}\n(pred: {{}})",
    ]
    for col in range(n):
        for row, (adv_imgs, preds) in enumerate(
                [(imgs_sub, pred_clean), (adv_s, pred_s), (adv_a, pred_a)]):
            ax = axes[row, col]
            ax.imshow(unnormalize(adv_imgs[col]))
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(
                    row_titles[row].format(CIFAR10_CLASSES[preds[col]]),
                    fontsize=8, rotation=0, ha="right", va="center"
                )
            else:
                ax.set_title(CIFAR10_CLASSES[preds[col]], fontsize=7)

    plt.suptitle(f"FGSM Comparison  (true label shown above each column)",
                 fontsize=10)
    plt.tight_layout()
    return fig, adv_s, adv_a


def plot_epsilon_curve(results: dict):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(results["eps"], results["clean"],   "g--", label="Clean")
    ax.plot(results["eps"], results["scratch"], "r-o", label="FGSM (scratch)")
    ax.plot(results["eps"], results["art"],     "b-s", label="FGSM (ART)")
    ax.set_xlabel("Epsilon (ε)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Perturbation Strength vs Accuracy Drop")
    ax.legend(); ax.grid(True)
    path = os.path.join(RESULTS_DIR, "fgsm_epsilon_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    wandb.init(project=WANDB_PROJECT, name="fgsm_comparison", config={
        "fgsm_epsilons": FGSM_EPSILONS,
        "default_eps":   FGSM_EPS,
    })

    model  = load_model()
    loader = get_test_loader()

    # Build ART classifier (needs model in eval mode)
    criterion = nn.CrossEntropyLoss()
    art_clf = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(
            ((0 - np.array([0.4914, 0.4822, 0.4465])) / np.array([0.2023, 0.1994, 0.2010])).min().item(),
            ((1 - np.array([0.4914, 0.4822, 0.4465])) / np.array([0.2023, 0.1994, 0.2010])).max().item(),
        ),
        device_type="gpu" if DEVICE.type == "cuda" else "cpu",
    )

    # ε sweep
    results = epsilon_sweep(model, loader, art_clf)

    # Log table
    table = wandb.Table(columns=["epsilon", "clean_acc", "scratch_acc", "art_acc"])
    for i in range(len(results["eps"])):
        table.add_data(results["eps"][i], results["clean"][i],
                       results["scratch"][i], results["art"][i])
    wandb.log({"fgsm_epsilon_table": table})

    # Epsilon curve
    curve_path = plot_epsilon_curve(results)
    wandb.log({"fgsm_epsilon_curve": wandb.Image(curve_path)})

    # Visual comparison — collect a batch
    all_imgs, all_lbls = [], []
    for imgs, lbls in loader:
        all_imgs.append(imgs); all_lbls.append(lbls)
        if sum(x.size(0) for x in all_imgs) >= 10:
            break
    all_imgs = torch.cat(all_imgs)[:10]
    all_lbls = torch.cat(all_lbls)[:10]

    fig, adv_s, adv_a = make_comparison_figure(
        model, all_imgs, all_lbls, art_clf, eps=FGSM_EPS, n=10)
    comp_path = os.path.join(RESULTS_DIR, "fgsm_comparison.png")
    fig.savefig(comp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Log 10 sample triplets to WandB
    wandb_imgs = []
    for i in range(10):
        true_lbl  = CIFAR10_CLASSES[all_lbls[i]]
        wandb_imgs.append(wandb.Image(
            unnormalize(all_imgs[i]),
            caption=f"[{i}] Original | true={true_lbl}"))
        wandb_imgs.append(wandb.Image(
            unnormalize(adv_s[i]),
            caption=f"[{i}] FGSM Scratch ε={FGSM_EPS}"))
        wandb_imgs.append(wandb.Image(
            unnormalize(adv_a[i]),
            caption=f"[{i}] FGSM ART ε={FGSM_EPS}"))

    wandb.log({
        "fgsm_visual_comparison": wandb.Image(comp_path),
        "fgsm_sample_images":     wandb_imgs,
    })

    # Print final summary at default ε
    idx = FGSM_EPSILONS.index(FGSM_EPS) if FGSM_EPS in FGSM_EPSILONS else -1
    if idx >= 0:
        print(f"\n=== Summary at ε={FGSM_EPS} ===")
        print(f"  Clean accuracy      : {results['clean'][idx]:.2f}%")
        print(f"  FGSM Scratch acc    : {results['scratch'][idx]:.2f}%")
        print(f"  FGSM ART acc        : {results['art'][idx]:.2f}%")

    wandb.finish()


if __name__ == "__main__":
    main()
