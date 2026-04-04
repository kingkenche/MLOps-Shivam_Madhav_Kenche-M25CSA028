"""
train_q1.py
===========
Q1: Fine-tune ViT-S on CIFAR-100.

  Mode 1 – baseline:  only the classification head is trainable (no LoRA)
  Mode 2 – lora:      LoRA injected into Q/K/V, head trainable

Usage (inside Docker):
    # Baseline
    python train_q1.py --mode baseline --epochs 10 --lr 1e-3

    # LoRA experiment
    python train_q1.py --mode lora --rank 4 --alpha 4 --dropout 0.1 \
                       --epochs 10 --lr 1e-3

    # LoRA with optional partial-freeze (Q3 bonus)
    python train_q1.py --mode partial_lora --rank 4 --alpha 4 --dropout 0.1 \
                       --freeze_blocks 6 --epochs 10 --lr 1e-3
"""

import argparse
import os
import torch
import wandb

from dataset import get_dataloaders, CIFAR100_CLASSES
from model   import get_baseline_vit, get_lora_vit, get_partial_lora_vit, count_trainable_params
from utils   import run_training, plot_train_val_curves, plot_classwise_histogram, evaluate
import torch.nn as nn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Q1 – ViT-S CIFAR-100 fine-tuning")
    p.add_argument("--mode",    type=str, default="baseline",
                   choices=["baseline", "lora", "partial_lora"])
    p.add_argument("--rank",    type=int,   default=4)
    p.add_argument("--alpha",   type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--freeze_blocks", type=int, default=6,
                   help="Only for partial_lora mode")
    p.add_argument("--epochs",  type=int,   default=10)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--batch",   type=int,   default=64)
    p.add_argument("--workers", type=int,   default=4)
    p.add_argument("--data_dir",type=str,   default="./data")
    p.add_argument("--save_dir",type=str,   default="./checkpoints")
    p.add_argument("--wandb_project", type=str, default="vit-cifar100-lora")
    p.add_argument("--wandb_entity",  type=str, default=None)
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Experiment name
    # ------------------------------------------------------------------
    if args.mode == "baseline":
        exp_name = "baseline_no_lora"
    elif args.mode == "lora":
        exp_name = f"lora_r{args.rank}_a{int(args.alpha)}_d{args.dropout}"
    else:
        exp_name = (f"partial_lora_r{args.rank}_a{int(args.alpha)}"
                    f"_frozen{args.freeze_blocks}")

    save_path = os.path.join(args.save_dir, f"{exp_name}_best.pth")

    # ------------------------------------------------------------------
    # WandB
    # ------------------------------------------------------------------
    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb.init(
            project = args.wandb_project,
            entity  = args.wandb_entity,
            name    = exp_name,
            config  = vars(args),
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir    = args.data_dir,
        batch_size  = args.batch,
        num_workers = args.workers,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if args.mode == "baseline":
        model = get_baseline_vit(num_classes=100)
    elif args.mode == "lora":
        model = get_lora_vit(
            rank      = args.rank,
            alpha     = args.alpha,
            dropout   = args.dropout,
            num_classes = 100,
        )
    else:  # partial_lora
        model = get_partial_lora_vit(
            rank                  = args.rank,
            alpha                 = args.alpha,
            dropout               = args.dropout,
            freeze_first_n_blocks = args.freeze_blocks,
            num_classes           = 100,
        )

    model = model.to(device)
    trainable_params = count_trainable_params(model)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    config = {
        "epochs":        args.epochs,
        "lr":            args.lr,
        "weight_decay":  1e-4,
        "warmup_epochs": 1,
    }

    log_grads = (args.mode != "baseline")   # log LoRA grad norms
    history, best_val_acc = run_training(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        config       = config,
        device       = device,
        wandb_run    = wandb_run,
        log_grads    = log_grads,
        save_path    = save_path,
    )

    # ------------------------------------------------------------------
    # Test evaluation with best weights
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load(save_path, map_location=device,
                                      weights_only=True), strict=False)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, per_class_acc = evaluate(
        model, test_loader, criterion, device,
        return_per_class=True, num_classes=100
    )
    print(f"\n{'='*60}")
    print(f"TEST RESULTS  [{exp_name}]")
    print(f"  Test Loss : {test_loss:.4f}")
    print(f"  Test Acc  : {test_acc:.2f}%")
    print(f"  Trainable : {trainable_params:,}")
    print(f"{'='*60}\n")

    if wandb_run:
        wandb_run.log({
            "test/loss":         test_loss,
            "test/acc":          test_acc,
            "trainable_params":  trainable_params,
        })

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_dir = os.path.join(args.save_dir, "plots")
    curves_path = plot_train_val_curves(history, exp_name, save_dir=plot_dir)
    hist_path   = plot_classwise_histogram(
        per_class_acc, CIFAR100_CLASSES, exp_name, save_dir=plot_dir
    )

    if wandb_run:
        wandb_run.log({
            "charts/train_val_curves": wandb.Image(curves_path),
            "charts/classwise_acc":    wandb.Image(hist_path),
        })

    # ------------------------------------------------------------------
    # Summary table row (printed for easy copy-paste into report)
    # ------------------------------------------------------------------
    lora = args.mode != "baseline"
    print("\n--- TABLE ROW ---")
    print(f"{'LoRA' if lora else 'No LoRA'} | "
          f"r={args.rank if lora else 'N/A'} | "
          f"α={int(args.alpha) if lora else 'N/A'} | "
          f"drop={args.dropout if lora else 'N/A'} | "
          f"Test Acc={test_acc:.2f}% | "
          f"Trainable={trainable_params:,}")
    print("-----------------\n")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
