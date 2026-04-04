"""
test_q1.py
==========
Load a saved checkpoint and evaluate on the CIFAR-100 test set.

Usage:
    # Baseline checkpoint
    python test_q1.py --mode baseline \
                      --ckpt checkpoints/baseline_no_lora_best.pth

    # LoRA checkpoint
    python test_q1.py --mode lora --rank 4 --alpha 4 --dropout 0.1 \
                      --ckpt checkpoints/lora_r4_a4_d0.1_best.pth
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import numpy as np
import wandb

from dataset import get_dataloaders, CIFAR100_CLASSES
from model   import get_baseline_vit, get_lora_vit, count_trainable_params
from utils   import evaluate, plot_classwise_histogram


def parse_args():
    p = argparse.ArgumentParser(description="Q1 Test evaluation")
    p.add_argument("--mode",    type=str, required=True,
                   choices=["baseline", "lora"])
    p.add_argument("--ckpt",    type=str, required=True,
                   help="Path to .pth checkpoint")
    p.add_argument("--rank",    type=int,   default=4)
    p.add_argument("--alpha",   type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch",   type=int,   default=64)
    p.add_argument("--workers", type=int,   default=4)
    p.add_argument("--data_dir",type=str,   default="./data")
    p.add_argument("--out_dir", type=str,   default="./test_results")
    p.add_argument("--wandb_project", type=str, default="vit-cifar100-lora")
    p.add_argument("--wandb_entity",  type=str, default=None)
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data (only test loader needed)
    _, _, test_loader = get_dataloaders(
        data_dir    = args.data_dir,
        batch_size  = args.batch,
        num_workers = args.workers,
    )

    # Model
    if args.mode == "baseline":
        model = get_baseline_vit(num_classes=100, pretrained=False)
        exp_name = "baseline_no_lora"
    else:
        model = get_lora_vit(rank=args.rank, alpha=args.alpha,
                             dropout=args.dropout, num_classes=100,
                             pretrained=False)
        exp_name = f"lora_r{args.rank}_a{int(args.alpha)}_d{args.dropout}"

    # For LoRA (PEFT) models, the checkpoint contains lora_ prefixed weights
    # alongside the base model weights. Use strict=False so missing base
    # weights (frozen) and extra lora_ keys don't cause a crash.
    state_dict = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    trainable = count_trainable_params(model)

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, per_class_acc = evaluate(
        model, test_loader, criterion, device,
        return_per_class=True, num_classes=100
    )

    # Report
    print("\n" + "="*60)
    print(f"TEST RESULTS  [{exp_name}]")
    print(f"  Checkpoint  : {args.ckpt}")
    print(f"  Test Loss   : {test_loss:.4f}")
    print(f"  Test Acc    : {test_acc:.2f}%")
    print(f"  Trainable   : {trainable:,}")
    print("  Per-class accuracy (top-5 / bottom-5):")
    idx_sorted = np.argsort(per_class_acc)
    for i in idx_sorted[-5:][::-1]:
        print(f"    {CIFAR100_CLASSES[i]:20s}: {per_class_acc[i]:.1f}%")
    print("  ...")
    for i in idx_sorted[:5]:
        print(f"    {CIFAR100_CLASSES[i]:20s}: {per_class_acc[i]:.1f}%")
    print("="*60 + "\n")

    # Save JSON results
    result = {
        "mode":           args.mode,
        "rank":           args.rank if args.mode == "lora" else None,
        "alpha":          args.alpha if args.mode == "lora" else None,
        "dropout":        args.dropout if args.mode == "lora" else None,
        "test_loss":      test_loss,
        "test_acc":       test_acc,
        "trainable_params": trainable,
        "per_class_acc":  per_class_acc.tolist(),
    }
    json_path = os.path.join(args.out_dir, f"{exp_name}_results.json")
    with open(json_path, "w") as f:
        import json as _json
        _json.dump(result, f, indent=2)
    print(f"Saved results → {json_path}")

    # Class-wise histogram
    hist_path = plot_classwise_histogram(
        per_class_acc, CIFAR100_CLASSES, exp_name, save_dir=args.out_dir
    )

    # WandB
    if not args.no_wandb:
        run = wandb.init(
            project = args.wandb_project,
            entity  = args.wandb_entity,
            name    = f"test_{exp_name}",
            config  = vars(args),
        )
        run.log({
            "test/loss":         test_loss,
            "test/acc":          test_acc,
            "trainable_params":  trainable,
            "charts/classwise":  wandb.Image(hist_path),
        })
        run.finish()


if __name__ == "__main__":
    main()
