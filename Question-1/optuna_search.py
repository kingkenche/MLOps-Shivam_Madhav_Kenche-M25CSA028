"""
optuna_search.py
================
Q1 Step 5 – Use Optuna to find the best LoRA hyperparameters on CIFAR-100.

Search space:
    rank   ∈ {2, 4, 8}
    alpha  ∈ {2, 4, 8}
    dropout = 0.1

Only LoRA hyperparameters are tuned, matching the assignment requirement.

Usage (inside Docker):
    python optuna_search.py --n_trials 20 --epochs 5 \
                            --data_dir ./data --save_dir ./optuna_out
"""

import argparse
import os
import torch
import torch.nn as nn
import optuna
from optuna.samplers import TPESampler
import wandb

from dataset import get_dataloaders
from model   import get_lora_vit
from utils   import run_training, evaluate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Optuna LoRA hyperparam search")
    p.add_argument("--n_trials",  type=int,   default=20)
    p.add_argument("--epochs",    type=int,   default=5,
                   help="Epochs per trial (use fewer for speed)")
    p.add_argument("--batch",     type=int,   default=64)
    p.add_argument("--workers",   type=int,   default=4)
    p.add_argument("--data_dir",  type=str,   default="./data")
    p.add_argument("--save_dir",  type=str,   default="./optuna_out")
    p.add_argument("--wandb_project", type=str, default="vit-cifar100-lora-optuna")
    p.add_argument("--wandb_entity",  type=str, default=None)
    p.add_argument("--no_wandb",  action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def make_objective(train_loader, val_loader, device, args):

    def objective(trial: optuna.Trial) -> float:
        # Tune LoRA hyperparameters ONLY
        rank   = trial.suggest_categorical("rank", [2, 4, 8])
        alpha  = trial.suggest_categorical("alpha", [2, 4, 8])
        dropout = 0.1
        lr     = 1e-3 # Fixed according to standard experiments

        exp_name = f"trial_{trial.number}_r{rank}_a{alpha}_d{dropout:.2f}"
        save_path = os.path.join(args.save_dir, f"{exp_name}.pth")

        wandb_run = None
        if not args.no_wandb:
            wandb_run = wandb.init(
                project  = args.wandb_project,
                entity   = args.wandb_entity,
                name     = exp_name,
                group    = "optuna_sweep",
                config   = {"rank": rank, "alpha": alpha, "dropout": dropout, "lr": lr,
                            "trial": trial.number},
                settings = wandb.Settings(reinit=True),
            )

        model = get_lora_vit(rank=rank, alpha=alpha, dropout=dropout)
        model = model.to(device)

        config = {
            "epochs":       args.epochs,
            "lr":           lr,
            "weight_decay": 1e-4,
            "warmup_epochs": 1,
        }

        _, best_val_acc = run_training(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            config       = config,
            device       = device,
            wandb_run    = wandb_run,
            log_grads    = True,
            save_path    = save_path,
        )

        if wandb_run:
            wandb_run.finish()

        return best_val_acc   # Optuna maximises this

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, _ = get_dataloaders(
        data_dir    = args.data_dir,
        batch_size  = args.batch,
        num_workers = args.workers,
    )

    study = optuna.create_study(
        direction = "maximize",
        sampler   = TPESampler(seed=42),
        study_name = "vit_lora_cifar100",
    )
    study.optimize(
        make_objective(train_loader, val_loader, device, args),
        n_trials = args.n_trials,
    )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("OPTUNA SEARCH COMPLETE")
    print(f"Best trial:  #{study.best_trial.number}")
    print(f"Best val acc: {study.best_value:.2f}%")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("="*60 + "\n")

    # Save best params
    import json
    result_path = os.path.join(args.save_dir, "best_params.json")
    with open(result_path, "w") as f:
        json.dump({
            "best_val_acc": study.best_value,
            "best_params":  study.best_params,
        }, f, indent=2)
    print(f"Saved best params → {result_path}")


if __name__ == "__main__":
    main()
