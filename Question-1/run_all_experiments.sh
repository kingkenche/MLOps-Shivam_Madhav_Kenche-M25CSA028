#!/usr/bin/env bash
# run_all_experiments.sh
# ======================
# Convenience script to launch every LoRA + baseline experiment for Q1.
# Run this INSIDE the Docker container.
#
# Usage:
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh

set -e

EPOCHS=10
BATCH=64
LR=1e-3
DATA_DIR=./data
SAVE_DIR=./checkpoints
WANDB_PROJECT=vit-cifar100-lora
# Docker containers default to 64 MB /dev/shm.
# Use WORKERS=0 to avoid "Bus error / out of shared memory".
# If you started the container with --shm-size=8g, set WORKERS=4.
WORKERS=0

echo "=========================================="
echo "  Q1 – ViT-S CIFAR-100 Experiments"
echo "=========================================="

# ---- Baseline (no LoRA) ----
echo ""
echo "[1/10] BASELINE – classification head only"
python train_q1.py \
    --mode baseline \
    --epochs $EPOCHS \
    --lr $LR \
    --batch $BATCH \
    --workers $WORKERS \
    --data_dir $DATA_DIR \
    --save_dir $SAVE_DIR \
    --wandb_project $WANDB_PROJECT

# ---- LoRA experiments (rank x alpha combinations) ----
exp=2
for RANK in 2 4 8; do
  for ALPHA in 2 4 8; do
    echo ""
    echo "[$exp/10] LoRA  rank=$RANK  alpha=$ALPHA  dropout=0.1"
    python train_q1.py \
        --mode lora \
        --rank $RANK \
        --alpha $ALPHA \
        --dropout 0.1 \
        --epochs $EPOCHS \
        --lr $LR \
        --batch $BATCH \
        --workers $WORKERS \
        --data_dir $DATA_DIR \
        --save_dir $SAVE_DIR \
        --wandb_project $WANDB_PROJECT
    exp=$((exp+1))
  done
done

echo ""
echo "=========================================="
echo "  All experiments complete!"
echo "=========================================="

# ---- Optuna search ----
echo ""
echo "Starting Optuna hyperparameter search (20 trials, 5 epochs each)..."
python optuna_search.py \
    --n_trials 20 \
    --epochs 5 \
    --batch $BATCH \
    --workers $WORKERS \
    --data_dir $DATA_DIR \
    --save_dir ./optuna_out \
    --wandb_project ${WANDB_PROJECT}-optuna

echo ""
echo "All done! Check ./checkpoints and ./optuna_out"
