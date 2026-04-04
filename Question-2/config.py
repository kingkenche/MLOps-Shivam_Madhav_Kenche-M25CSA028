# config.py — shared hyperparameters and paths

import os

# ── Paths ──────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR    = "results"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)

# ── ResNet-18 Training (Q2-i) ──────────────────────────────────────────────
RESNET18_EPOCHS      = 100
RESNET18_BATCH_SIZE  = 128
RESNET18_LR          = 0.1
RESNET18_MOMENTUM    = 0.9
RESNET18_WEIGHT_DECAY= 5e-4
RESNET18_CKPT        = os.path.join(CHECKPOINT_DIR, "resnet18_cifar10_best.pth")

# ── FGSM ──────────────────────────────────────────────────────────────────
FGSM_EPSILONS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.1]   # sweep
FGSM_EPS      = 0.03                                    # default single run

# ── PGD (evaluation — used in test_all.py & fgsm_attack.py) ───────────────
PGD_EPS       = 0.03
PGD_EPS_STEP  = 0.007
PGD_MAX_ITER  = 40

# ── BIM (evaluation) ──────────────────────────────────────────────────────
BIM_EPS       = 0.03
BIM_EPS_STEP  = 0.007
BIM_MAX_ITER  = 40

# ── Detector Training Attack Params (matches evaluation) ─────
DET_PGD_EPS      = 0.03
DET_PGD_EPS_STEP = 0.007
DET_PGD_MAX_ITER = 40
DET_BIM_EPS      = 0.03
DET_BIM_EPS_STEP = 0.007
DET_BIM_MAX_ITER = 40

# ── Detector (ResNet-34, Q2-ii) ────────────────────────────────────────────
DET_EPOCHS       = 40
DET_BATCH_SIZE   = 128
DET_LR           = 2e-4
DET_WEIGHT_DECAY = 1e-4
DET_N_SAMPLES    = 20000          # 10000 clean + 10000 adv
DET_PGD_CKPT     = os.path.join(CHECKPOINT_DIR, "detector_pgd.pth")
DET_BIM_CKPT     = os.path.join(CHECKPOINT_DIR, "detector_bim.pth")

# ── WandB ─────────────────────────────────────────────────────────────────
WANDB_PROJECT  = "Assignment5-AdversarialART"

# ── CIFAR-10 ──────────────────────────────────────────────────────────────
CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# ── Device ────────────────────────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
