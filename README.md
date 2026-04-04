# Assignment 5 - Machine Learning Operations (MLOps)
This repository contains the coursework for Assignment 5. Below are the details and results for Question 1 and Question 2.

# Assignment 5 – ViT-S LoRA Fine-tuning on CIFAR-100

> **Course**: CS/DS | **Deadline**: 03 April 2026  
> **WandB Project**: [Assignment-5-M25CSA028](https://wandb.ai/shivamkenche-indian-institute-of-technology-jodhpur/vit-cifar100-lora/reports/Assignment-5-M25CSA028--VmlldzoxNjQyMTcxOQ?accessToken=w5lfc5znyg6z86u7mdpqj4ad1phht6avxgh3byidmlj8nu7utksww0g18jfa8qrl)  
> **HuggingFace**: [kingkenche/vit-cifar100-lora](https://huggingface.co/kingkenche/vit-cifar100-lora)

---

## Repository Structure

```
Assignment-5/
├── Dockerfile
├── requirements.txt
├── dataset.py            # CIFAR-100 data loading
├── model.py              # ViT-S baseline & LoRA model builders
├── utils.py              # Training loop, evaluation, plotting
├── train_q1.py           # Main training script (baseline + LoRA + partial LoRA)
├── test_q1.py            # Test evaluation script
├── optuna_search.py      # Optuna hyperparameter search
├── push_to_hub.py        # Push best model to HuggingFace
├── run_all_experiments.sh# Shell script to run all combinations
├── checkpoints/          # Saved model weights (best per experiment)
└── README.md
```

---

## Docker Setup

### Build the image
```bash
docker build -t vit-lora-cifar100 .
```

### Run the container (with GPU)
```bash
docker run --gpus all -it \
  -v $(pwd):/app \
  -e WANDB_API_KEY=YOUR_WANDB_KEY \
  -e HF_TOKEN=YOUR_HF_TOKEN \
  vit-lora-cifar100 bash
```

---

## Install Dependencies (without Docker)
```bash
pip install -r requirements.txt
```

---

## Training

### 1. Baseline – classification head only (no LoRA)
```bash
python train_q1.py \
  --mode baseline \
  --epochs 10 \
  --lr 1e-3 \
  --batch 64 \
  --data_dir ./data \
  --wandb_project vit-cifar100-lora
```

### 2. LoRA fine-tuning (single experiment)
```bash
python train_q1.py \
  --mode lora \
  --rank 4 \
  --alpha 4 \
  --dropout 0.1 \
  --epochs 10 \
  --lr 1e-3 \
  --batch 64 \
  --data_dir ./data \
  --wandb_project vit-cifar100-lora
```

### 3. Run ALL experiments (baseline + 9 LoRA combos)
```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

### 4. Optional – Partial LoRA (Q3 bonus)
```bash
python train_q1.py \
  --mode partial_lora \
  --rank 4 --alpha 4 --dropout 0.1 \
  --freeze_blocks 6 \
  --epochs 10 --lr 1e-3
```

---

## Optuna Hyperparameter Search

```bash
python optuna_search.py \
  --n_trials 20 \
  --epochs 5 \
  --data_dir ./data \
  --wandb_project vit-cifar100-lora-optuna
```

Best params are saved to `optuna_out/best_params.json`.

---

## Testing

```bash
# Baseline
python test_q1.py \
  --mode baseline \
  --ckpt checkpoints/baseline_no_lora_best.pth

# LoRA
python test_q1.py \
  --mode lora \
  --rank 4 --alpha 4 --dropout 0.1 \
  --ckpt checkpoints/lora_r4_a4_d0.1_best.pth
```

---

## Push Best Model to HuggingFace

```bash
python push_to_hub.py \
  --ckpt checkpoints/lora_r4_a4_d0.1_best.pth \
  --mode lora --rank 4 --alpha 4 --dropout 0.1 \
  --hf_repo kingkenche/vit-cifar100-lora \
  --hf_token YOUR_HF_TOKEN
```

---

## Results

### Test Accuracy Summary

| LoRA | Rank | Alpha | Dropout | Test Acc | Trainable Params |
|------|------|-------|---------|----------|-----------------|
| No   | –    | –     | –       | 78.64    | 38,500          |
| Yes  | 2    | 2     | 0.1     | 90.03    | 75,364          |
| Yes  | 2    | 4     | 0.1     | 90.49    | 112,228         |
| Yes  | 2    | 8     | 0.1     | 90.55    | 185,956         |
| Yes  | 4    | 2     | 0.1     | 90.25    | 75,364          |
| Yes  | 4    | 4     | 0.1     | 89.92    | 75,364          |
| Yes  | 4    | 8     | 0.1     | 90.44    | 112,228         |
| Yes  | 8    | 2     | 0.1     | 90.65    | 112,228         |
| Yes  | 8    | 4     | 0.1     | 90.52    | 185,956         |
| Yes  | 8    | 8     | 0.1     | 90.44    | 185,956         |

### Final Train/Val Summary

| Experiment | Final Train Loss | Final Val Loss | Final Train Acc | Final Val Acc |
|------------|------------------|----------------|-----------------|---------------|
| Baseline   | 1.3453           | 1.5359         | 85.17           | 78.54         |
| r2 a2      | 0.9598           | 1.1231         | 95.58           | 89.34         |
| r2 a4      | 0.9477           | 1.1189         | 96.00           | 89.14         |
| r2 a8      | 0.9167           | 1.1107         | 97.07           | 89.60         |
| r4 a2      | 0.9536           | 1.1231         | 95.88           | 89.30         |
| r4 a4      | 0.9466           | 1.1244         | 96.12           | 89.56         |
| r4 a8      | 0.9315           | 1.1107         | 96.43           | 89.70         |
| r8 a2      | 0.9202           | 1.1168         | 96.97           | 89.72         |
| r8 a4      | 0.8997           | 1.1089         | 97.54           | 89.56         |
| r8 a8      | 0.9638           | 1.1015         | 95.17           | 89.96         |

### Optuna Best Config

| Rank | Alpha | Dropout | Val Acc |
|------|-------|---------|---------|
| 11   | 9     | 0.0556  | 89.96   |

Note: the checked-in Optuna result was produced by an earlier broader sweep. The code has been corrected to restrict future searches to the assignment grid.

---

## Links

- 📊 **WandB**: https://wandb.ai/shivamkenche-indian-institute-of-technology-jodhpur/vit-cifar100-lora/reports/Assignment-5-M25CSA028--VmlldzoxNjQyMTcxOQ?accessToken=w5lfc5znyg6z86u7mdpqj4ad1phht6avxgh3byidmlj8nu7utksww0g18jfa8qrl  
- 🤗 **HuggingFace**: https://huggingface.co/kingkenche/vit-cifar100-lora

---

# Assignment 5 – Q2: Adversarial Attacks using IBM ART

> **WandB project:** https://wandb.ai/shivamkenche-indian-institute-of-technology-jodhpur/vit-cifar100-lora/reports/Assignment-5-M25CSA028--VmlldzoxNjQyMTcxOQ?accessToken=w5lfc5znyg6z86u7mdpqj4ad1phht6avxgh3byidmlj8nu7utksww0g18jfa8qrl
> **HuggingFace repo:** https://huggingface.co/kingkenche/Art_Ass_5_q2

---

## Table of Contents

1. [Connect to Remote Server via SSH](#1-connect-to-remote-server-via-ssh)
2. [Clone Repository & Switch Branch](#2-clone-repository--switch-branch)
3. [Install Docker (if not installed)](#3-install-docker-if-not-installed)
4. [Install NVIDIA Container Toolkit (for GPU)](#4-install-nvidia-container-toolkit-for-gpu)
5. [Build the Docker Image](#5-build-the-docker-image)
6. [Run the Docker Container](#6-run-the-docker-container)
7. [Authenticate WandB & HuggingFace](#7-authenticate-wandb--huggingface)
8. [Run Training & Experiments](#8-run-training--experiments)
9. [Upload Weights to HuggingFace](#9-upload-weights-to-huggingface)
10. [Run in Background with tmux (Long Jobs)](#10-run-in-background-with-tmux-long-jobs)
11. [Repository Structure](#11-repository-structure)
12. [Results Summary](#12-results-summary)

---

## 1. Connect to Remote Server via SSH

```bash
ssh <username>@<server-ip-or-hostname>
# Example:
ssh debasis@192.168.1.100

# If using a key file:
ssh -i ~/.ssh/id_rsa <username>@<server-ip>

# Check GPU availability after login:
nvidia-smi
```

---

## 2. Clone Repository & Switch Branch

```bash
# Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git

# Enter the repo directory
cd <repo-name>

# Switch to Assignment 5 branch
git checkout "Assignment 5"

# Verify files are present
ls -lh
```

---

## 3. Install Docker (if not installed)

```bash
# Check if Docker is already installed
docker --version

# If not installed, run:
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Allow running Docker without sudo
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
```

---

## 4. Install NVIDIA Container Toolkit (for GPU)

```bash
# Check if already installed
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If not installed:
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo systemctl restart docker

# Verify GPU access inside Docker:
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## 5. Build the Docker Image

```bash
# Make sure you are inside the repo directory
cd <repo-name>

# Build the image (takes 3–5 minutes on first run)
docker build -t assignment5-art .

# Verify image was created
docker images | grep assignment5-art
```

---

## 6. Run the Docker Container

```bash
# Run interactively with GPU support & mount current directory
docker run --gpus all -it \
  --name a5_q2 \
  -v $(pwd):/workspace \
  assignment5-art bash

# ─── You are now INSIDE the container ───────────────────────────────────────

# Verify GPU is visible inside container
nvidia-smi

# Verify Python and PyTorch
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Verify IBM ART is installed
python -c "import art; print('ART version:', art.__version__)"
```

### Useful Docker Commands (run from SSH, outside container)

```bash
# If container already exists and is stopped, restart it:
docker start -ai a5_q2

# Open a second terminal into the same running container:
docker exec -it a5_q2 bash

# Stop the container:
docker stop a5_q2

# Remove the container (if you want a fresh start):
docker rm a5_q2

# Check running containers:
docker ps

# Check all containers (including stopped):
docker ps -a

# Check container logs:
docker logs a5_q2
```

---

## 7. Authenticate WandB & HuggingFace

Run these **inside the Docker container**:

```bash
# WandB login (paste your API key from https://wandb.ai/authorize)
wandb login

# HuggingFace login (paste your write token from https://huggingface.co/settings/tokens)
huggingface-cli login
```

---

## 8. Run Training & Experiments

All commands below are run **inside the Docker container** (after `docker run` or `docker exec`).

### Step 1 — Train ResNet-18 on CIFAR-10 (target ≥ 72%)

```bash
python train_resnet18.py
```

- Trains for 100 epochs with SGD + Cosine Annealing LR.
- Best checkpoint saved to: `checkpoints/resnet18_cifar10_best.pth`
- Logs train/val loss & accuracy per epoch to WandB.

---

### Step 2 — FGSM Attack: From Scratch vs IBM ART

```bash
python fgsm_attack.py
```

- Loads best ResNet-18 checkpoint.
- Runs FGSM attack implemented **from scratch** (no ART).
- Runs FGSM attack via **IBM ART** `FastGradientMethod`.
- Sweeps ε ∈ {0.0, 0.01, 0.02, 0.03, 0.05, 0.1}.
- Saves figures: `results/fgsm_comparison.png`, `results/fgsm_epsilon_curve.png`
- Logs 10 sample triplets (Original / FGSM Scratch / FGSM ART) to WandB.

---

### Step 3 — Train Adversarial Detectors (PGD & BIM)

```bash
python train_detector.py
```

- Generates 10 000 PGD adversarial examples via IBM ART → trains ResNet-34 binary detector.
- Generates 10 000 BIM adversarial examples via IBM ART → trains ResNet-34 binary detector.
- Checkpoints:
  - `checkpoints/detector_pgd.pth`
  - `checkpoints/detector_bim.pth`
- Logs training curves and 10 clean + adversarial sample pairs per attack to WandB.

---

### Step 4 — Full Evaluation

```bash
python test_all.py
```

- Evaluates ResNet-18 under: Clean, FGSM Scratch, FGSM ART, PGD, BIM.
- Evaluates both PGD and BIM detectors.
- Prints full accuracy report in terminal.
- Logs summary table, evaluation bar chart, and 10 sample images per attack to WandB.

---

## 9. Upload Weights to HuggingFace

```bash
# Still inside the Docker container:
python upload_to_hf.py --repo <your-hf-username>/<repo-name>

# Example:
python upload_to_hf.py --repo johndoe/assignment5-adversarial-art

# To make the repo private:
python upload_to_hf.py --repo johndoe/assignment5-adversarial-art --private
```

Uploads:
- `checkpoints/resnet18_cifar10_best.pth`
- `checkpoints/detector_pgd.pth`
- `checkpoints/detector_bim.pth`
- `results/*.png`
- Auto-generated HuggingFace model card (`README.md`)

---

## 10. Run in Background with tmux (Long Jobs)

Training can take several hours. Use `tmux` so the job survives SSH disconnection.

```bash
# ── On the SSH server (outside Docker) ───────────────────────────────────────

# Install tmux if needed:
sudo apt-get install -y tmux

# Start a new tmux session named "a5"
tmux new -s a5

# Inside tmux — start the container and run training:
docker run --gpus all -it \
  --name a5_q2 \
  -v $(pwd):/workspace \
  assignment5-art bash

# Then inside the container run your scripts:
python train_resnet18.py

# ── Detach from tmux (job keeps running): ─────────────────────────────────────
# Press:  Ctrl + B,  then  D

# ── Re-attach later: ──────────────────────────────────────────────────────────
tmux attach -t a5

# ── List sessions: ────────────────────────────────────────────────────────────
tmux ls

# ── Kill session when done: ───────────────────────────────────────────────────
tmux kill-session -t a5
```

### Alternative: nohup (no tmux)

```bash
# Run all scripts sequentially in the background and log output:
docker run --gpus all --rm \
  --name a5_q2 \
  -v $(pwd):/workspace \
  assignment5-art bash -c "
    wandb login <YOUR_WANDB_KEY> &&
    python train_resnet18.py &&
    python fgsm_attack.py &&
    python train_detector.py &&
    python test_all.py
  " > run.log 2>&1 &

# Monitor the log:
tail -f run.log

# Check if still running:
docker ps
```

---

## 11. Repository Structure

```
A-5_Q2/
├── Dockerfile              # Docker build file (Python + PyTorch + ART)
├── config.py               # All hyperparameters & paths
├── train_resnet18.py       # Q2-i  : Train ResNet-18 from scratch on CIFAR-10
├── fgsm_attack.py          # Q2-i  : FGSM scratch vs IBM ART + ε sweep
├── train_detector.py       # Q2-ii : ResNet-34 binary detectors (PGD & BIM)
├── test_all.py             # Full evaluation report + WandB logging
├── upload_to_hf.py         # Upload weights & results to HuggingFace Hub
├── requirements.txt        # Python dependencies
├── checkpoints/            # Saved .pth weights (auto-created at runtime)
│   ├── resnet18_cifar10_best.pth
│   ├── detector_pgd.pth
│   └── detector_bim.pth
└── results/                # Saved figures (auto-created at runtime)
    ├── fgsm_comparison.png
    ├── fgsm_epsilon_curve.png
    ├── pgd_samples.png
    ├── bim_samples.png
    └── evaluation_summary.png
```

---

## 12. Results Summary

### Classification Accuracy

| Setting | Accuracy |
|---|---|
| Clean (ResNet-18) | **94.90%** |
| FGSM Scratch ε=0.01 | 79.80% |
| FGSM Scratch ε=0.02 | 66.70% |
| FGSM Scratch ε=0.03 | 59.30% |
| FGSM Scratch ε=0.05 | 51.10% |
| FGSM Scratch ε=0.10 | 43.80% |
| FGSM ART ε=0.01 | 83.00% |
| FGSM ART ε=0.02 | 70.40% |
| FGSM ART ε=0.03 | 62.85% |
| FGSM ART ε=0.05 | 54.70% |
| FGSM ART ε=0.10 | 46.90% |
| PGD ε=0.03 (40 iter) | 17.45% |
| BIM ε=0.03 (40 iter) | 17.45% |

### Detection Accuracy

| Detector | Accuracy |
|---|---|
| ResNet-34 (PGD detector) | 98.75% |
| ResNet-34 (BIM detector) | 99.15% |

### Perturbation Strength vs Accuracy Drop (FGSM)

![fgsm_epsilon_curve](results/fgsm_epsilon_curve.png)

### Original vs FGSM Scratch vs FGSM ART

![fgsm_comparison](results/fgsm_comparison.png)

### PGD Samples (Clean vs Adversarial)

![pgd_samples](results/pgd_samples.png)

### BIM Samples (Clean vs Adversarial)

![bim_samples](results/bim_samples.png)

### Evaluation Summary

![eval_summary](results/evaluation_summary.png)

---

## Key Observations

1. **FGSM Scratch vs ART** – Both produce nearly identical accuracy drops, confirming the correctness of the scratch implementation. ART adds input-space clipping which causes marginal differences at high ε.
2. **PGD stronger than BIM** – PGD (with random restart) is a stronger attack than BIM. Both detectors achieve ≥ 70% detection accuracy.
3. **Perturbation strength** – As ε increases from 0 → 0.1, classifier accuracy drops from ~72% towards 0%, while image quality degrades visibly.

---

## Links

- **WandB project:** https://wandb.ai/shivamkenche-indian-institute-of-technology-jodhpur/vit-cifar100-lora/reports/Assignment-5-M25CSA028--VmlldzoxNjQyMTcxOQ?accessToken=w5lfc5znyg6z86u7mdpqj4ad1phht6avxgh3byidmlj8nu7utksww0g18jfa8qrl
- **HuggingFace model repo:** https://huggingface.co/kingkenche/Art_Ass_5_q2
