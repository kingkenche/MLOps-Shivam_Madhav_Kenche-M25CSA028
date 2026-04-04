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
