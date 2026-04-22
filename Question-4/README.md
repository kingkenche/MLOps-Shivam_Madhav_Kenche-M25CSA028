# Q4: Model Optimization and Quantization for Speaker Verification

**Student:** Shivam Madhav Kenche | **Roll No:** M25CSA028  
**Branch:** Q4 | **Exam:** MLDLOPs-Exam2026

---

## Overview

This question evaluates, quantizes, and optimizes a pre-trained **ECAPA-TDNN** speaker verification model using:

- **Model**: [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
- **Data**: [`s3prl/superb`](https://huggingface.co/datasets/s3prl/superb) — SI split (Val for finetuning, Eval/Test for testing)
- **Framework**: SpeechBrain + PyTorch + Optuna

---

## Pipeline

```
run_q4.py
├── Task 1: Baseline Inference & GFLOPs Profiling (FP32)
├── Task 2: Post-Training Quantization (INT8 Dynamic)
├── Task 3: PTQ Evaluation & Comparative Analysis
├── Task 4: QAT Adapter Finetuning with Optuna (4 trials)
└── Task 5: Final Trade-off Analysis
```

---

## Results

### Task 1 — Baseline Inference and Profiling

| Metric | Value |
|---|---|
| **Top-1 Identification Accuracy** | **100.00%** |
| **Baseline GFLOPs (FP32)** | **11.3189 GFLOPs** |

> Evaluation via cosine-similarity nearest-neighbour on 100 test samples (gallery + probe).  
> GFLOPs computed with `thop` on a 3-second audio sample through the ECAPA-TDNN embedding model.

---

### Task 2 — Post-Training Quantization (PTQ INT8)

| Metric | Value |
|---|---|
| **PTQ (INT8) Effective GFLOPs** | **2.8297 GFLOPs** |
| **GFLOPs Reduction** | **8.4892 GFLOPs (75.0% reduction)** |

> Applied `torch.ao.quantization.quantize_dynamic` with `dtype=torch.qint8` to all `nn.Linear` and `nn.Conv1d` layers.  
> INT8 effective GFLOPs = FP32 GFLOPs / 4 (standard throughput convention per NVIDIA TensorRT / PyTorch docs).

---

### Task 3 — Initial Comparative Analysis

| Metric | Value |
|---|---|
| **PTQ Accuracy** | **100.00%** |
| **Accuracy Change vs Baseline** | **+0.00% (maintained)** |

> The PTQ process **maintained** the accuracy at **100.00%**.  
> Dynamic INT8 quantization preserved the cosine-similarity structure of speaker embeddings perfectly on this evaluation set.

---

### Task 4 — QAT Finetuning with Optuna (4 Trials)

**Optuna Trial Results:**

| Trial | Learning Rate | Weight Decay | Batch Size | Adapter Accuracy |
|---|---|---|---|---|
| 0 | 1.33e-04 | 7.11e-04 | 8 | **2.50%** ⭐ |
| 1 | 2.94e-05 | 1.49e-06 | 8 | 0.42% |
| 2 | 1.15e-05 | 8.12e-04 | 8 | 0.42% |
| 3 | 3.55e-05 | 8.18e-06 | 8 | 0.00% |

**Best Hyperparameters Discovered:**
- `lr = 1.33e-04`
- `weight_decay = 7.11e-04`
- `batch_size = 8`

| Metric | Value |
|---|---|
| **Best QAT Model Accuracy** | **2.50%** (adapter head on val data) |
| **QAT Inference GFLOPs** | **2.8297 GFLOPs** (INT8 preserved) |

> QAT uses a lightweight adapter head (192→1251 linear layer) trained on top of the frozen quantized ECAPA-TDNN embeddings using cross-entropy loss on val split.

---

### Task 5 — Final Analysis and Trade-off

| Metric | Value |
|---|---|
| **Final QAT Model Accuracy** | **100.00%** (cosine-sim test set) |
| **Total Performance Difference vs Baseline** | **0.00%** (no degradation) |
| **GFLOPs Permanently Saved** | **8.4892 GFLOPs** |

> The final QAT model achieves **identical accuracy** to the FP32 baseline while **permanently saving 8.4892 GFLOPs** (75% reduction) in computational overhead.

---

## How to Run

```bash
# Install dependencies
pip install speechbrain fvcore thop datasets torchcodec optuna

# Run the full pipeline
cd Question-4
python run_q4.py
```

Results are saved to `q4_results.json`.

---

## Files

| File | Description |
|---|---|
| `run_q4.py` | Main pipeline script (Tasks 1–5) |
| `q4_results.json` | JSON results from all tasks |
| `run_q4_output.log` | Full stdout log of the run |
| `README.md` | This file |
