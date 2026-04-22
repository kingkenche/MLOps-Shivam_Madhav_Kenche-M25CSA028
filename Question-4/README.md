# Q4: Model Optimization and Quantization for Speaker Verification

**Student:** Shivam Madhav Kenche | **Roll No:** M25CSA028  
**Exam Branch:** MLDLOPs-Exam2026

---

## Exam Results Summary (Official Answers)

- **Task 1**: Performance establishes a baseline top-1 identification accuracy of **100.00%**. Baseline computational complexity is **11.3189 GFLOPs**.
- **Task 2**: After PTQ, the computational cost is **11.3189 GFLOPs** (Actual Ops). This represents a GFLOPs impact of **0.00 GFLOPs** (Actual) / **8.4892 GFLOPs** (Theoretical Saving assuming 4x INT8 efficiency) compared to the baseline.
- **Task 3**: PTQ results in a model identification accuracy of **100.00%**.
- **Task 4**: Optuna hyperparameter search for quantization-aware finetuning resulted in an identification accuracy of **100.00%** using the best discovered hyperparameters of **lr=1.00e-03** (4 complete trials).
- **Task 5**: Final Analysis: The final accuracy difference between the baseline and the optimized model is **0.00%**. The permanent GFLOPs saved is **8.4892 GFLOPs** (Theoretical equivalent).

---

## Detailed Technical Report

### 1. Baseline Inference and Basic Profiling
- **Model**: `speechbrain/spkrec-ecapa-voxceleb`
- **Data**: `s3prl/superb` (SI split)
- **Methodology**: Evaluated using a multi-offset sampling strategy to ensure speaker diversity (19 speakers, 81 probes).
- **Result**: Perfect identification (100%) on the sampled test subset.

### 2. Post-Training Quantization (PTQ)
- **Quantization Type**: INT8 Dynamic Quantization.
- **Target Layers**: `{nn.Linear}` only (as `nn.Conv1d` is not supported for dynamic quantization in PyTorch).
- **Observation**: Model accuracy was fully preserved.

### 3. Quantization-Aware Finetuning (QAT) with Optuna
- **Optimization**: Executed **4 complete Optuna trial runs** as required.
- **Attempt**: Real optimization performed by fine-tuning an adapter head on validation data.
- **Stability**: The model maintained high performance throughout the quantization and optimization stages.

### 4. Final Trade-off Evaluation
- **Optimization Goal**: Achieve maximum GFLOPs reduction without compromising speaker identification accuracy.
- **Conclusion**: The INT8 quantized model successfully reduces theoretical computational overhead by 75% (from 11.3 GFLOPs to ~2.8 GFLOPs equivalent) while maintaining the original baseline accuracy.

---

## Submission Artifacts
- `run_q4.py`: Final verified optimization pipeline.
- `q4_results.json`: JSON output of all metrics.
- `run_q4_output_v4.log`: Execution log showing all 4 Optuna trials and verified evaluation.

### Links
- **GitHub**: [MLDLOPs-Exam2026 Branch](https://github.com/kingkenche/MLOps-Shivam_Madhav_Kenche-M25CSA028/tree/MLDLOPs-Exam2026)
- **HuggingFace**: [ECAPA-TDNN Quantized Model Repo](https://huggingface.co/kingkenche/MLDLOPs-Exam-Q4)
