# Assignment 4: Optimizing Transformer Translation with Ray Tune & Optuna

## 📋 Overview

This assignment implements hyperparameter optimization for an English-to-Hindi Transformer translation model using **Ray Tune** and **Optuna**. The goal is to match or exceed baseline performance (BLEU score of 0.50) in significantly fewer epochs using intelligent hyperparameter search.

---

## 📚 Repository Links

### GitHub Repository
- **Main Repository**: [MLOps-Shivam_Madhav_Kenche-M25CSA028](https://github.com/kingkenche/MLOps-Shivam_Madhav_Kenche-M25CSA028)
- **Branches**:
  - `main`: Code, notebooks, report, and visualizations
  - `assignment-4`: All files including model checkpoints

### Hugging Face Model Hub
- **Repository**: [kingkenche/transformer-en-to-hi-assignment-4](https://huggingface.co/kingkenche/transformer-en-to-hi-assignment-4)
- **Access**: Complete assignment files including:
  - Notebooks & code
  - Report & visualizations
  - Model checkpoints (136 MB + 192 MB + 577 MB)
  - Training data

---

## 📦 Contents

### Code & Notebooks
- `M25CSA028-ass-4-tuned-en-to-hi.ipynb` - Main tuned notebook with Ray Tune + Optuna
- `en_to_hi.ipynb` - Baseline notebook (100 epochs, no tuning)
- `rollno_ass_4_tuned_en_to_hi_after_retraining.py` - Retraining script

### Model Checkpoints
- `M25CSA028_ass_4_best_model.pth` (136 MB) - Best tuned model weights
- `transformer_translation_final.pth` (192 MB) - Baseline model weights
- `checkpoint.pt` (577 MB) - Training checkpoint

### Report & Data
- `M25CSA028_ass_4_report_pdf.pdf` - Comprehensive report with metrics & findings
- `English-Hindi.tsv` - Dataset (26.15 MB)

### Visualizations
- `plot_baseline_loss.png` - Baseline training loss curve
- `plot_dataset_eda.png` - Dataset exploratory analysis
- `plot_hyperparameter_impact.png` - Hyperparameter impact analysis
- `plot_loss_comparison.png` - Baseline vs Best comparison
- `plot_metrics_comparison.png` - Key metrics comparison
- `plot_per_sentence_bleu.png` - Per-sentence BLEU scores
- `plot_radar_comparison.png` - Performance radar chart
- `plot_sweep_analysis.png` - Sweep analysis dashboard

---

## 🎯 Key Results

| Metric | Baseline (100 epochs) | Best Model (≤50 epochs) |
|--------|----------------------|------------------------|
| Training Time | ~3200s | ~1100s |
| Final Loss | ~1.42 | ~1.28 |
| BLEU Score | 50.00% | 52.00%+ |
| Efficiency | Baseline | **65% faster** |

---

## 🔍 Hyperparameters Tuned (8 total)

1. **Learning Rate** (`lr`) - 1e-5 → 1e-3 (loguniform)
2. **Batch Size** (`batch_size`) - 32, 64, 128 (choice)
3. **Number of Attention Heads** (`num_heads`) - 4, 8 (choice)
4. **FeedForward Dimension** (`d_ff`) - 1024, 2048, 4096 (choice)
5. **Dropout Rate** (`dropout`) - 0.05 → 0.4 (uniform)
6. **Model Dimension** (`d_model`) - 256, 512 (choice)
7. **Number of Layers** (`num_layers`) - 3, 4, 6 (choice)
8. **Weight Decay** (`weight_decay`) - 1e-6 → 1e-3 (loguniform)

---

## 🚀 Quick Start

### Clone Repository
```bash
git clone https://github.com/kingkenche/MLOps-Shivam_Madhav_Kenche-M25CSA028.git
cd MLOps-Shivam_Madhav_Kenche-M25CSA028
git checkout assignment-4
```

### Download from Hugging Face
```bash
pip install huggingface-hub
huggingface-cli download kingkenche/transformer-en-to-hi-assignment-4
```

### Run Notebook
```bash
jupyter notebook M25CSA028-ass-4-tuned-en-to-hi.ipynb
```

---

## 📊 Technologies Used

- **PyTorch** - Deep learning framework
- **Ray Tune** - Hyperparameter optimization framework
- **Optuna** - Bayesian optimization with TPE sampler
- **ASHA Scheduler** - Early stopping & trial pruning
- **NLTK** - BLEU score evaluation
- **Hugging Face** - Model hosting & versioning

---

## ✅ Submission Checklist

- [x] Baseline metrics documented (100 epochs)
- [x] 8+ hyperparameters configured
- [x] OptunaSearch + ASHA scheduler implemented
- [x] Best model weights saved
- [x] Final metrics achieved (≤50 epochs, matched/exceeded BLEU)
- [x] Comprehensive report generated
- [x] Code refactored for Ray Tune
- [x] Files pushed to GitHub
- [x] Complete repository on Hugging Face

---

## 📖 Notes

- All model files (906 MB total) are hosted on Hugging Face for easy access
- The assignment-4 branch contains all checkpoints and model files
- The main branch contains clean code and documentation
- Recommended to download from Hugging Face for full model weights

---

## 📞 Contact

- **GitHub**: [kingkenche](https://github.com/kingkenche)
- **Hugging Face**: [kingkenche](https://huggingface.co/kingkenche)

---

**Status**: ✅ Complete - Assignment 4 submitted with full hyperparameter tuning and optimization

