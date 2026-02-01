# 🚀 Quick Start Guide - CIFAR-10 CNN Training

## Prerequisites Check

Before starting, ensure you have:
- ✅ Python 3.8+ installed
- ✅ CUDA-capable GPU (recommended, but CPU works too)
- ✅ WandB account ([create one here](https://wandb.ai/site/signup))

## Step-by-Step Training Instructions

### 1. Install Dependencies (if not already done)

```bash
pip install -r requirements.txt
```

### 2. Login to WandB

First time only - authenticate with WandB:

```bash
wandb login
```

You'll be prompted to enter your API key from https://wandb.ai/authorize

### 3. Start Training

**Option A: Default Settings (30 epochs)**
```bash
python train_model.py
```

**Option B: Custom Settings**
```bash
python train_model.py --epochs 25 --batch-size 128 --lr 0.1 --project-name "my-cifar10" --run-name "experiment-1"
```

### 4. Monitor Training

During training, you'll see:
- Real-time progress bars for each epoch
- Training/validation metrics
- WandB dashboard link (click to view visualizations)

Example output:
```
============================================================
Epoch 1/30
============================================================
Epoch 1 [Train]: 100%|████████| 390/390 [02:15<00:00]
Epoch 1 [Val]:   100%|████████| 78/78 [00:15<00:00]

Epoch 1 Summary:
  Train Loss: 1.8234 | Train Acc: 32.45%
  Val Loss: 1.6543 | Val Acc: 38.21%
  Learning Rate: 0.100000
```

### 5. View Results in WandB

The training script will output a WandB dashboard URL like:
```
✓ Check your results at: https://wandb.ai/<username>/<project>/<run-id>
```

Click this link to see:
- 📊 Training/validation curves
- 📈 Gradient flow visualizations (updated every 100 steps)
- 📉 Weight update visualizations (updated every 100 steps)
- 🔍 Model complexity metrics (FLOPs, parameters)

### 6. After Training Completes

Your trained model will be saved in:
- `models/best_model.pth` - Best performing model
- `checkpoints/checkpoint_epoch_X.pth` - Checkpoints every 5 epochs

**IMPORTANT**: These files are gitignored and won't be pushed to GitHub.

## Expected Training Time

- **With GPU (CUDA)**: ~2-3 minutes per epoch → ~60-90 minutes for 30 epochs
- **With CPU**: ~15-20 minutes per epoch → ~7-10 hours for 30 epochs

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch size
```bash
python train_model.py --batch-size 64
```

### Issue: WandB login fails
**Solution**: 
1. Get your API key from https://wandb.ai/authorize
2. Run `wandb login` and paste the key

### Issue: Slow data loading
**Solution**: The first run downloads CIFAR-10 dataset (~170MB). Subsequent runs will be faster.

## Next Steps After Training

1. **Review WandB Dashboard**: Analyze all visualizations
2. **Document Findings**: Update README.md with your observations
3. **Prepare Report**: Include key metrics and insights
4. **GitHub Submission**: 
   ```bash
   # Create a branch with your name and roll number
   git checkout -b name_rollnumber_lab2_worksheet
   
   # Push to GitHub (models are auto-excluded)
   git remote add origin <your-github-repo-url>
   git push -u origin name_rollnumber_lab2_worksheet
   ```

## What Gets Tracked in WandB?

### Metrics (Every Step)
- Training loss
- Training accuracy
- Learning rate

### Metrics (Every Epoch)
- Validation loss
- Validation accuracy

### Visualizations (Every 100 Steps)
- **Gradient Flow**: Mean, std, and norm of gradients per layer
- **Weight Updates**: Mean change, norm change, and relative change per layer

### Model Info (Once)
- Total FLOPs
- Total parameters
- Trainable parameters
- Model architecture

## Tips for Better Results

1. **Monitor gradient flow**: Watch for vanishing/exploding gradients
2. **Check weight updates**: Ensure all layers are learning
3. **Validation accuracy**: Should improve over epochs
4. **Learning rate**: Cosine annealing automatically adjusts

## Questions?

Check the main README.md for detailed documentation or review the code comments in `src/` directory.

---

**Ready to train?** Run: `python train_model.py`
