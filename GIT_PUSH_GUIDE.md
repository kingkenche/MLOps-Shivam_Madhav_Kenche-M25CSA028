# Quick Git Push Guide

## Push to GitHub Repository

```bash
# Navigate to submission folder
cd G:\Ass-1\GitHub_Submission\Assignment1

# Initialize git if not already done
git init

# Add remote if not already added
git remote add origin https://github.com/kingkenche/MLOps-Shivam_Madhav_Kenche-M25CSA028.git

# Checkout/create Assignment-1 branch
git checkout -b Assignment-1

# Add all files
git add .

# Commit with message
git commit -m "Add Assignment 1: Q1(a) ResNet and Q1(b) SVM Classification - Complete Submission"

# Push to GitHub
git push -u origin Assignment-1
```

## Verify Submission

1. Check GitHub repository: https://github.com/kingkenche/MLOps-Shivam_Madhav_Kenche-M25CSA028/tree/Assignment-1
2. Verify GitHub Pages: https://kingkenche.github.io/MLOps-Shivam_Madhav_Kenche-M25CSA028/
3. Test Colab link: https://colab.research.google.com/drive/1aEUrsukzyS6WkJQORQW1TazpADNcQgTv?usp=sharing

## Files to Push (17 files)

✅ Documentation (5):
- README.md
- M25CSA028_SHIVAM_MADHAV_KENCHE_Ass1_Report.md
- M25CSA028_SHIVAM_MADHAV_KENCHE_Ass1_Report.pdf
- index.html
- SUBMISSION_CHECKLIST.md

✅ Code (2):
- models/resnet.py
- models/svm_classifier.py

✅ Notebooks (2):
- notebooks/Q1a_Submission_Colab.ipynb
- notebooks/Assignment1_2.ipynb

✅ Results (8):
- results/mnist_q1a_results_final.csv
- results/fashion_q1a_results_final.csv
- results/q1b_svm_results.csv
- results/mnist_resnet18_bs16_SGD_lr0.001_best.pth
- results/fashion_resnet18_bs16_Adam_lr0.0001_best.pth
- results/mnist_training_curves.png
- results/fashion_training_curves.png
- results/combined_training_curves.png

## Important Notes

⚠️ Large files (.pth model files ~45MB each) - GitHub may require Git LFS
⚠️ If push fails due to file size, consider:
   1. Using Git LFS: `git lfs install` and `git lfs track "*.pth"`
   2. Or storing models elsewhere (Google Drive) and linking in README

## Alternative: Direct Upload to GitHub

If command line fails:
1. Go to https://github.com/kingkenche/MLOps-Shivam_Madhav_Kenche-M25CSA028
2. Switch to Assignment-1 branch (or create it)
3. Use "Add file" > "Upload files"
4. Drag and drop all folders/files
5. Commit directly through web interface
