# Model Files - Large File Storage Instructions

Due to GitHub's 100MB file size limit, the trained model files are too large to be stored directly in the repository.

## 📦 Model Files Available Locally:

- **best_model.pth** (128.09 MB) - Complete PyTorch model checkpoint
- **pytorch_model.bin** (42.73 MB) - HuggingFace format model

## 📤 Upload Options:

### Option 1: Git LFS (Git Large File Storage)
```bash
# Install Git LFS if not already installed
git lfs install

# Track large files
git lfs track "models/*.pth"
git lfs track "models/*.bin"

# Add and commit
git add .gitattributes models/
git commit -m "Add large model files with Git LFS"
git push origin stl10-classification-results
```

### Option 2: External Storage Solutions

#### A. Google Drive / OneDrive
1. Upload model files to cloud storage
2. Share public links
3. Add links to README.md

#### B. HuggingFace Hub
1. Create model repository on [HuggingFace](https://huggingface.co/new)
2. Upload using the web interface or:
```bash
pip install huggingface_hub
python huggingface_upload.py
```

#### C. Repository Releases
1. Go to GitHub repository
2. Create a new release
3. Attach model files as assets

### Option 3: Model Recreation
The complete code is available in this repository. You can:
1. Run `python run_experiment.py` locally
2. Or use the Google Colab notebook: `STL10_Classification_Colab.ipynb`
3. Models will be regenerated with your training

## 📊 Model Performance

- **Test Accuracy**: Available after training completion
- **Validation Accuracy**: Saved in model checkpoint
- **Class-wise Accuracy**: For all 10 STL-10 classes
- **Confusion Matrix**: Generated automatically
- **Sample Predictions**: 20 examples with visualizations

## 🔗 Quick Links

- **Repository**: [MLOps-Shivam_Madhav_Kenche-M25CSA028](https://github.com/kingkenche/MLOps-Shivam_Madhav_Kenche-M25CSA028)  
- **Colab Notebook**: `STL10_Classification_Colab.ipynb`
- **Training Script**: `run_experiment.py`
- **HuggingFace Upload**: `huggingface_upload.py`

---
**Note**: All source code, documentation, and results (except large model files) are included in this repository.