# STL-10 Image Classification - Google Colab Version

🎯 **Complete STL-10 image classification using ResNet-18 on Google Colab with GPU acceleration**

## 🚀 Quick Start Guide

### 1. Upload to Google Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload the `STL10_Classification_Colab.ipynb` notebook
3. Ensure GPU runtime is enabled: Runtime → Change runtime type → Hardware accelerator: GPU

### 2. Run the Notebook
Simply run all cells in sequence! The notebook will automatically:
- ✅ Install all required dependencies
- ✅ Load STL-10 data from HuggingFace
- ✅ Train ResNet-18 model on GPU
- ✅ Log everything to Wandb
- ✅ Generate all required visualizations
- ✅ Provide exam answers automatically
- ✅ Prepare model for HuggingFace upload

## 📊 What You'll Get

### Automatic Results:
- **Test Accuracy** (Exam Question 10)
- **Class-wise accuracy** for all 10 classes (Exam Question 11) 
- **Confusion Matrix** visualization
- **Sample Predictions** (10 correct + 10 incorrect) with images
- **Training/Validation curves** on Wandb
- **Model checkpoint** ready for download
- **HuggingFace format** model for deployment

### Wandb Dashboard:
Complete experiment tracking with:
- Training and validation loss/accuracy curves
- Confusion matrix heatmap
- Class-wise accuracy bar chart
- Sample prediction tables with images
- Hyperparameter logging

## 📁 Files Included

- **STL10_Classification_Colab.ipynb**: Main notebook (run this!)
- **colab_config.py**: Configuration file (optional)
- **colab_requirements.txt**: Dependencies list (FYI only)
- **README_COLAB.md**: This guide

## 🎓 Exam Requirements ✅

All requirements automatically fulfilled:

1. **Data Loading** ✅
   - Loads STL-10 from `Chiranjeev007/STL-10_Subset`
   - Train, Validation, Test splits

2. **Custom Dataloaders** ✅
   - Data augmentation for training
   - Proper transformations

3. **ResNet-18 Model** ✅
   - Pretrained from torchvision
   - Modified for STL-10 (10 classes)

4. **Wandb Integration** ✅
   - Training/validation loss graphs
   - Training/validation accuracy graphs

5. **Model Management** ✅
   - Best model saved
   - HuggingFace format preparation

6. **Evaluation** ✅
   - Confusion matrix on W&B
   - Class-wise accuracy bar plot

7. **Predictions** ✅
   - 20 samples (10 correct, 10 incorrect)
   - Shows: True label, Predicted label, Image

8. **Exam Answers** ✅
   - **Question 10**: Test accuracy
   - **Question 11**: Class-wise accuracy for each class

## ⚡ GPU Optimization

The notebook is optimized for Google Colab:
- **Reduced epochs**: 15 (instead of 20) for faster training
- **Early stopping**: Patience of 3 epochs
- **Batch size**: 32 (optimized for Colab GPU memory)
- **Reduced workers**: 2 (safe for Colab)

## 🔧 Technical Details

- **Model**: ResNet-18 pretrained on ImageNet
- **Classes**: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck
- **Input Size**: 224x224 RGB images
- **Device**: Automatically detects GPU (T4/P100/V100)
- **Training**: Adam optimizer, ReduceLROnPlateau scheduler
- **Wandb**: Pre-configured with provided API key

## 📥 Download Results

After running, you can download:
1. **STL10_Classification_Results.zip**: Complete results package
2. **EXPERIMENT_SUMMARY.md**: Detailed summary with all answers
3. **Individual files**: Model checkpoint, plots, model card

## 📋 Expected Runtime

On Google Colab with GPU:
- **T4 GPU**: ~15-20 minutes total
- **P100 GPU**: ~10-15 minutes total  
- **V100 GPU**: ~8-12 minutes total

## 🎯 Success Metrics

Expected performance:
- **Test Accuracy**: ~70-85% (depends on data split)
- **Training**: Should converge within 10-15 epochs
- **GPU Utilization**: ~80-90% during training

## 🔗 Important Links

- **HuggingFace Dataset**: [Chiranjeev007/STL-10_Subset](https://huggingface.co/datasets/Chiranjeev007/STL-10_Subset)
- **GitHub Repo**: [MLOps-Shivam_Madhav_Kenche-M25CSA028](https://github.com/kingkenche/MLOps-Shivam_Madhav_Kenche-M25CSA028)
- **Wandb Project**: Will be created automatically

## 🚨 Troubleshooting

1. **GPU not detected**: Runtime → Change runtime type → GPU
2. **Memory issues**: Reduce batch_size in the config cell
3. **Wandb issues**: API key is pre-configured
4. **Package errors**: Restart runtime and run installation cell again

## 👨‍💻 Author

**Shivam Madhav Kenche (M25CSA028)**  
STL-10 Image Classification with MLOps Pipeline

---

🎉 **Ready to run! Just upload the notebook to Colab and execute all cells!**