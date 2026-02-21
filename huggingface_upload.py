import torch
import os
from transformers import AutoImageProcessor, AutoModelForImageClassification
from huggingface_hub import HfApi, Repository
from config import Config
from model import create_model

def upload_model_to_huggingface(config):
    """Upload trained model to HuggingFace Hub"""
    
    # Load the best model
    model_path = os.path.join(config.MODEL_SAVE_PATH, "best_model.pth")
    
    # Create model
    model = create_model(config)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Convert to HuggingFace format
    # Note: This is a simplified approach. For a complete implementation,
    # you would need to create a proper HuggingFace model configuration
    
    print(f"Model loaded from {model_path}")
    print(f"Best validation accuracy: {checkpoint['accuracy']:.4f}")
    
    # Save model in a format that can be uploaded
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'num_classes': config.NUM_CLASSES,
            'class_names': config.CLASS_NAMES,
            'accuracy': checkpoint['accuracy']
        }
    }, os.path.join(config.MODEL_SAVE_PATH, "pytorch_model.bin"))
    
    # Create model card content
    model_card_content = f"""
---
license: mit
tags:
- image-classification
- stl10
- resnet18
- pytorch
datasets:
- {config.DATASET_NAME}
metrics:
- accuracy
---

# STL-10 ResNet-18 Classification Model

This model is a fine-tuned ResNet-18 for STL-10 image classification.

## Model Details
- Base Model: ResNet-18 (pretrained on ImageNet)
- Dataset: STL-10 Subset
- Classes: {config.NUM_CLASSES}
- Accuracy: {checkpoint['accuracy']:.4f}

## Class Names
{', '.join(config.CLASS_NAMES)}

## Usage

```python
import torch
from torchvision import transforms
from PIL import Image

# Load model (implement loading logic)
# model = load_model()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Inference
image = Image.open("path/to/image.jpg")
input_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    outputs = model(input_tensor)
    predicted_class = torch.argmax(outputs, dim=1)
```
"""
    
    # Save model card
    with open(os.path.join(config.MODEL_SAVE_PATH, "README.md"), "w") as f:
        f.write(model_card_content)
    
    print(f"Model prepared for upload to {config.HF_MODEL_NAME}")
    print("Model card created in models/README.md")
    print("To upload to HuggingFace Hub:")
    print(f"1. Create repository: {config.HF_MODEL_NAME}")
    print("2. Upload files from models/ directory")
    print("3. Run: huggingface-cli upload {config.HF_MODEL_NAME} models/")

def load_model_from_huggingface(config):
    """Load model from HuggingFace Hub"""
    try:
        # This is a placeholder - adjust based on your actual HF model structure
        model_path = os.path.join(config.MODEL_SAVE_PATH, "pytorch_model.bin")
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            model = create_model(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model loaded successfully from local HuggingFace format")
            return model
        else:
            print("Model not found. Please ensure the model is uploaded to HuggingFace first.")
            return None
            
    except Exception as e:
        print(f"Error loading model from HuggingFace: {e}")
        return None

if __name__ == "__main__":
    config = Config()
    upload_model_to_huggingface(config)