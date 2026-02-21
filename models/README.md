---
license: mit
tags:
- image-classification
- stl10
- resnet18
- pytorch
datasets:
- Chiranjeev007/STL-10_Subset
metrics:
- accuracy: 0.8520
---

# STL-10 ResNet-18 Classification Model

This model is a fine-tuned ResNet-18 for STL-10 image classification trained on Google Colab.

## Model Details
- **Base Model**: ResNet-18 (pretrained on ImageNet)
- **Dataset**: STL-10 Subset from HuggingFace
- **Classes**: 10
- **Test Accuracy**: 0.8520
- **Validation Accuracy**: 0.8400
- **Training Platform**: Google Colab

## Class Names
airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck

## Class-wise Accuracy
- **airplane**: 0.9100
- **bird**: 0.8200
- **car**: 0.9300
- **cat**: 0.8400
- **deer**: 0.8600
- **dog**: 0.6000
- **horse**: 0.8200
- **monkey**: 0.8700
- **ship**: 0.8800
- **truck**: 0.9900

## Training Configuration
- Batch Size: 32
- Learning Rate: 0.001
- Epochs: 15
- Device: cuda

## Usage

```python
import torch
from torchvision import models, transforms
from PIL import Image

# Load model
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 10)

# Load weights (implement loading logic)
# checkpoint = torch.load('pytorch_model.bin')
# model.load_state_dict(checkpoint['model_state_dict'])

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

## Training Results
- Wandb Dashboard: https://wandb.ai/shivamkenche-indian-institute-of-technology-jodhpur/stl10-classification-colab/runs/1ohtxjh9

## Author
Shivam Madhav Kenche (M25CSA028)
