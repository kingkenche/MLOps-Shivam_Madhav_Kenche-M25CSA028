# Code Citations

## License: unknown
https://github.com/MariusLotz/Explainable_AI/blob/b57d457b7ed44d231589aba0b7fb92bf6b6bf9c1/Data/make_confusion_matrix.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt
```


## License: unknown
https://github.com/MariusLotz/Explainable_AI/blob/b57d457b7ed44d231589aba0b7fb92bf6b6bf9c1/Data/make_confusion_matrix.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt
```


## License: unknown
https://github.com/MariusLotz/Explainable_AI/blob/b57d457b7ed44d231589aba0b7fb92bf6b6bf9c1/Data/make_confusion_matrix.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt
```


## License: unknown
https://github.com/MariusLotz/Explainable_AI/blob/b57d457b7ed44d231589aba0b7fb92bf6b6bf9c1/Data/make_confusion_matrix.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt
```


## License: unknown
https://github.com/MariusLotz/Explainable_AI/blob/b57d457b7ed44d231589aba0b7fb92bf6b6bf9c1/Data/make_confusion_matrix.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt
```


## License: unknown
https://github.com/MariusLotz/Explainable_AI/blob/b57d457b7ed44d231589aba0b7fb92bf6b6bf9c1/Data/make_confusion_matrix.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt
```


## License: unknown
https://github.com/MariusLotz/Explainable_AI/blob/b57d457b7ed44d231589aba0b7fb92bf6b6bf9c1/Data/make_confusion_matrix.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt
```


## License: unknown
https://github.com/MariusLotz/Explainable_AI/blob/b57d457b7ed44d231589aba0b7fb92bf6b6bf9c1/Data/make_confusion_matrix.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt
```


## License: unknown
https://github.com/MariusLotz/Explainable_AI/blob/b57d457b7ed44d231589aba0b7fb92bf6b6bf9c1/Data/make_confusion_matrix.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt
```


## License: unknown
https://github.com/Liadrid/AI-Computing-Systems/blob/2fd9ebf187d62e2b9d4c6e225485fa0436c651a0/mnist/LeNet.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/AIS-22/UNI-AIS-BiometricSystems/blob/7e01e3686c24b73b7f56ed08e49ff0103bf0f17b/src/classifier/impl/ResnetClassifier.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/napronald/Synthetic-Data-Generation-for-Deep-Learning-Model-Enhancement/blob/831c739df9dc34147f9899253a8877938a49b1ef/classifier.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/Liadrid/AI-Computing-Systems/blob/2fd9ebf187d62e2b9d4c6e225485fa0436c651a0/mnist/LeNet.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/AIS-22/UNI-AIS-BiometricSystems/blob/7e01e3686c24b73b7f56ed08e49ff0103bf0f17b/src/classifier/impl/ResnetClassifier.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/napronald/Synthetic-Data-Generation-for-Deep-Learning-Model-Enhancement/blob/831c739df9dc34147f9899253a8877938a49b1ef/classifier.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/Liadrid/AI-Computing-Systems/blob/2fd9ebf187d62e2b9d4c6e225485fa0436c651a0/mnist/LeNet.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/AIS-22/UNI-AIS-BiometricSystems/blob/7e01e3686c24b73b7f56ed08e49ff0103bf0f17b/src/classifier/impl/ResnetClassifier.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/napronald/Synthetic-Data-Generation-for-Deep-Learning-Model-Enhancement/blob/831c739df9dc34147f9899253a8877938a49b1ef/classifier.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/Liadrid/AI-Computing-Systems/blob/2fd9ebf187d62e2b9d4c6e225485fa0436c651a0/mnist/LeNet.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/AIS-22/UNI-AIS-BiometricSystems/blob/7e01e3686c24b73b7f56ed08e49ff0103bf0f17b/src/classifier/impl/ResnetClassifier.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/napronald/Synthetic-Data-Generation-for-Deep-Learning-Model-Enhancement/blob/831c739df9dc34147f9899253a8877938a49b1ef/classifier.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/Liadrid/AI-Computing-Systems/blob/2fd9ebf187d62e2b9d4c6e225485fa0436c651a0/mnist/LeNet.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/AIS-22/UNI-AIS-BiometricSystems/blob/7e01e3686c24b73b7f56ed08e49ff0103bf0f17b/src/classifier/impl/ResnetClassifier.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/napronald/Synthetic-Data-Generation-for-Deep-Learning-Model-Enhancement/blob/831c739df9dc34147f9899253a8877938a49b1ef/classifier.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/Liadrid/AI-Computing-Systems/blob/2fd9ebf187d62e2b9d4c6e225485fa0436c651a0/mnist/LeNet.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/AIS-22/UNI-AIS-BiometricSystems/blob/7e01e3686c24b73b7f56ed08e49ff0103bf0f17b/src/classifier/impl/ResnetClassifier.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/napronald/Synthetic-Data-Generation-for-Deep-Learning-Model-Enhancement/blob/831c739df9dc34147f9899253a8877938a49b1ef/classifier.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/Liadrid/AI-Computing-Systems/blob/2fd9ebf187d62e2b9d4c6e225485fa0436c651a0/mnist/LeNet.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/AIS-22/UNI-AIS-BiometricSystems/blob/7e01e3686c24b73b7f56ed08e49ff0103bf0f17b/src/classifier/impl/ResnetClassifier.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/napronald/Synthetic-Data-Generation-for-Deep-Learning-Model-Enhancement/blob/831c739df9dc34147f9899253a8877938a49b1ef/classifier.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/Liadrid/AI-Computing-Systems/blob/2fd9ebf187d62e2b9d4c6e225485fa0436c651a0/mnist/LeNet.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/AIS-22/UNI-AIS-BiometricSystems/blob/7e01e3686c24b73b7f56ed08e49ff0103bf0f17b/src/classifier/impl/ResnetClassifier.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/napronald/Synthetic-Data-Generation-for-Deep-Learning-Model-Enhancement/blob/831c739df9dc34147f9899253a8877938a49b1ef/classifier.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/Liadrid/AI-Computing-Systems/blob/2fd9ebf187d62e2b9d4c6e225485fa0436c651a0/mnist/LeNet.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/AIS-22/UNI-AIS-BiometricSystems/blob/7e01e3686c24b73b7f56ed08e49ff0103bf0f17b/src/classifier/impl/ResnetClassifier.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/napronald/Synthetic-Data-Generation-for-Deep-Learning-Model-Enhancement/blob/831c739df9dc34147f9899253a8877938a49b1ef/classifier.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(
```


## License: unknown
https://github.com/Liadrid/AI-Computing-Systems/blob/2fd9ebf187d62e2b9d4c6e225485fa0436c651a0/mnist/LeNet.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
```


## License: unknown
https://github.com/AIS-22/UNI-AIS-BiometricSystems/blob/7e01e3686c24b73b7f56ed08e49ff0103bf0f17b/src/classifier/impl/ResnetClassifier.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
```


## License: unknown
https://github.com/napronald/Synthetic-Data-Generation-for-Deep-Learning-Model-Enhancement/blob/831c739df9dc34147f9899253a8877938a49b1ef/classifier.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
```


## License: unknown
https://github.com/Liadrid/AI-Computing-Systems/blob/2fd9ebf187d62e2b9d4c6e225485fa0436c651a0/mnist/LeNet.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size
```


## License: unknown
https://github.com/AIS-22/UNI-AIS-BiometricSystems/blob/7e01e3686c24b73b7f56ed08e49ff0103bf0f17b/src/classifier/impl/ResnetClassifier.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size
```


## License: unknown
https://github.com/napronald/Synthetic-Data-Generation-for-Deep-Learning-Model-Enhancement/blob/831c739df9dc34147f9899253a8877938a49b1ef/classifier.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size
```


## License: unknown
https://github.com/randr97/squeeze-res/blob/525db8a757585c7aacdd268b8cd59e6a4e4993d8/train.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': correct_predictions / total_samples
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct
```


## License: unknown
https://github.com/randr97/squeeze-res/blob/525db8a757585c7aacdd268b8cd59e6a4e4993d8/train.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': correct_predictions / total_samples
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct
```


## License: unknown
https://github.com/randr97/squeeze-res/blob/525db8a757585c7aacdd268b8cd59e6a4e4993d8/train.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': correct_predictions / total_samples
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct
```


## License: unknown
https://github.com/randr97/squeeze-res/blob/525db8a757585c7aacdd268b8cd59e6a4e4993d8/train.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': correct_predictions / total_samples
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct
```


## License: unknown
https://github.com/randr97/squeeze-res/blob/525db8a757585c7aacdd268b8cd59e6a4e4993d8/train.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': correct_predictions / total_samples
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct
```


## License: unknown
https://github.com/randr97/squeeze-res/blob/525db8a757585c7aacdd268b8cd59e6a4e4993d8/train.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': correct_predictions / total_samples
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct
```


## License: unknown
https://github.com/randr97/squeeze-res/blob/525db8a757585c7aacdd268b8cd59e6a4e4993d8/train.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': correct_predictions / total_samples
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct
```


## License: unknown
https://github.com/randr97/squeeze-res/blob/525db8a757585c7aacdd268b8cd59e6a4e4993d8/train.py

```
Let me create a comprehensive solution for your STL-10 image classification project. I'll start by setting up the project structure and implementing all the required components.

```python
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
wandb>=0.15.0
huggingface_hub>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
numpy>=1.24.0
pillow>=9.5.0
tqdm>=4.65.0
```

```python
# config.py
import os

class Config:
    # Dataset configuration
    DATASET_NAME = "Chiranjeev007/STL-10_Subset"
    NUM_CLASSES = 10
    CLASS_NAMES = [
        "airplane", "bird", "car", "cat", "deer", 
        "dog", "horse", "monkey", "ship", "truck"
    ]
    
    # Model configuration
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    
    # Training configuration
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    PATIENCE = 5
    
    # Data augmentation
    IMG_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Wandb configuration
    WANDB_PROJECT = "stl10-classification"
    WANDB_ENTITY = None  # Set your wandb username if needed
    WANDB_API_KEY = "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"
    
    # HuggingFace configuration
    HF_MODEL_NAME = "kingkenche/stl10-resnet18"
    
    # Paths
    MODEL_SAVE_PATH = "models"
    RESULTS_PATH = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

```python
# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import wandb
from PIL import Image

def create_confusion_matrix(y_true, y_pred, class_names):
    """Create and plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt

def calculate_class_wise_accuracy(y_true, y_pred, class_names):
    """Calculate accuracy for each class"""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    return dict(zip(class_names, class_accuracies))

def plot_class_wise_accuracy(class_accuracies):
    """Plot class-wise accuracy bar chart"""
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color='skyblue', alpha=0.8)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def log_sample_predictions(model, dataloader, class_names, device, num_correct=10, num_incorrect=10):
    """Log sample predictions to wandb"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
                    break
                    
                img = images[i].cpu()
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                
                # Denormalize image for display
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = torch.clamp(img, 0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                sample_data = {
                    "image": wandb.Image(img, caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}"),
                    "true_label": class_names[true_label],
                    "predicted_label": class_names[pred_label],
                    "correct": true_label == pred_label
                }
                
                if true_label == pred_label and len(correct_samples) < num_correct:
                    correct_samples.append(sample_data)
                elif true_label != pred_label and len(incorrect_samples) < num_incorrect:
                    incorrect_samples.append(sample_data)
    
    # Log correct predictions
    correct_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in correct_samples:
        correct_table.add_data(sample["image"], sample["true_label"], 
                             sample["predicted_label"], sample["correct"])
    
    # Log incorrect predictions
    incorrect_table = wandb.Table(columns=["image", "true_label", "predicted_label", "correct"])
    for sample in incorrect_samples:
        incorrect_table.add_data(sample["image"], sample["true_label"], 
                               sample["predicted_label"], sample["correct"])
    
    wandb.log({
        "correct_predictions": correct_table,
        "incorrect_predictions": incorrect_table
    })
```

```python
# data_loader.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class STL10Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(config):
    """Define data transformations"""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])
    
    return train_transform, val_test_transform

def create_dataloaders(config):
    """Create train, validation, and test dataloaders"""
    # Load dataset from HuggingFace
    print(f"Loading dataset: {config.DATASET_NAME}")
    dataset = load_dataset(config.DATASET_NAME)
    
    # Get transforms
    train_transform, val_test_transform = get_transforms(config)
    
    # Create datasets
    train_dataset = STL10Dataset(dataset['train'], transform=train_transform)
    val_dataset = STL10Dataset(dataset['validation'], transform=val_test_transform)
    test_dataset = STL10Dataset(dataset['test'], transform=val_test_transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader
```

```python
# model.py
import torch
import torch.nn as nn
from torchvision import models

def create_model(config):
    """Create ResNet-18 model for STL-10 classification"""
    # Load pretrained ResNet-18
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify the final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

def save_model_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_model_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']
```

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from tqdm import tqdm
import numpy as np
from config import Config
from data_loader import create_dataloaders
from model import create_model, save_model_checkpoint
from utils import create_confusion_matrix, calculate_class_wise_accuracy, plot_class_wise_accuracy, log_sample_predictions

class STL10Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Initialize wandb
        os.environ["WANDB_API_KEY"] = config.WANDB_API_KEY
        wandb.init(project=config.WANDB_PROJECT, entity=config.WANDB_ENTITY)
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': correct_predictions / total_samples
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct
```

