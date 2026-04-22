import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import torchmetrics
import matplotlib.pyplot as plt
import argparse

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CityScapeDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=(256, 256)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path) # It has 4 channels, label in channel 0

        # Resize
        image = image.resize(self.img_size, Image.BILINEAR)
        mask = mask.resize(self.img_size, Image.NEAREST)

        # Convert to numpy
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask)
        
        # Mask target is in channel 0
        mask = mask[:, :, 0].astype(np.int64)

        # HWC to CHW for image
        image = np.transpose(image, (2, 0, 1))

        return torch.tensor(image), torch.tensor(mask)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    rgb_dir = "data/CameraRGB/"
    mask_dir = "data/CameraMask/"

    img_filenames = sorted(os.listdir(rgb_dir))
    mask_filenames = sorted(os.listdir(mask_dir))

    # Verify matching
    assert len(img_filenames) == len(mask_filenames)
    for i, m in zip(img_filenames, mask_filenames):
        assert i == m

    img_paths = [os.path.join(rgb_dir, f) for f in img_filenames]
    mask_paths = [os.path.join(mask_dir, f) for f in mask_filenames]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(img_paths, mask_paths, test_size=0.2, random_state=42)

    # Dataloaders
    train_dataset = CityScapeDataset(X_train, y_train, img_size=(128, 128))
    test_dataset = CityScapeDataset(X_test, y_test, img_size=(128, 128))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=23,
    )
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Metrics
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=23).to(device)
    dice = torchmetrics.F1Score(task="multiclass", num_classes=23, average='macro').to(device)

    # Training Loop
    epochs = 15
    history = {'loss': [], 'miou': [], 'mdice': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_miou = 0.0
        train_mdice = 0.0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            # compute metrics
            preds = torch.argmax(outputs, dim=1)
            train_miou += jaccard(preds, masks).item()
            train_mdice += dice(preds, masks).item()

        num_batches = len(train_loader)
        epoch_loss = train_loss / num_batches
        epoch_miou = train_miou / num_batches
        epoch_mdice = train_mdice / num_batches

        history['loss'].append(epoch_loss)
        history['miou'].append(epoch_miou)
        history['mdice'].append(epoch_mdice)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - mIOU: {epoch_miou:.4f} - mDice: {epoch_mdice:.4f}")

    # Save model
    torch.save(model.state_dict(), "unet_model.pth")

    # Save plots
    os.makedirs("Question2", exist_ok=True)
    
    plt.figure()
    plt.plot(range(1, epochs+1), history['loss'], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig('Question2/training_loss.png')
    
    plt.figure()
    plt.plot(range(1, epochs+1), history['miou'], label='mIOU')
    plt.plot(range(1, epochs+1), history['mdice'], label='mDice')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title('mIOU and mDice Scores during Training')
    plt.legend()
    plt.savefig('Question2/metrics.png')

    # Testing
    model.eval()
    test_miou = 0.0
    test_mdice = 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            test_miou += jaccard(preds, masks).item()
            test_mdice += dice(preds, masks).item()
    
    num_test_batches = len(test_loader)
    final_test_miou = test_miou / num_test_batches
    final_test_mdice = test_mdice / num_test_batches
    
    print(f"\n--- Test Results ---")
    print(f"Test mIOU: {final_test_miou:.4f}")
    print(f"Test mDice: {final_test_mdice:.4f}")

    # Write to a file so app can read
    with open("Question2/test_metrics.txt", "w") as f:
        f.write(f"mIOU:{final_test_miou:.4f}\n")
        f.write(f"mDICE:{final_test_mdice:.4f}\n")

if __name__ == "__main__":
    main()
