    # Fixed Model Creation Cell - Copy this to your Colab

# Create ResNet-18 model
def create_model():
    """Create ResNet-18 model for STL-10 classification"""
    model = models.resnet18(pretrained=config.PRETRAINED)
    
    # Modify final layer for STL-10 (10 classes)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, config.NUM_CLASSES)
    
    return model

# Create model and move to device
model = create_model().to(config.DEVICE)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

# Fixed scheduler without verbose parameter
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=2
)

print("✅ Model created successfully!")
print(f"🧠 Model: ResNet-18")
print(f"📊 Output classes: {config.NUM_CLASSES}")
print(f"🎯 Device: {next(model.parameters()).device}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📈 Total parameters: {total_params:,}")
print(f"🔧 Trainable parameters: {trainable_params:,}")

# Print PyTorch version for debugging
print(f"🔧 PyTorch version: {torch.__version__}")