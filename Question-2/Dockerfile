# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile – Assignment 5, Q2: Adversarial Attacks using IBM ART
# Base: PyTorch 2.2 + CUDA 12.1 (change tag if your GPU driver differs)
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer-cache friendly)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create directories that the code expects
RUN mkdir -p checkpoints results data

# Default command – override at runtime
CMD ["bash"]
