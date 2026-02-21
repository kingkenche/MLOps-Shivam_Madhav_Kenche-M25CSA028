# Base image with Python (CPU-only; swap tag for CUDA if needed)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY train.py .
COPY evaluate.py .
COPY data/ ./data/

# Default command — run training then evaluation
CMD ["sh", "-c", "python train.py && python evaluate.py"]
