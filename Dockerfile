# ── Dockerfile ─────────────────────────────────────────────────────────────────
# Development Docker image for Assignment 3 (Task 2)
# Includes all dependencies for training & evaluation.
# ───────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Default command: run the full training pipeline
CMD ["python", "main.py"]
