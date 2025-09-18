FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 AS base

RUN apt-get update && apt-get install -y \
    git \
    rsync \
    wget \
    tmux \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install uv for dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./
# Create a minimal src/__init__.py for package installation
RUN mkdir -p src && echo '__version__ = "0.1.0"' > src/__init__.py

# Install dependencies using uv for reproducible builds
RUN uv sync --frozen --extra cuda --extra optimized && \
    uv pip install --no-cache torchcodec --index-url=https://download.pytorch.org/whl/cu121

# Now copy the actual source code - changes here won't invalidate dependency cache
COPY src/ ./src/
COPY configs/ ./configs/

# Set environment variables for CUDA training
ENV HF_HOME=/workspace/.cache/huggingface
ENV HF_DATASETS_CACHE=/workspace/datasets
ENV TORCH_HOME=/workspace/.cache/torch
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_DATASETS_DOWNLOAD_WORKERS=32
ENV HF_DATASETS_IN_MEMORY_MAX_SIZE=0
ENV OMP_NUM_THREADS=9

# Run the training command directly
ENTRYPOINT ["bash", "-c", "OMP_NUM_THREADS=9 accelerate launch --config_file configs/accelerate/a40.yaml src/train.py +experiments=production"]