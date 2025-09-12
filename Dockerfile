# RunPod PyTorch container with CUDA support
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    sox \
    libsndfile1 \
    ffmpeg \
    screen \
    htop \
    nvtop \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy training script and requirements
COPY train.py /workspace/train.py
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install flash-attn bitsandbytes  # Install flash-attn for GPU environment

# Set environment variables
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    CUDA_LAUNCH_BLOCKING=0 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=8 \
    HF_DATASETS_CACHE=/workspace/cache \
    TRANSFORMERS_CACHE=/workspace/cache

# Create necessary directories
RUN mkdir -p /workspace/ASR_Conformer_SmolLM2_Optimized/{checkpoints,models,logs,cache}

# Default command
CMD ["/bin/bash"]