# RunPod PyTorch container with CUDA support
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

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

# Copy training script and setup files
COPY train.py /workspace/train.py
COPY runpod_setup.sh /workspace/runpod_setup.sh

# Make setup script executable
RUN chmod +x /workspace/runpod_setup.sh

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --upgrade \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --upgrade \
    transformers \
    datasets \
    accelerate \
    tokenizers \
    jiwer \
    huggingface_hub \
    sentencepiece \
    einops \
    peft \
    numpy \
    tensorboard \
    flash-attn \
    evaluate \
    wandb \
    bitsandbytes

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