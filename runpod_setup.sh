#!/bin/bash

# RunPod ASR Training Setup Script
echo "🚀 Setting up ASR training environment on RunPod..."

# Detect GPU count
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "📊 Detected $GPU_COUNT GPU(s)"

# Update system packages
apt-get update && apt-get install -y \
    git \
    wget \
    sox \
    libsndfile1 \
    ffmpeg \
    screen \
    htop \
    nvtop

# Install Python packages
pip install --upgrade pip
pip install --upgrade \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
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

# Setup Accelerate configuration based on GPU count
if [ $GPU_COUNT -gt 1 ]; then
    echo "🔧 Configuring Accelerate for multi-GPU training..."
    cat > /workspace/accelerate_config.yaml <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: $GPU_COUNT
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
else
    echo "🔧 Configuring Accelerate for single GPU training..."
    cat > /workspace/accelerate_config.yaml <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: '0'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
fi

# Create workspace directories
mkdir -p /workspace/ASR_Conformer_SmolLM2_Optimized/{checkpoints,models,logs,cache}

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT-1)))
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export NCCL_P2P_DISABLE=0
export HF_DATASETS_CACHE=/workspace/ASR_Conformer_SmolLM2_Optimized/cache
export TRANSFORMERS_CACHE=/workspace/ASR_Conformer_SmolLM2_Optimized/cache

# Create training launcher script
cat > /workspace/launch_training.sh <<'EOF'
#!/bin/bash

# Check for Hugging Face token
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  Warning: HF_TOKEN not set. Model upload will be skipped."
    echo "Set it with: export HF_TOKEN='your_token_here'"
fi

# Check for WandB token (optional)
if [ -z "$WANDB_API_KEY" ]; then
    echo "ℹ️  WandB API key not set. Using TensorBoard only."
fi

# Detect GPU count for launch command
GPU_COUNT=$(nvidia-smi -L | wc -l)

echo "🎯 Starting ASR training with $GPU_COUNT GPU(s)..."

if [ $GPU_COUNT -gt 1 ]; then
    echo "🚀 Launching multi-GPU training with Accelerate..."
    accelerate launch --config_file /workspace/accelerate_config.yaml /workspace/train.py
else
    echo "🚀 Launching single GPU training..."
    python /workspace/train.py
fi
EOF

chmod +x /workspace/launch_training.sh

# Create monitoring script
cat > /workspace/monitor.sh <<'EOF'
#!/bin/bash
watch -n 1 'nvidia-smi; echo ""; free -h; echo ""; df -h /workspace'
EOF
chmod +x /workspace/monitor.sh

echo "✅ RunPod setup complete!"
echo ""
echo "📝 Quick Start Guide:"
echo "1. Set your Hugging Face token: export HF_TOKEN='your_token_here'"
echo "2. (Optional) Set WandB token: export WANDB_API_KEY='your_key_here'"
echo "3. Start training: ./launch_training.sh"
echo "4. Monitor GPUs: ./monitor.sh"
echo ""
echo "📊 GPU Information:"
nvidia-smi