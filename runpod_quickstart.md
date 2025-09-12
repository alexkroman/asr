# RunPod ASR Training Quick Start Guide

## 🚀 Quick Deploy on RunPod

### Option 1: Using RunPod Templates (Easiest)

1. **Create a RunPod Pod**:
   - Go to [RunPod](https://runpod.io)
   - Select "GPU Cloud" → "Secure Cloud"
   - Choose GPU type (recommended: RTX 4090, A5000, or A100)
   - Select PyTorch template: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel`

2. **Connect to Pod**:
   ```bash
   ssh root@[your-pod-ip] -p [your-pod-port]
   # Or use RunPod's web terminal
   ```

3. **Setup and Run**:
   ```bash
   # Clone your repository or upload files
   cd /workspace
   
   # Download the files
   wget https://raw.githubusercontent.com/your-repo/train.py
   wget https://raw.githubusercontent.com/your-repo/runpod_setup.sh
   
   # Run setup
   chmod +x runpod_setup.sh
   ./runpod_setup.sh
   
   # Set your tokens
   export HF_TOKEN="your_huggingface_token"
   export WANDB_API_KEY="your_wandb_key"  # Optional
   
   # Start training
   ./launch_training.sh
   ```

### Option 2: Using Docker Image

1. **Build and Push Docker Image** (do this locally):
   ```bash
   # Build image
   docker build -t your-dockerhub-username/asr-training:latest .
   
   # Push to Docker Hub
   docker push your-dockerhub-username/asr-training:latest
   ```

2. **Deploy on RunPod**:
   - Create new pod with custom Docker image
   - Image: `your-dockerhub-username/asr-training:latest`
   - Container Disk: 50GB minimum
   - Volume Disk: 100GB recommended

3. **Run Training**:
   ```bash
   export HF_TOKEN="your_token"
   python /workspace/train.py --push-to-hub
   ```

## 📊 GPU Recommendations

| GPU Type | VRAM | Batch Size | Est. Speed | Cost/hr |
|----------|------|------------|------------|---------|
| RTX 4090 | 24GB | 32 | Fast | ~$0.44 |
| A5000 | 24GB | 32 | Fast | ~$0.79 |
| A100 40GB | 40GB | 64 | Fastest | ~$1.89 |
| A100 80GB | 80GB | 96 | Fastest | ~$3.89 |
| 2x A100 40GB | 80GB | 128 | Ultra Fast | ~$3.78 |

## 🎯 Training Commands

### Basic Training:
```bash
python train.py
```

### With Model Upload to HuggingFace:
```bash
python train.py --push-to-hub --hub-model-id "my-asr-model"
```

### Custom Settings:
```bash
python train.py \
  --max-steps 10000 \
  --eval-steps 500 \
  --batch-size 32 \
  --push-to-hub
```

### Multi-GPU Training:
```bash
# Automatically detected and configured
accelerate launch train.py
```

## 📈 Monitoring

### GPU Usage:
```bash
./monitor.sh  # Real-time GPU, memory, disk monitoring
```

### Training Logs:
```bash
# TensorBoard
tensorboard --logdir /workspace/ASR_Conformer_SmolLM2_Optimized/logs

# Weights & Biases (if configured)
# View at https://wandb.ai/your-username/asr_training
```

### Check Training Progress:
```bash
# View latest checkpoint
ls -la /workspace/ASR_Conformer_SmolLM2_Optimized/checkpoints/

# Tail training logs
tail -f /workspace/ASR_Conformer_SmolLM2_Optimized/logs/events.out.tfevents.*
```

## 🔧 Troubleshooting

### Out of Memory:
```bash
# Reduce batch size
python train.py --batch-size 16

# Or enable gradient checkpointing (edit train.py)
# Set gradient_checkpointing: bool = True in TrainingConfig
```

### CUDA Errors:
```bash
# Reset GPU
nvidia-smi --gpu-reset

# Check CUDA version
nvcc --version
nvidia-smi
```

### Slow Training:
```bash
# Ensure using compiled model (check logs)
# Verify mixed precision is enabled
# Check GPU utilization: nvidia-smi -l 1
```

## 💾 Saving & Resuming

### Auto-saves checkpoints every 250 steps to:
```
/workspace/ASR_Conformer_SmolLM2_Optimized/checkpoints/
```

### Resume from checkpoint:
```bash
# Training automatically resumes from last checkpoint
python train.py
```

### Download Model:
```bash
# From RunPod terminal
cd /workspace/ASR_Conformer_SmolLM2_Optimized/models/
tar -czf final_model.tar.gz final_model/

# Then download via RunPod file browser or scp
```

## 🌐 Environment Variables

```bash
# Required for model upload
export HF_TOKEN="hf_xxxxxxxxxxxxx"

# Optional for experiment tracking
export WANDB_API_KEY="xxxxxxxxxxxxx"

# RunPod automatically sets
echo $RUNPOD_POD_ID  # Your pod ID
echo $CUDA_VISIBLE_DEVICES  # Available GPUs
```

## 📝 Tips for RunPod

1. **Persistent Storage**: Use RunPod's network volumes to persist data between sessions
2. **Spot Instances**: Use spot instances for 50-70% cost savings (may be interrupted)
3. **Multi-GPU**: Scales automatically - just select multi-GPU pod
4. **Preemptible Pods**: Great for testing, not recommended for full training runs
5. **SSH Access**: Set up SSH keys for easier access than web terminal

## 🚨 Important Notes

- Training uses ~50GB for dataset cache on first run
- Model checkpoints use ~2GB each
- Full training on train-clean-100 takes ~12-24 hours on A100
- WER should reach <10% after full training
- Remember to stop your pod when done to avoid charges!