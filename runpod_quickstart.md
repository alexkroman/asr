# RunPod ASR Training Quick Start Guide

## Prerequisites

1. **RunPod Account**: Sign up at [runpod.io](https://runpod.io)
2. **RunPod API Key**: Get from [RunPod Settings](https://www.runpod.io/console/settings) → API Keys
3. **GitHub CLI**: Install with `brew install gh` and authenticate with `gh auth login`
4. **Docker**: Install [Docker Desktop](https://docker.com)

## Quick Deploy

### One-Command Deployment

```bash
# Set your RunPod API key
export RUNPOD_API_KEY="your-api-key-here"

# Deploy everything (build, push to GitHub Container Registry, deploy pod)
./deploy-to-runpod.sh
```

The script will:
1. Auto-install `runpodctl` if needed
2. Build your Docker image
3. Push to GitHub Container Registry (private)
4. Deploy a RunPod pod with your image
5. Show you the pod ID and connection details

### Deployment Options

```bash
# Deploy with spot instance (50-70% cheaper)
./deploy-to-runpod.sh --use-spot --bid-price 0.60

# Deploy with multiple GPUs
./deploy-to-runpod.sh --gpu-type "NVIDIA A100 80GB" --gpu-count 2

# Deploy with custom GPU
./deploy-to-runpod.sh --gpu-type "NVIDIA GeForce RTX 4090"
```

## GPU Recommendations

| Model Size | GPU Type | VRAM | Cost/hr (On-Demand) | Cost/hr (Spot) |
|------------|----------|------|---------------------|-----------------|
| Small | RTX 4090 | 24GB | ~$0.74 | ~$0.44 |
| Medium | A100 40GB | 40GB | ~$1.89 | ~$0.79 |
| Large | A100 80GB | 80GB | ~$3.89 | ~$1.89 |
| XLarge | 2x A100 80GB | 160GB | ~$7.78 | ~$3.78 |

## Managing Your Deployment

### Check Pod Status

```bash
# List all your pods
./deploy-to-runpod.sh --list

# Or use runpodctl directly
runpodctl get pods
```

### Connect to Your Pod

Once deployed, connect via:

1. **Web Terminal**: Go to [RunPod Console](https://www.runpod.io/console/pods) and click "Connect"

2. **SSH** (if you exposed port 22):
```bash
# Get connection details from pod list
ssh root@[pod-ip] -p [ssh-port]
```

### Monitor Training

1. **TensorBoard** (if port 6006 exposed):
   - URL: `https://[pod-id]-6006.proxy.runpod.net`

2. **Logs**:
```bash
# View pod logs
runpodctl logs [pod-id]
```

3. **Weights & Biases** (if configured):
   - View at [wandb.ai](https://wandb.ai)

### Stop Pod

```bash
# Stop when done to avoid charges
./deploy-to-runpod.sh --stop [pod-id]

# Or terminate completely
runpodctl terminate pod [pod-id]
```

## Environment Variables

Set these before deployment:

```bash
# Required
export RUNPOD_API_KEY="your-runpod-api-key"

# Optional
export HF_TOKEN="your-huggingface-token"  # For model uploads
export WANDB_API_KEY="your-wandb-key"     # For experiment tracking
export GITHUB_USER="your-github-username"  # Default: alexkroman
export IMAGE_NAME="custom-image-name"      # Default: asr-training
export POD_NAME="custom-pod-name"         # Default: asr-training
```

## Advanced Usage

### Build and Push Only

```bash
# Just build and push to GHCR, no deployment
./deploy-to-runpod.sh --push-only
```

### Deploy Existing Image

```bash
# Skip build and push, deploy existing image
./deploy-to-runpod.sh --deploy-only --skip-build --skip-push
```

### Custom Configuration

```bash
# Full control
./deploy-to-runpod.sh \
  --gpu-type "NVIDIA A100 80GB" \
  --gpu-count 2 \
  --use-spot \
  --bid-price 1.50
```

## Data Management

### Using Network Volumes

Network volumes persist data across pod restarts:

1. Create a network volume in RunPod console
2. Mount it when creating pod (script handles this automatically)
3. Access at `/workspace` in your pod

### Upload Training Data

```bash
# From local to pod
rsync -avz -e "ssh -p [ssh-port]" \
  ./data/ root@[pod-ip]:/workspace/data/
```

### Download Results

```bash
# From pod to local
rsync -avz -e "ssh -p [ssh-port]" \
  root@[pod-ip]:/workspace/checkpoints/ \
  ./checkpoints/
```

## Troubleshooting

### Authentication Issues

```bash
# Check GitHub CLI auth
gh auth status

# Re-authenticate if needed
gh auth login

# Check RunPod API key
echo $RUNPOD_API_KEY
```

### Pod Creation Failed

1. Check available GPUs: Some GPU types might be out of stock
2. Try spot instances: Often more availability
3. Try different regions: Change in RunPod settings

### Container Pull Errors

The script automatically handles GitHub Container Registry authentication, but if you have issues:

1. Ensure your GitHub token has `read:packages` scope
2. Verify the image was pushed: Check GitHub → Packages
3. Make sure the container is set to private (script does this automatically)

### Common Commands

```bash
# Check GPU availability in pod
nvidia-smi

# Monitor resource usage
htop
nvtop

# Check disk space
df -h

# Test CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

## Cost Optimization Tips

1. **Use Spot Instances**: 50-70% cheaper, good for training
2. **Set Auto-Stop**: Configure in pod settings to stop when idle
3. **Use Network Volumes**: Avoid re-uploading data
4. **Monitor Usage**: Check RunPod dashboard regularly
5. **Stop When Done**: Always stop pods after training

## Quick Example

Complete training workflow:

```bash
# 1. Set credentials
export RUNPOD_API_KEY="runpod_api_..."
export HF_TOKEN="hf_..."

# 2. Deploy with spot instance
./deploy-to-runpod.sh --use-spot --gpu-type "NVIDIA GeForce RTX 4090"

# 3. Monitor training (get pod-id from output)
runpodctl logs [pod-id] --follow

# 4. Stop when complete
./deploy-to-runpod.sh --stop [pod-id]
```

## Support

- **RunPod Documentation**: [docs.runpod.io](https://docs.runpod.io)
- **RunPod Discord**: [discord.gg/runpod](https://discord.gg/runpod)
- **runpodctl CLI**: [github.com/runpod/runpodctl](https://github.com/runpod/runpodctl)