# Training Deployment Guide

## Overview

Multiple deployment options are available depending on your needs and budget:

| Method | Cost | GPUs | Best For |
|--------|------|------|----------|
| **Local** | Free | Your GPU | Development & testing |
| **RunPod** | $$$ | A100/H100 | Production training |
| **Vast.ai** | $ | RTX 4090 | Budget training |
| **Docker** | Varies | Any | Reproducible deployments |
| **Own Server** | Fixed | Your GPUs | Long-term training |

## Quick Start

```bash
# Install deployment tools
pip install -r requirements.txt

# Deploy based on your needs
make deploy-local      # Test locally
make deploy-vast       # Budget GPU ($0.5-1/hr)
make deploy-runpod     # Premium GPU ($2-8/hr)
make deploy-server     # Your own server
```

## Deployment Methods

### 1. Local Development

Best for testing and small experiments:

```bash
# Quick test
make train-test

# Mac optimized
make train-mac

# Full local training
python src/train.py +experiments=production
```

### 2. RunPod (Premium Cloud GPUs)

Best for production training with A100/H100 GPUs:

```bash
# Configure API key
export RUNPOD_API_KEY="your-key-here"

# Deploy
python scripts/runpod_deploy.py --environment production

# Monitor
python scripts/runpod_deploy.py --action status

# Stop when done
python scripts/runpod_deploy.py --action stop
```

**Pricing:**
- RTX 3090: ~$0.44/hr
- A40: ~$0.79/hr
- A100 40GB: ~$1.89/hr
- H100: ~$4.89/hr

### 3. Vast.ai (Budget Cloud GPUs)

Best for budget-conscious training:

```bash
# Install Vast CLI
pip install vastai

# Configure
export VAST_API_KEY="your-key-here"

# Find best deal
python scripts/deploy_vast.py --action find --gpu RTX_4090 --max-price 0.8

# Deploy
python scripts/deploy_vast.py --environment production --gpu RTX_4090

# Monitor logs
python scripts/deploy_vast.py --action monitor
```

**Pricing:**
- RTX 3090: ~$0.20-0.40/hr
- RTX 4090: ~$0.40-0.80/hr
- A100: ~$0.80-1.50/hr

### 4. Docker Deployment

Best for reproducible deployments:

```bash
# Build image
docker build -f Dockerfile.simple -t asr-training .

# Run with Docker Compose
docker compose up train

# Or run directly
docker run --gpus all \
  -v $(pwd)/outputs:/workspace/outputs \
  asr-training +experiments=production
```

### 5. Your Own Server

Best if you have dedicated GPUs:

```bash
# Configure SSH in .env
SSH_HOST=your-server.com
SSH_USER=ubuntu
SSH_PORT=22
SSH_KEY=~/.ssh/id_rsa

# Deploy with rsync (fastest)
./scripts/deploy.sh rsync production

# Deploy with git
./scripts/deploy.sh git production

# Deploy and start training
./scripts/deploy.sh rsync production --start
```

## GitHub Actions Deployment

Push-button deployment via GitHub:

1. Set secrets in GitHub:
   - `RUNPOD_API_KEY`
   - `VAST_API_KEY`
   - `HF_TOKEN`
   - `WANDB_API_KEY`

2. Go to Actions → Deploy Training

3. Select environment and GPU type

4. Click "Run workflow"

## Monitoring

### TensorBoard

```bash
# Local
make tensorboard

# Remote (SSH tunnel)
ssh -L 6006:localhost:6006 user@server
# Then open http://localhost:6006
```

### Weights & Biases

```bash
export WANDB_API_KEY="your-key"
# Training will auto-log to W&B
```

### Logs

```bash
# Follow training logs
make monitor

# Check specific deployment
tail -f outputs/*/training.log
```

## Cost Optimization

### Tips to Reduce Costs:

1. **Use spot/preemptible instances** (up to 70% savings)
2. **Start with Vast.ai** for experiments
3. **Use RunPod only for final training**
4. **Enable gradient checkpointing** to use smaller GPUs
5. **Stop instances immediately** after training

### Cost Comparison (per training run):

| Dataset | RTX 4090 (Vast) | A100 (RunPod) | H100 (RunPod) |
|---------|-----------------|---------------|---------------|
| Small (1K samples) | ~$2 | ~$5 | ~$10 |
| Medium (10K samples) | ~$15 | ~$40 | ~$80 |
| Large (100K samples) | ~$100 | ~$300 | ~$500 |

## Troubleshooting

### Out of Memory

```yaml
# Reduce batch size in config
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32
  gradient_checkpointing: true
```

### Slow Training

```bash
# Use multiple GPUs
accelerate launch --multi_gpu --num_processes 4 \
  src/train.py +experiments=production
```

### Connection Issues

```bash
# Use tmux/screen for persistent sessions
tmux new -s training
./scripts/train_server.sh production
# Ctrl+B, D to detach
```

## Best Practices

1. **Always test locally first** with small data
2. **Use version control** for configs
3. **Monitor costs** closely on cloud platforms
4. **Set up automatic stops** to prevent overruns
5. **Use checkpointing** to resume interrupted training
6. **Store outputs in cloud storage** (S3/GCS)

## Quick Commands Reference

```bash
# Development
make train-test           # Quick test
make lint                # Code quality
make test               # Run tests

# Deployment
make deploy-local       # Local GPU
make deploy-vast        # Budget cloud
make deploy-runpod      # Premium cloud
make deploy-docker      # Container

# Management
make monitor            # Watch logs
make tensorboard        # Visualize metrics
make stop              # Stop training
make clean             # Clean outputs
```

## Security Notes

1. Never commit API keys - use `.env` file
2. Use SSH keys, not passwords
3. Encrypt model checkpoints if sensitive
4. Set resource limits to prevent overcharges
5. Use private Docker registries for proprietary code