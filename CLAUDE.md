# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is an ASR (Automatic Speech Recognition) training pipeline that combines a Whisper encoder with SmolLM2/Qwen decoder using PyTorch and Hugging Face transformers. The project uses Hydra for configuration management and supports both local Mac development and cloud GPU training on RunPod.

## Key Commands

### Development & Linting (Local)
- **Lint code**: `uv run ruff check src && uv run mypy src`
- **Format code**: `uv run ruff format src && uv run black src`
- **Run specific test**: `uv run pytest tests/test_file.py::test_function`
- **Run all tests**: `uv run pytest tests/`

### Training
- **Train locally on Mac**: `uv run python src/train.py +experiments=mac_minimal`
- **Train with custom config**: `uv run python src/train.py +experiments=your_experiment`
- **Train on RunPod (uses system Python)**: `python3 -m accelerate launch --config_file configs/accelerate/a40.yaml src/train.py +experiments=production`
- **Resume from checkpoint**: `python3 src/train.py +experiments=production resume_from_checkpoint=./outputs/checkpoint-5000`

### Automatic Evaluation During Training
When TensorBoard is enabled (`report_to: [tensorboard]` in training config), the trainer will automatically:
- Log sample predictions and WER metrics every N training steps (default: 500)
- Display predictions in TensorBoard's TEXT tab
- Track WER over time in TensorBoard's SCALARS tab
- Note: Predictions are logged only at specified step intervals, not during regular evaluation

Configure evaluation frequency:
```yaml
# In your experiment config or via CLI
log_predictions_every_n_steps: 500  # How often to evaluate
log_predictions_samples: 10         # Number of samples to evaluate
```

View TensorBoard logs:
```bash
tensorboard --logdir ./logs
```

### Automatic Hub Pushing During Training
The trainer can automatically push checkpoints to HuggingFace Hub during training:

1. **Set up authentication** (one-time setup):
   ```bash
   huggingface-cli login
   # Or set environment variable:
   export HF_TOKEN='your_token_here'
   ```

2. **Configure in your experiment YAML** or override via CLI:
   ```yaml
   training:
     push_to_hub: true
     hub_model_id: "username/model-name"  # Your HF repo
     hub_strategy: checkpoint  # Push every checkpoint
     hub_private_repo: false  # Set true for private
   ```

3. **Or use CLI overrides**:
   ```bash
   python3 src/train.py +experiments=production \
     training.push_to_hub=true \
     training.hub_model_id="username/my-asr-model" \
     training.hub_strategy=checkpoint
   ```

The production config already has `push_to_hub: true` configured. Just update `hub_model_id` to your repository.

### RunPod Deployment
- **Deploy to RunPod**: `python scripts/deploy_runpod.py <IP_ADDRESS> <PORT>`
- **Example with IP**: `python scripts/deploy_runpod.py 192.168.1.100 22222`
- **Example with hostname**: `python scripts/deploy_runpod.py pod.runpod.io 22222`
- **Skip system setup**: Add `--skip-setup` flag
- **Skip file sync**: Add `--skip-sync` flag
- **Skip dependency install**: Add `--skip-deps` flag

### Remote Training Management
- **Start training**: `python scripts/start_remote_training.py <IP_ADDRESS> <PORT>`
- **Start with custom experiment**: `python scripts/start_remote_training.py <IP> <PORT> --experiment mac_minimal`
- **Start without attaching**: `python scripts/start_remote_training.py <IP> <PORT> --no-attach`
- **Attach to session**: `python scripts/attach_remote_session.py <IP_ADDRESS> <PORT>`
- **List sessions**: `python scripts/attach_remote_session.py <IP> <PORT> --list`
- **View logs without attaching**: `python scripts/attach_remote_session.py <IP> <PORT> --logs`
- **Tmux controls when attached**:
  - Detach (leave running): `Ctrl+B` then `D`
  - Scroll mode: `Ctrl+B` then `[` (exit with `q`)
  - Kill session: `Ctrl+B` then `:` then type `kill-session`

## Architecture

### Core Components
1. **WhisperEncoder** (`src/train.py`): Wraps OpenAI's Whisper-small model for audio encoding. Frozen weights, outputs 768-dim features.

2. **AudioProjector** (`src/train.py`): Projects Whisper features to match decoder's input dimension using a 2-layer MLP with GELU activation and RMSNorm.

3. **ASRModel** (`src/train.py`): Main model combining encoder, projector, and decoder (SmolLM2 or Qwen). Supports LoRA for efficient fine-tuning.

### Configuration System (Hydra)
- **Base config**: `configs/hydra/config.yaml` - defines defaults and output directories
- **Model configs**: `configs/hydra/model/` - defines model architecture (small/medium/large)
- **Data configs**: `configs/hydra/data/` - dataset settings
- **Training configs**: `configs/hydra/training/` - training hyperparameters
- **Experiments**: `configs/hydra/experiments/` - predefined experiment configurations combining the above
- **Accelerate configs**: `configs/accelerate/` - hardware-specific acceleration settings

### Training Pipeline
The training script (`src/train.py`) uses Hugging Face's Trainer API with custom data collation for handling variable-length audio inputs. Key features:
- Dynamic batching with proper padding/masking
- Mixed precision training (bf16 on compatible hardware)
- Gradient checkpointing for memory efficiency
- TensorBoard logging
- Checkpoint management with best model selection

## Environment Variables
For Mac development:
- `PYTORCH_ENABLE_MPS_FALLBACK=1` - Enable fallback for unsupported MPS operations
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` - Optimize MPS memory usage
- `TOKENIZERS_PARALLELISM=false` - Disable parallel tokenization to avoid deadlocks with multiprocessing

For RunPod/CUDA:
- `HF_HOME=/workspace/.cache/huggingface` - Cache directory
- `HF_DATASETS_CACHE=/workspace/datasets` - Dataset cache
- `HF_HUB_ENABLE_HF_TRANSFER=1` - Enable fast downloads

## RunPod Environment
RunPod instances come with PyTorch pre-installed with CUDA support. The training scripts use the system Python (python3) directly rather than a virtual environment to avoid conflicts with the pre-installed CUDA libraries. The deployment script installs only the additional required packages (transformers, accelerate, etc.) via pip3.

## Dataset
The project uses LibriSpeech, GigaSpeech, and Common Voice datasets from Hugging Face.

**Important for GigaSpeech and Common Voice**: These are gated datasets requiring authentication:
1. Accept the license at https://huggingface.co/datasets/speechcolab/gigaspeech
2. Create a token at https://huggingface.co/settings/tokens
3. Export it: `export HUGGING_FACE_HUB_TOKEN='your_token'`
4. Or run: `./scripts/setup_hf_token.sh` for guided setup

Data processing includes:
- Audio resampling to 16kHz
- Mel-spectrogram extraction (80 bins)
- Dynamic sequence length handling with proper masking
- Tokenization using the decoder's tokenizer

## Key Dependencies
- PyTorch ecosystem: torch, torchaudio, torchvision
- Hugging Face: transformers, datasets, accelerate, peft
- Configuration: hydra-core, omegaconf
- Audio: librosa, soundfile
- Evaluation: jiwer (WER calculation), evaluate