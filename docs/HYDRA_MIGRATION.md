# Hydra Migration Guide

This document describes the migration from JSON-based configuration to Hydra for the ASR training pipeline.

## Overview

Hydra provides several advantages over static JSON configs:
- Hierarchical configuration with composition
- Command-line overrides
- Automatic output directory management
- Multi-run capabilities for hyperparameter sweeps
- Configuration validation

## File Structure

```
configs/
├── hydra/                 # New Hydra configs
│   ├── config.yaml       # Main config file
│   ├── model/            # Model configurations
│   │   ├── default.yaml
│   │   ├── small.yaml
│   │   └── large.yaml
│   ├── data/             # Data configurations
│   │   ├── default.yaml
│   │   ├── small.yaml
│   │   └── full.yaml
│   ├── training/         # Training configurations
│   │   ├── default.yaml
│   │   ├── mac.yaml
│   │   └── production.yaml
│   └── experiments/      # Pre-composed experiment configs
│       ├── test.yaml
│       ├── mac_minimal.yaml
│       └── production.yaml
└── experiments/          # Legacy JSON configs (kept for compatibility)
    └── *.json
```

## Usage

### Basic Training

```bash
# Use default configuration
python src/train_hydra.py

# Use a predefined experiment
python src/train_hydra.py +experiments=test
python src/train_hydra.py +experiments=mac_minimal
python src/train_hydra.py +experiments=production

# Or using hatch scripts
hatch run train
hatch run train-test
hatch run train-mac
hatch run train-prod
```

### Command-line Overrides

Hydra allows you to override any configuration parameter from the command line:

```bash
# Override single parameters
python src/train_hydra.py training.learning_rate=1e-4

# Override multiple parameters
python src/train_hydra.py \
    model.lora_r=16 \
    training.per_device_train_batch_size=4 \
    training.max_steps=5000

# Use different model/data/training configs
python src/train_hydra.py \
    model=large \
    data=full \
    training=production
```

### Evaluation

```bash
# Evaluate a checkpoint
python src/train_hydra.py eval_checkpoint=./outputs/model/checkpoint-1000

# With experiment config
python src/train_hydra.py +experiments=test eval_checkpoint=./checkpoint-100
```

### Output Management

Hydra automatically manages output directories:
- Training outputs go to: `outputs/YYYY-MM-DD/HH-MM-SS/`
- Each run gets its own directory with config snapshot
- Logs are automatically organized

### Configuration Composition

You can compose configurations from different sources:

```bash
# Use large model with small data for testing
python src/train_hydra.py model=large data=small training=default

# Mac setup with production model
python src/train_hydra.py model=large data=full training=mac
```

### Creating Custom Experiments

Create a new experiment config in `configs/hydra/experiments/`:

```yaml
# configs/hydra/experiments/my_experiment.yaml
# @package _global_

defaults:
  - override /model: large
  - override /data: full
  - override /training: default

# Custom overrides
training:
  output_dir: ./outputs/my_experiment
  learning_rate: 3e-5
  max_steps: 10000

model:
  lora_r: 16
```

Then run with:
```bash
python src/train_hydra.py +experiments=my_experiment
```

## Multi-run and Sweeps

Hydra supports parameter sweeps:

```bash
# Grid search over learning rates
python src/train_hydra.py -m training.learning_rate=1e-5,3e-5,5e-5

# Sweep over multiple parameters
python src/train_hydra.py -m \
    model.lora_r=8,16,32 \
    training.learning_rate=1e-5,5e-5
```

## Legacy Compatibility

The original JSON-based configuration is still available:

```bash
# Use legacy train.py with JSON configs
python src/train.py configs/experiments/test_config.json

# Or using hatch scripts
hatch run train-legacy configs/experiments/test_config.json
```

## Migration Checklist

- [x] Install Hydra dependencies (`hydra-core`, `omegaconf`)
- [x] Create Hydra configuration structure
- [x] Implement `train_hydra.py` with Hydra support
- [x] Update hatch scripts in `pyproject.toml`
- [x] Keep legacy support for backward compatibility
- [ ] Test with existing experiments
- [ ] Migrate CI/CD pipelines if needed

## Benefits of Migration

1. **Flexibility**: Override any parameter without editing files
2. **Reproducibility**: Automatic config snapshots for each run
3. **Organization**: Structured output directories with timestamps
4. **Experimentation**: Easy parameter sweeps and grid searches
5. **Validation**: Type checking and schema validation
6. **Modularity**: Compose configs from reusable components

## Tips

- Use `+` prefix to add new config groups: `+experiments=test`
- Use `~` prefix to remove config groups
- Use `++` to force override even if key doesn't exist
- Check resolved config with: `python src/train_hydra.py --cfg job`
- Dry run without execution: `python src/train_hydra.py --cfg job --hydra-help`