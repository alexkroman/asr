#!/usr/bin/env python3
"""Quick evaluation script for ASR checkpoints."""

import torch
import sys
from pathlib import Path
from src.train import (
    load_checkpoint,
    parse_config,
    load_datasets,
    setup_model_and_tokenizer,
    setup_trainer
)
from transformers import set_seed
from accelerate import Accelerator

def main():
    # Parse config
    config_file = sys.argv[1] if len(sys.argv) > 1 else "configs/experiments/mac_minimal.json"
    checkpoint_dir = sys.argv[2] if len(sys.argv) > 2 else "./mac_minimal_output/checkpoint-1000"

    print(f"📊 Evaluating checkpoint: {checkpoint_dir}")
    print(f"📋 Using config: {config_file}")

    # Initialize accelerator
    accelerator = Accelerator()

    # Parse configuration
    model_args, data_args, training_args = parse_config(config_file, accelerator)

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_args, accelerator)

    # Load checkpoint
    if not load_checkpoint(checkpoint_dir, model, accelerator):
        print("❌ Failed to load checkpoint")
        return

    print("✅ Checkpoint loaded successfully")

    # Load datasets
    train_dataset, val_dataset = load_datasets(data_args, accelerator)

    # Setup trainer
    trainer = setup_trainer(
        model, tokenizer, train_dataset, val_dataset,
        training_args, data_args, model_args, accelerator
    )

    # Run evaluation
    print("🔍 Running evaluation...")
    metrics = trainer.evaluate()

    # Print results
    print("\n📊 Evaluation Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Optional: Generate a few sample predictions
    print("\n🎤 Generating sample predictions...")
    from src.train import show_sample_predictions
    show_sample_predictions(
        model, val_dataset, tokenizer,
        trainer.data_collator, accelerator.device,
        num_samples=3
    )

if __name__ == "__main__":
    main()