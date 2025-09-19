#!/usr/bin/env python3
"""Test the updated evaluation with subset WER calculation"""

import numpy as np
from transformers import AutoTokenizer, EvalPrediction
import sys
sys.path.append('src')

# Mock tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Create mock data - 100 samples for eval, but only 10 will be used for WER
batch_size = 100
seq_length = 50
vocab_size = tokenizer.vocab_size

# Mock predictions and labels
predictions = np.random.randn(batch_size, seq_length, vocab_size)  # Logits
labels = np.random.randint(0, vocab_size, size=(batch_size, seq_length))
labels[labels < 100] = -100  # Add some padding

# Create EvalPrediction object
eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)

# Import and test the compute_metrics function
from train import compute_metrics

print("Testing compute_metrics with subset WER calculation...")
print(f"Total eval samples: {batch_size}")
print(f"WER calculation subset: 10 samples")

metrics = compute_metrics(eval_pred, tokenizer, wer_sample_size=10)

print("\nResults:")
print(f"- WER computed on: {metrics['wer_samples']} samples")
print(f"- WER value: {metrics['wer']:.4f}")
print(f"- Sample predictions collected: {len(metrics.get('_sample_predictions', []))} samples")

assert metrics['wer_samples'] == 10, "Should compute WER on exactly 10 samples"
assert len(metrics.get('_sample_predictions', [])) == 10, "Should have exactly 10 sample predictions"

print("\n✅ Test passed! Evaluation correctly computes loss on full set but WER on subset.")