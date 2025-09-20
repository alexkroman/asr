#!/usr/bin/env python3
"""
Run evaluation on a remote RunPod server.
Evaluates a checkpoint and calculates WER on validation data.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_remote_eval(
    host: str,
    port: int,
    checkpoint: str = "latest",
    num_samples: int = 10,
    dataset: str = "librispeech_asr",
    config: str = "clean",
    split: str = "validation",
    batch_size: int = 1,
):
    """Run evaluation on the remote server."""

    # Build the remote command
    if checkpoint == "latest":
        # Find the latest checkpoint across all timestamp directories
        find_cmd = (
            f"ls -d /workspace/outputs/*/*/outputs/production_model/checkpoint-* 2>/dev/null | "
            f"sort -t'-' -k2 -n | tail -1"
        )

        print(f"🔍 Finding latest checkpoint on {host}...")
        result = subprocess.run(
            ["ssh", "-p", str(port), f"root@{host}", find_cmd],
            capture_output=True,
            text=True
        )

        if result.returncode != 0 or not result.stdout.strip():
            # Try without timestamp structure as fallback
            find_cmd = (
                f"ls -d /workspace/outputs/production_model/checkpoint-* 2>/dev/null | "
                f"sort -t'-' -k2 -n | tail -1"
            )
            result = subprocess.run(
                ["ssh", "-p", str(port), f"root@{host}", find_cmd],
                capture_output=True,
                text=True
            )

            if result.returncode != 0 or not result.stdout.strip():
                print("❌ No checkpoints found in /workspace/outputs/")
                print("   Searched in:")
                print("   - /workspace/outputs/YYYY-MM-DD/HH-MM-SS/outputs/production_model/")
                print("   - /workspace/outputs/production_model/")
                return 1

        checkpoint = result.stdout.strip()
        print(f"📦 Found checkpoint: {checkpoint}")
    else:
        # Use specified checkpoint path
        if not checkpoint.startswith("/"):
            # Try to find the checkpoint in timestamp directories first
            find_cmd = (
                f"ls -d /workspace/outputs/*/*/outputs/production_model/{checkpoint} 2>/dev/null | "
                f"head -1"
            )
            result = subprocess.run(
                ["ssh", "-p", str(port), f"root@{host}", find_cmd],
                capture_output=True,
                text=True
            )

            if result.stdout.strip():
                checkpoint = result.stdout.strip()
            else:
                # Fallback to direct path
                checkpoint = f"/workspace/outputs/production_model/{checkpoint}"

    # Build the evaluation command using accelerate
    eval_cmd = (
        f"cd /workspace && "
        f"python3 -m accelerate.commands.launch --mixed_precision fp16 "
        f"src/eval.py {checkpoint} "
        f"--dataset {dataset} "
        f"--config {config} "
        f"--split {split} "
        f"--num-samples {num_samples} "
        f"--batch-size {batch_size}"
    )

    print(f"\n🚀 Running evaluation on {host}:{port}")
    print(f"📊 Dataset: {dataset} ({config}) - {split} split")
    print(f"🔢 Samples: {num_samples}, Batch size: {batch_size}")
    print(f"💾 Checkpoint: {checkpoint}")
    print("\n" + "="*60 + "\n")

    # Run the evaluation
    ssh_command = ["ssh", "-p", str(port), f"root@{host}", eval_cmd]

    try:
        result = subprocess.run(ssh_command, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error running evaluation: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation on a remote RunPod server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate latest checkpoint with defaults
  %(prog)s 192.168.1.100 22222

  # Evaluate specific checkpoint
  %(prog)s 192.168.1.100 22222 --checkpoint checkpoint-5000

  # Evaluate with more samples
  %(prog)s 192.168.1.100 22222 --num-samples 100 --batch-size 4

  # Evaluate on LibriSpeech "other" test set
  %(prog)s 192.168.1.100 22222 --config other --split test

  # Full path to checkpoint
  %(prog)s 192.168.1.100 22222 --checkpoint /workspace/outputs/my_model/checkpoint-1000
"""
    )

    parser.add_argument("host", help="RunPod instance IP or hostname")
    parser.add_argument("port", type=int, help="SSH port (usually 22222)")

    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="Checkpoint to evaluate (default: latest). Can be 'latest', 'checkpoint-N', or full path"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to evaluate (default: 10)"
    )
    parser.add_argument(
        "--dataset",
        default="librispeech_asr",
        help="Dataset to use (default: librispeech_asr)"
    )
    parser.add_argument(
        "--config",
        default="clean",
        help="Dataset config (default: clean). Options: clean, other"
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split (default: validation). Options: validation, test"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation (default: 1)"
    )

    args = parser.parse_args()

    sys.exit(run_remote_eval(
        args.host,
        args.port,
        args.checkpoint,
        args.num_samples,
        args.dataset,
        args.config,
        args.split,
        args.batch_size
    ))


if __name__ == "__main__":
    main()