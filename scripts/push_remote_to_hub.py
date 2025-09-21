#!/usr/bin/env python3
"""
Push a model checkpoint from remote RunPod server to Hugging Face Hub.

Usage:
    python scripts/push_remote_to_hub.py <IP_ADDRESS> <PORT> <checkpoint_path> [--repo-id mazesmazes/asr]
"""

import argparse
import subprocess
import sys
import os


def push_remote_to_hub(
    ip_address: str,
    port: str,
    checkpoint_path: str,
    repo_id: str = "mazesmazes/asr",
    commit_message: str = "Update ASR model",
    private: bool = False,
):
    """Push model from remote server to HuggingFace Hub."""

    # Build the SSH command
    ssh_base = ["ssh", "-p", port, f"root@{ip_address}"]

    print(f"🔌 Connecting to {ip_address}:{port}")

    # First check if the checkpoint exists on remote
    # Normalize the path - remove /workspace prefix if it's duplicated
    if checkpoint_path.startswith('/workspace/'):
        normalized_path = checkpoint_path
    else:
        normalized_path = f"/workspace/{checkpoint_path.lstrip('/')}"

    check_cmd = ssh_base + [f"test -d {normalized_path} && echo 'EXISTS' || echo 'NOT_FOUND'"]
    print(f"\n📁 Checking if checkpoint exists on remote: {normalized_path}")
    result = subprocess.run(check_cmd, capture_output=True, text=True)

    if "NOT_FOUND" in result.stdout:
        print(f"❌ Error: Checkpoint path not found on remote: {normalized_path}")
        print("\n📂 Available checkpoints on remote:")
        list_cmd = ssh_base + ["find /workspace/outputs -name 'checkpoint-*' -type d 2>/dev/null | head -20"]
        subprocess.run(list_cmd)
        sys.exit(1)

    # Use normalized path for the rest of the script
    checkpoint_path = normalized_path

    # Get HF token from local environment
    hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN') or ''

    if not hf_token:
        print("\n⚠️  No HuggingFace token found in local environment.")
        print("\nTo set it up:")
        print("1. Run locally: huggingface-cli login")
        print("2. Or export HUGGING_FACE_HUB_TOKEN='your_token'")
        print("3. Or export HF_TOKEN='your_token'")

        response = input("\nContinue anyway? The push might fail without authentication. (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        print("✅ Using HuggingFace token from local environment")

    # Build the push command
    private_flag = "--private" if private else ""

    push_command = f"""
export HUGGING_FACE_HUB_TOKEN='{hf_token}'
export HF_TOKEN='{hf_token}'
cd /workspace && python3 -c "
import sys
import os
sys.path.append('/workspace')
from src.train import ASRModel
from transformers import AutoTokenizer, WhisperFeatureExtractor
from huggingface_hub import HfApi, create_repo
from pathlib import Path

# Set token in Python environment as well
os.environ['HUGGING_FACE_HUB_TOKEN'] = '{hf_token}'
os.environ['HF_TOKEN'] = '{hf_token}'

checkpoint_path = Path('{checkpoint_path}')
repo_id = '{repo_id}'
commit_message = '{commit_message}'
private = {private}

print('Loading model from', checkpoint_path)
try:
    model = ASRModel.from_pretrained(checkpoint_path)
    tokenizer = model.decoder.tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained('openai/whisper-small')

    print('Creating repository...')
    try:
        create_repo(repo_id, private=private, exist_ok=True)
    except Exception as e:
        print(f'Note: {{e}}')

    print('Pushing model...')
    model.push_to_hub(repo_id, commit_message=commit_message, safe_serialization=True)

    print('Pushing tokenizer...')
    tokenizer.push_to_hub(repo_id, commit_message=commit_message)

    print('Pushing feature extractor...')
    feature_extractor.push_to_hub(repo_id, commit_message=commit_message)

    print('✅ Successfully pushed to https://huggingface.co/' + repo_id)

except Exception as e:
    print(f'❌ Error: {{e}}')
    # Try loading from subdirectory
    import glob
    checkpoints = sorted(glob.glob(str(checkpoint_path / 'checkpoint-*')))
    if checkpoints:
        latest = checkpoints[-1]
        print(f'Trying to load from {{latest}}...')
        model = ASRModel.from_pretrained(latest)
        tokenizer = model.decoder.tokenizer
        feature_extractor = WhisperFeatureExtractor.from_pretrained('openai/whisper-small')

        create_repo(repo_id, private=private, exist_ok=True)
        model.push_to_hub(repo_id, commit_message=commit_message, safe_serialization=True)
        tokenizer.push_to_hub(repo_id, commit_message=commit_message)
        feature_extractor.push_to_hub(repo_id, commit_message=commit_message)
        print('✅ Successfully pushed to https://huggingface.co/' + repo_id)
    else:
        raise
"
"""

    print(f"\n🚀 Pushing model to {repo_id}...")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Message: {commit_message}\n")

    # Execute the push command
    push_cmd = ssh_base + [push_command]
    result = subprocess.run(push_cmd)

    if result.returncode != 0:
        print("\n❌ Push failed!")
        print("\nTroubleshooting:")
        print("1. Make sure you have a valid HF token set locally:")
        print("   export HUGGING_FACE_HUB_TOKEN='your_token'")
        print("   Or run: huggingface-cli login")
        print("2. Check that the checkpoint path is correct")
        print("3. Ensure the model checkpoint contains required files (config.json, model.safetensors)")
        sys.exit(1)

    print(f"\n✨ Model successfully pushed to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Push model checkpoint from remote RunPod to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Push latest checkpoint from production training
  python scripts/push_remote_to_hub.py 192.168.1.100 22222 /workspace/outputs/production_model/checkpoint-5000

  # Push to custom repo
  python scripts/push_remote_to_hub.py pod.runpod.io 22222 /workspace/outputs/my_model \\
    --repo-id username/my-asr-model

  # Push with custom commit message
  python scripts/push_remote_to_hub.py 192.168.1.100 22222 /workspace/outputs/production_model \\
    --commit-message "Fine-tuned on custom dataset"

  # Make repository private
  python scripts/push_remote_to_hub.py 192.168.1.100 22222 /workspace/outputs/model --private
"""
    )

    parser.add_argument(
        "ip_address",
        type=str,
        help="IP address or hostname of the RunPod instance"
    )
    parser.add_argument(
        "port",
        type=str,
        help="SSH port of the RunPod instance"
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to checkpoint on remote (e.g., /workspace/outputs/production_model/checkpoint-5000)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="mazesmazes/asr",
        help="HuggingFace repository ID (default: mazesmazes/asr)"
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Update ASR model",
        help="Commit message for the push"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )

    args = parser.parse_args()

    # Normalize checkpoint path
    if not args.checkpoint_path.startswith("/"):
        args.checkpoint_path = f"/workspace/{args.checkpoint_path}"

    push_remote_to_hub(
        ip_address=args.ip_address,
        port=args.port,
        checkpoint_path=args.checkpoint_path,
        repo_id=args.repo_id,
        commit_message=args.commit_message,
        private=args.private,
    )


if __name__ == "__main__":
    main()