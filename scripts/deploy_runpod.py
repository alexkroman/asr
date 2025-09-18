#!/usr/bin/env python3
"""Deploy and sync ASR project to RunPod instance."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True, capture_output=False):
    """Run a shell command."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True, check=check)
    if capture_output:
        return result.stdout.strip()
    return result.returncode


def test_ssh_connection(host, port):
    """Test SSH connection to the RunPod instance."""
    print(f"Testing SSH connection to {host}:{port}...")
    cmd = f"ssh -i ~/.ssh/id_ed25519 -p {port} -o ConnectTimeout=10 -o StrictHostKeyChecking=no root@{host} 'echo Connected successfully'"
    try:
        run_command(cmd)
        print("SSH connection successful!")
        return True
    except subprocess.CalledProcessError:
        print("Failed to connect via SSH. Please check:")
        print("  - The host and port are correct")
        print("  - The pod is running")
        print("  - Your SSH key ~/.ssh/id_ed25519 is added to the pod")
        return False


def setup_remote_dependencies(host, port):
    """Install required system dependencies on the remote instance."""
    print("\nInstalling system dependencies on remote...")

    # Create setup script content
    setup_script = """#!/bin/bash
set -e

echo "Updating package lists..."
apt-get update

echo "Installing system dependencies..."
apt-get install -y ffmpeg tmux rsync curl

echo "System setup complete!"
"""

    # Execute setup script on remote
    cmd = f"""ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no root@{host} 'cat > /tmp/setup.sh && chmod +x /tmp/setup.sh && bash /tmp/setup.sh' <<'EOF'
{setup_script}
EOF"""

    try:
        run_command(cmd)
        print("Remote dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        return False


def sync_project(host, port, project_root):
    """Sync the project files to the RunPod instance."""
    print(f"\nSyncing project from {project_root} to {host}:{port}...")

    # Define exclusions for rsync
    exclusions = [
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        "--exclude=.git",
        "--exclude=.venv",
        "--exclude=venv",
        "--exclude=env",
        "--exclude=.env",
        "--exclude=.claude",  # Don't sync Claude settings
        "--exclude=data/",  # Don't sync data directory
        "--exclude=datasets_cache/",  # Don't sync dataset cache
        "--exclude=outputs",
        "--exclude=logs",
        "--exclude=runs",
        "--exclude=wandb",
        "--exclude=.mypy_cache",
        "--exclude=.pytest_cache",
        "--exclude=.ruff_cache",
        "--exclude=*.egg-info",
        "--exclude=dist",
        "--exclude=build",
        "--exclude=.DS_Store",
        "--exclude=.idea",
        "--exclude=.vscode",
        "--exclude=node_modules",
        "--exclude=.cache",
        "--exclude=datasets",  # Don't sync large dataset caches
        "--exclude=checkpoints",  # Don't sync model checkpoints
        "--exclude=*.ckpt",
        "--exclude=*.pth",
        "--exclude=*.pt",
    ]

    exclusion_str = " ".join(exclusions)

    # Rsync the entire project to /workspace
    rsync_cmd = f"""rsync -avz --delete --no-owner --no-group {exclusion_str} \
        -e "ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no" \
        {project_root}/ root@{host}:/workspace/"""

    try:
        run_command(rsync_cmd)
        print("Project synced successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to sync project: {e}")
        return False


def install_python_dependencies(host, port):
    """Install Python dependencies using system pip on RunPod."""
    print("\nInstalling Python dependencies...")

    # RunPod already has PyTorch installed, we just need the other packages
    print("Checking existing PyTorch installation...")
    cmd_check = f"""ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no root@{host} \
        'python3 -c "import torch; print(\\"PyTorch \\" + torch.__version__ + \\" with CUDA \\" + str(torch.cuda.is_available()))" 2>&1'"""

    try:
        result = run_command(cmd_check, capture_output=True)
        print(f"Found: {result}")
    except subprocess.CalledProcessError:
        print("Warning: Could not verify PyTorch installation")

    # Install only the packages not provided by RunPod
    required_packages = [
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "datasets>=2.14.0",
        "peft>=0.6.0",
        "evaluate>=0.4.0",
        "tensorboard>=2.14.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "hf-transfer",  # For fast HuggingFace downloads
        "ninja",  # For faster CUDA kernel compilation
    ]

    # These packages can speed up training but may require special installation
    optional_packages = [
        # "flash-attn>=2.0.0",  # Requires CUDA compilation, install separately if needed
        # "xformers",  # Memory-efficient transformers, install with: pip3 install xformers --index-url https://download.pytorch.org/whl/cu118
        # "deepspeed",  # For distributed training, needs special setup
    ]

    packages_str = " ".join(f'"{pkg}"' for pkg in required_packages)

    print(f"Installing required packages: {', '.join(required_packages)}")
    cmd = f"""ssh -i ~/.ssh/id_ed25519 -p {port} -o StrictHostKeyChecking=no root@{host} \
        'pip3 install {packages_str} --upgrade 2>&1'"""

    try:
        run_command(cmd)
        print("Python dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install Python dependencies: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Deploy ASR project to RunPod instance")
    parser.add_argument("host", help="RunPod instance IP address (e.g., 192.168.1.100 or pod.runpod.io)")
    parser.add_argument("port", type=int, help="SSH port for the RunPod instance (e.g., 22222)")
    parser.add_argument("--skip-setup", action="store_true",
                       help="Skip system dependency installation")
    parser.add_argument("--skip-sync", action="store_true",
                       help="Skip project file sync")
    parser.add_argument("--skip-deps", action="store_true",
                       help="Skip Python dependency installation")

    args = parser.parse_args()

    # Get project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent.absolute()

    # Test SSH connection
    if not test_ssh_connection(args.host, args.port):
        sys.exit(1)

    # Setup remote dependencies
    if not args.skip_setup:
        if not setup_remote_dependencies(args.host, args.port):
            print("\nWarning: Some system dependencies may not have installed correctly.")
            print("You can continue, but some features might not work.")

    # Sync project files
    if not args.skip_sync:
        if not sync_project(args.host, args.port, project_root):
            sys.exit(1)

    # Install Python dependencies
    if not args.skip_deps:
        if not install_python_dependencies(args.host, args.port):
            print("\nWarning: Python dependencies installation had issues.")
            print("You may need to run 'uv sync' manually on the remote instance.")

    print("\n" + "="*50)
    print("Deployment complete!")
    print(f"You can now SSH into your RunPod instance:")
    print(f"  ssh -i ~/.ssh/id_ed25519 -p {args.port} root@{args.host}")
    print("\nTo start training:")
    print("  cd /workspace")
    print("  python3 src/train.py +experiments=production")
    print("\nOr use the training script:")
    print(f"  python scripts/start_remote_training.py {args.host} {args.port}")
    print("="*50)


if __name__ == "__main__":
    main()