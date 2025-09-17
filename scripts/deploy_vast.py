#!/usr/bin/env python3
"""
Vast.ai deployment script for budget-friendly GPU training
"""

import os
import json
import subprocess
import argparse
from typing import Dict, Any

# Vast.ai configuration
VAST_API_KEY = os.getenv("VAST_API_KEY")


def find_best_instance(gpu_type: str = "RTX_4090", max_price: float = 1.0) -> Dict[str, Any]:
    """Find best Vast.ai instance based on requirements"""

    # Query available instances
    cmd = [
        "vastai", "search", "offers",
        f"gpu_name={gpu_type}",
        f"dph<{max_price}",
        "disk_space>50",
        "inet_up>100",
        "reliability>0.95",
        "--raw"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    offers = json.loads(result.stdout)

    if not offers:
        raise Exception(f"No instances found for {gpu_type} under ${max_price}/hour")

    # Sort by price and reliability
    offers.sort(key=lambda x: (x['dph_total'], -x['reliability']))

    return offers[0]


def deploy_to_vast(instance_id: str, environment: str = "production") -> str:
    """Deploy to Vast.ai instance"""

    # Create instance with Docker image
    cmd = [
        "vastai", "create", "instance", str(instance_id),
        "--image", "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
        "--disk", "50",
        "--env", f"EXPERIMENT={environment}",
        "--env", f"HF_TOKEN={os.getenv('HF_TOKEN', '')}",
        "--env", f"WANDB_API_KEY={os.getenv('WANDB_API_KEY', '')}",
        "--onstart-cmd", f"""
            cd /workspace && \
            git clone https://github.com/yourusername/asr.git && \
            cd asr && \
            pip install -r requirements.txt && \
            python src/train.py +experiments={environment}
        """,
        "--raw"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    instance_info = json.loads(result.stdout)

    instance_id = instance_info['new_instance_id']
    print(f"✅ Instance created: {instance_id}")

    return instance_id


def get_instance_info(instance_id: str) -> Dict[str, Any]:
    """Get Vast.ai instance information"""

    cmd = ["vastai", "show", "instance", str(instance_id), "--raw"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    return json.loads(result.stdout)


def monitor_training(instance_id: str) -> None:
    """Monitor training logs from Vast.ai instance"""

    print(f"📊 Monitoring instance {instance_id}...")
    cmd = ["vastai", "logs", str(instance_id), "--tail", "100", "-f"]
    subprocess.run(cmd)


def stop_instance(instance_id: str) -> None:
    """Stop Vast.ai instance"""

    cmd = ["vastai", "destroy", "instance", str(instance_id)]
    subprocess.run(cmd)
    print(f"✅ Instance {instance_id} stopped")


def main():
    parser = argparse.ArgumentParser(description="Deploy training to Vast.ai")
    parser.add_argument("--environment", default="production",
                       choices=["production", "test", "dev"])
    parser.add_argument("--action", default="deploy",
                       choices=["deploy", "monitor", "stop", "find"])
    parser.add_argument("--gpu", default="RTX_4090",
                       choices=["RTX_4090", "RTX_3090", "A100", "H100"])
    parser.add_argument("--max-price", type=float, default=1.0,
                       help="Maximum price per hour in USD")
    parser.add_argument("--instance-id", help="Instance ID for monitor/stop actions")

    args = parser.parse_args()

    if not VAST_API_KEY:
        print("❌ Error: VAST_API_KEY not set")
        print("Get your API key from: https://vast.ai/account")
        return 1

    os.environ["VAST_API_KEY"] = VAST_API_KEY

    if args.action == "find":
        offer = find_best_instance(args.gpu, args.max_price)
        print(f"Best offer found:")
        print(f"  GPU: {offer['gpu_name']} x{offer['num_gpus']}")
        print(f"  Price: ${offer['dph_total']:.2f}/hour")
        print(f"  RAM: {offer['cpu_ram']:.0f}GB")
        print(f"  Reliability: {offer['reliability']:.2%}")
        print(f"  ID: {offer['id']}")

    elif args.action == "deploy":
        offer = find_best_instance(args.gpu, args.max_price)
        print(f"Deploying to: {offer['gpu_name']} @ ${offer['dph_total']:.2f}/hr")
        instance_id = deploy_to_vast(offer['id'], args.environment)

        # Save instance ID
        with open(".last_vast_instance", "w") as f:
            f.write(str(instance_id))

    elif args.action == "monitor":
        instance_id = args.instance_id or open(".last_vast_instance").read().strip()
        monitor_training(instance_id)

    elif args.action == "stop":
        instance_id = args.instance_id or open(".last_vast_instance").read().strip()
        stop_instance(instance_id)

    return 0


if __name__ == "__main__":
    exit(main())