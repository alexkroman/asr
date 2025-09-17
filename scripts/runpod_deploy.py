#!/usr/bin/env python3
"""
RunPod deployment script using their API
"""

import os
import json
import argparse
import requests
from datetime import datetime
from typing import Dict, Any

# RunPod API configuration
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_API_URL = "https://api.runpod.ai/v2"


def create_pod_config(environment: str = "production") -> Dict[str, Any]:
    """Create RunPod configuration"""

    # Base configuration
    config = {
        "name": f"asr-training-{environment}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "imageUrl": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        "gpuType": "NVIDIA A40" if environment == "production" else "NVIDIA RTX 3090",
        "gpuCount": 1,
        "containerDiskSize": 50,
        "volumeSize": 100,
        "minVcpu": 8,
        "minRam": 32,
        "env": {
            "HF_TOKEN": os.getenv("HF_TOKEN", ""),
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
            "EXPERIMENT": environment,
        },
        "dockerArgs": "",
        "onStartCommand": """
            # Install dependencies
            cd /workspace
            wget -O deployment.tar.gz $DEPLOYMENT_URL
            tar -xzf deployment.tar.gz
            pip install -r requirements.txt

            # Start training
            python src/train.py +experiments={environment}
        """.format(environment=environment),
        "volumeMountPath": "/workspace",
        "ports": "6006/tcp,8888/tcp",  # TensorBoard and Jupyter
    }

    # Environment-specific settings
    if environment == "production":
        config.update({
            "gpuCount": 4,
            "minVcpu": 32,
            "minRam": 128,
        })
    elif environment == "test":
        config.update({
            "gpuType": "NVIDIA RTX 3090",
            "stopAfter": "2h",  # Auto-stop after 2 hours
        })

    return config


def deploy_to_runpod(config: Dict[str, Any]) -> str:
    """Deploy to RunPod via API"""

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }

    # Create pod
    response = requests.post(
        f"{RUNPOD_API_URL}/pods",
        headers=headers,
        json=config
    )

    if response.status_code != 200:
        raise Exception(f"Failed to create pod: {response.text}")

    result = response.json()
    pod_id = result["id"]

    print(f"✅ Pod created: {pod_id}")
    print(f"📊 Dashboard: https://www.runpod.io/console/pods/{pod_id}")

    return pod_id


def get_pod_status(pod_id: str) -> Dict[str, Any]:
    """Get status of a RunPod instance"""

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
    }

    response = requests.get(
        f"{RUNPOD_API_URL}/pods/{pod_id}",
        headers=headers
    )

    if response.status_code != 200:
        raise Exception(f"Failed to get pod status: {response.text}")

    return response.json()


def stop_pod(pod_id: str) -> None:
    """Stop a RunPod instance"""

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
    }

    response = requests.post(
        f"{RUNPOD_API_URL}/pods/{pod_id}/stop",
        headers=headers
    )

    if response.status_code != 200:
        raise Exception(f"Failed to stop pod: {response.text}")

    print(f"✅ Pod {pod_id} stopped")


def main():
    parser = argparse.ArgumentParser(description="Deploy training to RunPod")
    parser.add_argument("--environment", default="production",
                       choices=["production", "test", "dev"])
    parser.add_argument("--action", default="deploy",
                       choices=["deploy", "status", "stop"])
    parser.add_argument("--pod-id", help="Pod ID for status/stop actions")

    args = parser.parse_args()

    if not RUNPOD_API_KEY:
        print("❌ Error: RUNPOD_API_KEY not set")
        return 1

    if args.action == "deploy":
        config = create_pod_config(args.environment)
        pod_id = deploy_to_runpod(config)

        # Save pod ID for later reference
        with open(".last_pod_id", "w") as f:
            f.write(pod_id)

    elif args.action == "status":
        pod_id = args.pod_id or open(".last_pod_id").read().strip()
        status = get_pod_status(pod_id)
        print(json.dumps(status, indent=2))

    elif args.action == "stop":
        pod_id = args.pod_id or open(".last_pod_id").read().strip()
        stop_pod(pod_id)

    return 0


if __name__ == "__main__":
    exit(main())