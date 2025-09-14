#!/bin/bash

apt update
echo "⬆️  Upgrading all system packages..."
apt upgrade -y
apt install -y ffmpeg libsndfile1 libsndfile1-dev
pip install --upgrade pip
cd .. && pip install -e ".[cuda,optimized]" && cd scripts
pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
