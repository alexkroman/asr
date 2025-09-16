#!/bin/bash

# Install system dependencies for ASR project

echo "Installing system dependencies..."

# Update package list
apt update

# Install required packages
apt install -y \
    tmux \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev

echo "Dependencies installed successfully!"