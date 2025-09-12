#!/bin/bash

# Configuration
TAG="${TAG:-latest}"
FULL_IMAGE_NAME="alexkroman/asr-trainer:$TAG"

echo "Building Docker image: $FULL_IMAGE_NAME"

# Build the Docker image for linux/amd64 platform
docker buildx build --platform linux/amd64 -t "$FULL_IMAGE_NAME" --load .

if [ $? -ne 0 ]; then
    echo "Error: Docker build failed"
    exit 1
fi

echo "Successfully built $FULL_IMAGE_NAME"

# Login to Docker Hub (if not already logged in)
echo "Checking Docker Hub login..."
if ! docker info 2>/dev/null | grep -q "Username"; then
    echo "Please login to Docker Hub:"
    docker login
    if [ $? -ne 0 ]; then
        echo "Error: Docker login failed"
        exit 1
    fi
fi

# Push the image to Docker Hub
echo "Pushing image to Docker Hub..."
docker push "$FULL_IMAGE_NAME"

if [ $? -ne 0 ]; then
    echo "Error: Docker push failed"
    exit 1
fi

echo "Successfully pushed $FULL_IMAGE_NAME to Docker Hub"