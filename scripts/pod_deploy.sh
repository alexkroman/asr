#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Error: .env file not found. Please create one from .env.example"
    exit 1
fi

# Validate required environment variables
if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ] || [ -z "$SSH_USER" ] || [ -z "$SSH_KEY" ]; then
    echo "Error: Missing required environment variables. Check your .env file."
    exit 1
fi

echo "Deploying to $SSH_USER@$SSH_HOST:$SSH_PORT..."

# Create directories on remote server
ssh -p $SSH_PORT -i $SSH_KEY $SSH_USER@$SSH_HOST 'mkdir -p /workspace/src /workspace/configs/accelerate /workspace/configs/experiments'

# Copy files to remote server
scp -P $SSH_PORT -i $SSH_KEY pyproject.toml README.md install-deps.sh $SSH_USER@$SSH_HOST:/workspace/
scp -P $SSH_PORT -i $SSH_KEY src/train.py src/__init__.py $SSH_USER@$SSH_HOST:/workspace/src/
scp -P $SSH_PORT -i $SSH_KEY configs/accelerate/a40_optimized.yaml $SSH_USER@$SSH_HOST:/workspace/configs/accelerate/
scp -P $SSH_PORT -i $SSH_KEY configs/experiments/production.json $SSH_USER@$SSH_HOST:/workspace/configs/experiments/

echo "Deployment complete!"