#!/bin/bash
# Modern deployment script with multiple strategies

set -e

# Configuration
DEPLOYMENT_METHOD=${1:-rsync}  # rsync, git, docker, or runpod
ENVIRONMENT=${2:-production}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo -e "${RED}Error: .env file not found${NC}"
    exit 1
fi

echo -e "${GREEN}🚀 Deploying with method: $DEPLOYMENT_METHOD${NC}"

case $DEPLOYMENT_METHOD in
    rsync)
        echo -e "${YELLOW}Using rsync deployment (fastest for updates)${NC}"

        # Create remote directories
        ssh -p $SSH_PORT -i $SSH_KEY $SSH_USER@$SSH_HOST \
            'mkdir -p /workspace/{src,configs/{hydra/{model,data,training,experiments},accelerate,experiments},scripts,outputs,logs}'

        # Sync files efficiently (only changed files)
        rsync -avz --progress \
            -e "ssh -p $SSH_PORT -i $SSH_KEY" \
            --exclude='*.pyc' \
            --exclude='__pycache__' \
            --exclude='.git' \
            --exclude='outputs/' \
            --exclude='logs/' \
            --exclude='datasets_cache/' \
            --exclude='.mypy_cache/' \
            --exclude='*.egg-info/' \
            src/ configs/ scripts/ pyproject.toml requirements.txt \
            $SSH_USER@$SSH_HOST:/workspace/

        # Install dependencies if needed
        ssh -p $SSH_PORT -i $SSH_KEY $SSH_USER@$SSH_HOST \
            "cd /workspace && pip install -e . --no-deps && pip install -r requirements.txt"
        ;;

    git)
        echo -e "${YELLOW}Using git deployment (version controlled)${NC}"

        # Push to git
        git add -A
        git commit -m "Deploy: $ENVIRONMENT $(date +%Y%m%d_%H%M%S)" || true
        git push origin main

        # Pull on remote
        ssh -p $SSH_PORT -i $SSH_KEY $SSH_USER@$SSH_HOST \
            "cd /workspace && git pull origin main && pip install -e ."
        ;;

    docker)
        echo -e "${YELLOW}Using Docker deployment (reproducible)${NC}"

        # Build and push Docker image
        docker build -f Dockerfile.simple -t $DOCKER_REGISTRY/asr-training:$ENVIRONMENT .
        docker push $DOCKER_REGISTRY/asr-training:$ENVIRONMENT

        # Run on remote
        ssh -p $SSH_PORT -i $SSH_KEY $SSH_USER@$SSH_HOST \
            "docker pull $DOCKER_REGISTRY/asr-training:$ENVIRONMENT && \
             docker run --gpus all -v /workspace/data:/workspace/data \
             $DOCKER_REGISTRY/asr-training:$ENVIRONMENT +experiments=$ENVIRONMENT"
        ;;

    runpod)
        echo -e "${YELLOW}Using RunPod API deployment${NC}"

        # Create deployment package
        tar -czf deployment.tar.gz \
            --exclude='*.pyc' \
            --exclude='__pycache__' \
            --exclude='outputs' \
            --exclude='datasets_cache' \
            src/ configs/ scripts/ pyproject.toml requirements.txt

        # Upload to storage (S3/GCS/etc)
        aws s3 cp deployment.tar.gz s3://$S3_BUCKET/deployments/$(date +%Y%m%d_%H%M%S).tar.gz

        # Trigger RunPod job via API
        python scripts/runpod_deploy.py --environment $ENVIRONMENT
        ;;

    *)
        echo -e "${RED}Unknown deployment method: $DEPLOYMENT_METHOD${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}✅ Deployment complete!${NC}"

# Optional: Start training immediately
if [ "$3" == "--start" ]; then
    echo -e "${YELLOW}Starting training...${NC}"
    ssh -p $SSH_PORT -i $SSH_KEY $SSH_USER@$SSH_HOST \
        "cd /workspace && nohup ./scripts/train_server.sh $ENVIRONMENT > training.log 2>&1 &"
    echo -e "${GREEN}Training started in background${NC}"
fi