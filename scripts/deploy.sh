#!/bin/bash
# Docker deployment script

set -e

# Configuration
ENVIRONMENT=${1:-production}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo -e "${GREEN}🚀 Deploying with Docker${NC}"
echo -e "${YELLOW}Environment: $ENVIRONMENT${NC}"

# Build Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -f Dockerfile -t asr-training:$ENVIRONMENT .

# Tag for registry if DOCKER_REGISTRY is set
if [ ! -z "$DOCKER_REGISTRY" ]; then
    echo -e "${YELLOW}Tagging image for registry...${NC}"
    docker tag asr-training:$ENVIRONMENT $DOCKER_REGISTRY/asr-training:$ENVIRONMENT

    # Push to registry
    echo -e "${YELLOW}Pushing to registry...${NC}"
    docker push $DOCKER_REGISTRY/asr-training:$ENVIRONMENT
fi

echo -e "${GREEN}✅ Docker deployment complete!${NC}"

# Optional: Run locally
if [ "$2" == "--run" ]; then
    echo -e "${YELLOW}Running Docker container locally...${NC}"
    docker run --gpus all \
        -v $(pwd)/outputs:/workspace/outputs \
        -v $(pwd)/datasets_cache:/workspace/datasets_cache \
        asr-training:$ENVIRONMENT \
        +experiments=$ENVIRONMENT
fi