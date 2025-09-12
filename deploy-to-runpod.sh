#!/bin/bash

# Deploy to RunPod using runpodctl CLI
# Builds, pushes to GHCR, and deploys pods

set -e

# Configuration
IMAGE_NAME="${IMAGE_NAME:-asr-training}"
TAG="${TAG:-latest}"
RUNPOD_USER="${RUNPOD_USER:-$USER}"
POD_NAME="${POD_NAME:-asr-training}"
GPU_TYPE="${GPU_TYPE:-NVIDIA GeForce RTX 4090}"
GPU_COUNT="${GPU_COUNT:-1}"
VOLUME_SIZE="${VOLUME_SIZE:-100}"
CONTAINER_DISK="${CONTAINER_DISK:-50}"
CLOUD_TYPE="${CLOUD_TYPE:-SECURE}"  # SECURE or COMMUNITY
BID_PRICE="${BID_PRICE:-0.50}"
USE_SPOT="${USE_SPOT:-false}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_color() {
    echo -e "${!1}${2}${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_color "YELLOW" "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_color "RED" "Error: Docker is not installed"
        exit 1
    fi
    
    # Check runpodctl CLI
    if ! command -v runpodctl &> /dev/null; then
        print_color "RED" "Error: runpodctl is not installed"
        print_color "YELLOW" "Install from: https://github.com/runpod/runpodctl"
        exit 1
    fi
     
    # Check RunPod API key
    if [ -z "$RUNPOD_API_KEY" ]; then
        print_color "YELLOW" "RunPod API key not set"
        read -sp "Enter RunPod API key: " RUNPOD_API_KEY
        echo
        export RUNPOD_API_KEY
    fi
    
    # Configure runpodctl
    runpodctl config --apiKey "$RUNPOD_API_KEY" > /dev/null 2>&1 || true
    
    print_color "GREEN" "✓ Prerequisites checked"
}

# Build Docker image
build_image() {
    print_color "YELLOW" "Building Docker image..."
    
    if [ ! -f "Dockerfile" ]; then
        print_color "RED" "Error: Dockerfile not found"
        exit 1
    fi
    
    docker build -t ${IMAGE_NAME}:${TAG} .
    
    if [ $? -ne 0 ]; then
        print_color "RED" "Error: Failed to build Docker image"
        exit 1
    fi
    
    print_color "GREEN" "✓ Docker image built"
}

# Push to RunPod Registry
push_to_runpod() {
    local full_image="registry.runpod.io/${RUNPOD_USER}/${IMAGE_NAME}:${TAG}"
    
    print_color "YELLOW" "Pushing to RunPod Registry..."
    
    # Login to RunPod registry using API key
    echo "$RUNPOD_API_KEY" | docker login registry.runpod.io -u "$RUNPOD_USER" --password-stdin
    
    if [ $? -ne 0 ]; then
        print_color "RED" "Error: Failed to login to RunPod registry"
        print_color "YELLOW" "Make sure your RUNPOD_API_KEY is valid"
        exit 1
    fi
    
    # Tag and push
    docker tag ${IMAGE_NAME}:${TAG} ${full_image}
    docker push ${full_image}
    
    if [ $? -ne 0 ]; then
        print_color "RED" "Error: Failed to push to RunPod registry"
        exit 1
    fi
    
    print_color "GREEN" "✓ Pushed to RunPod Registry: ${full_image}"
    echo "$full_image"
}

# Create pod with runpodctl
create_pod() {
    local image=$1
    
    print_color "YELLOW" "Creating RunPod pod..."
    
    # No additional auth needed for RunPod registry - the pod will use the API key
    
    # Build runpodctl command
    local cmd="runpodctl create pod"
    cmd="$cmd --name \"$POD_NAME\""
    cmd="$cmd --imageName \"$image\""
    cmd="$cmd --gpuType \"$GPU_TYPE\""
    cmd="$cmd --gpuCount $GPU_COUNT"
    cmd="$cmd --volumeSize $VOLUME_SIZE"
    cmd="$cmd --containerDiskSize $CONTAINER_DISK"
    cmd="$cmd --ports \"22/tcp,6006/http\""
    cmd="$cmd --env \"HF_TOKEN=$HF_TOKEN\""
    
    # RunPod registry images are automatically accessible with the API key
    # No additional credentials needed
    
    # Add spot instance options
    if [ "$USE_SPOT" = "true" ]; then
        cmd="$cmd --spot"
        cmd="$cmd --bidPrice $BID_PRICE"
    fi
    
    # Add cloud type
    cmd="$cmd --cloud $CLOUD_TYPE"
    
    # Execute command
    print_color "BLUE" "Running: $cmd"
    POD_ID=$(eval $cmd | grep -oP 'Pod created successfully with ID: \K[a-z0-9]+' || true)
    
    if [ -z "$POD_ID" ]; then
        # Try alternative parsing
        POD_ID=$(eval $cmd | grep -oP '"id":\s*"\K[^"]+' || true)
    fi
    
    if [ -z "$POD_ID" ]; then
        print_color "YELLOW" "Pod creation initiated. Check RunPod dashboard for status."
    else
        print_color "GREEN" "✓ Pod created with ID: $POD_ID"
        
        # Get pod details
        print_color "YELLOW" "Getting pod details..."
        sleep 5
        runpodctl get pod $POD_ID || true
    fi
}

# List pods
list_pods() {
    print_color "BLUE" "Current RunPod pods:"
    runpodctl get pods
}

# Stop pod
stop_pod() {
    local pod_id=$1
    print_color "YELLOW" "Stopping pod $pod_id..."
    runpodctl stop pod $pod_id
    print_color "GREEN" "✓ Pod stopped"
}

# Main execution
main() {
    print_color "GREEN" "=== RunPod Deployment Script ==="
    echo ""
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build-only)
                BUILD_ONLY=true
                shift
                ;;
            --push-only)
                PUSH_ONLY=true
                shift
                ;;
            --deploy-only)
                DEPLOY_ONLY=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-push)
                SKIP_PUSH=true
                shift
                ;;
            --use-spot)
                USE_SPOT=true
                shift
                ;;
            --list)
                check_prerequisites
                list_pods
                exit 0
                ;;
            --stop)
                check_prerequisites
                stop_pod $2
                exit 0
                ;;
            --gpu-type)
                GPU_TYPE="$2"
                shift 2
                ;;
            --gpu-count)
                GPU_COUNT="$2"
                shift 2
                ;;
            --bid-price)
                BID_PRICE="$2"
                shift 2
                ;;
            --help)
                cat << EOF
Usage: $0 [OPTIONS]

Options:
    --build-only        Only build Docker image
    --push-only         Only push to RunPod registry
    --deploy-only       Only deploy to RunPod
    --skip-build        Skip Docker build
    --skip-push         Skip GHCR push
    --use-spot          Use spot instances
    --gpu-type TYPE     GPU type (default: NVIDIA GeForce RTX 4090)
    --gpu-count N       Number of GPUs (default: 1)
    --bid-price PRICE   Spot instance bid price (default: 0.50)
    --list              List all pods
    --stop POD_ID       Stop a specific pod
    --help              Show this help message

Environment Variables:
    RUNPOD_API_KEY      RunPod API key (required)
    RUNPOD_USER         RunPod username (default: current user)
    IMAGE_NAME          Docker image name (default: asr-training)
    TAG                 Docker tag (default: latest)
    POD_NAME            Pod name (default: asr-training)
    HF_TOKEN            HuggingFace token (optional)

Examples:
    # Full deployment
    $0

    # Deploy with spot instance
    $0 --use-spot --bid-price 0.60

    # Deploy with multiple GPUs
    $0 --gpu-type "NVIDIA A100 80GB" --gpu-count 2

    # List and stop pods
    $0 --list
    $0 --stop abc123
EOF
                exit 0
                ;;
            *)
                print_color "RED" "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Check prerequisites
    check_prerequisites
    
    # Determine workflow
    if [ "$BUILD_ONLY" = "true" ]; then
        build_image
        exit 0
    fi
    
    if [ "$PUSH_ONLY" = "true" ]; then
        if [ "$SKIP_BUILD" != "true" ]; then
            build_image
        fi
        push_to_runpod
        exit 0
    fi
    
    if [ "$DEPLOY_ONLY" = "true" ]; then
        FULL_IMAGE="registry.runpod.io/${RUNPOD_USER}/${IMAGE_NAME}:${TAG}"
        create_pod "$FULL_IMAGE"
        exit 0
    fi
    
    # Default: full pipeline
    if [ "$SKIP_BUILD" != "true" ]; then
        build_image
    fi
    
    if [ "$SKIP_PUSH" != "true" ]; then
        FULL_IMAGE=$(push_to_runpod)
    else
        FULL_IMAGE="registry.runpod.io/${RUNPOD_USER}/${IMAGE_NAME}:${TAG}"
    fi
    
    create_pod "$FULL_IMAGE"
    
    echo ""
    print_color "GREEN" "=== Deployment Complete ==="
    print_color "BLUE" "Image: $FULL_IMAGE"
    print_color "YELLOW" "Monitor your pod at: https://www.runpod.io/console/pods"
}

# Run main function
main "$@"