# Makefile for easy deployment and management

.PHONY: help deploy train monitor stop clean

help:
	@echo "ASR Training Deployment Commands:"
	@echo ""
	@echo "  make deploy-local    - Deploy and run locally"
	@echo "  make deploy-runpod   - Deploy to RunPod"
	@echo "  make deploy-vast     - Deploy to Vast.ai (budget option)"
	@echo "  make deploy-docker   - Deploy using Docker"
	@echo "  make deploy-server   - Deploy to your own server"
	@echo ""
	@echo "  make monitor         - Monitor training logs"
	@echo "  make tensorboard     - Launch TensorBoard"
	@echo "  make stop           - Stop current training"
	@echo "  make clean          - Clean outputs and cache"
	@echo ""
	@echo "  make test           - Run tests"
	@echo "  make lint           - Run linters"

# Local deployment
deploy-local:
	@echo "🚀 Deploying locally..."
	python src/train.py +experiments=test training.max_steps=100

# RunPod deployment (premium GPUs)
deploy-runpod:
	@echo "🚀 Deploying to RunPod..."
	python scripts/runpod_deploy.py --environment production

# Vast.ai deployment (budget option)
deploy-vast:
	@echo "🚀 Deploying to Vast.ai..."
	python scripts/deploy_vast.py --environment production --gpu RTX_4090 --max-price 0.8

# Docker deployment
deploy-docker:
	@echo "🚀 Building and running Docker..."
	docker compose up train

# Server deployment (your own GPU server)
deploy-server:
	@echo "🚀 Deploying to server..."
	./scripts/deploy.sh rsync production --start

# Monitor training
monitor:
	@if [ -f .last_pod_id ]; then \
		python scripts/runpod_deploy.py --action status; \
	elif [ -f .last_vast_instance ]; then \
		python scripts/deploy_vast.py --action monitor; \
	else \
		tail -f outputs/*/training.log; \
	fi

# TensorBoard
tensorboard:
	tensorboard --logdir outputs --port 6006

# Stop training
stop:
	@if [ -f .last_pod_id ]; then \
		python scripts/runpod_deploy.py --action stop; \
	elif [ -f .last_vast_instance ]; then \
		python scripts/deploy_vast.py --action stop; \
	else \
		pkill -f "python src/train.py" || true; \
	fi

# Clean outputs
clean:
	rm -rf outputs/* logs/* __pycache__ .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -f .last_pod_id .last_vast_instance

# Development commands
test:
	pytest tests/

lint:
	black src/ scripts/
	flake8 src/ scripts/
	mypy src/

format:
	black src/ scripts/
	isort src/ scripts/

# Install dependencies
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Docker commands
docker-build:
	docker build -f Dockerfile.simple -t asr-training:latest .

docker-push:
	docker tag asr-training:latest ${DOCKER_REGISTRY}/asr-training:latest
	docker push ${DOCKER_REGISTRY}/asr-training:latest

# Quick training commands
train-test:
	python src/train.py +experiments=test

train-mac:
	python src/train.py +experiments=mac_minimal

train-prod:
	accelerate launch src/train.py +experiments=production