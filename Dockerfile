# Multi-stage build for efficient image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install Hatch for environment management
RUN pip install --no-cache-dir hatch

WORKDIR /workspace

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/
COPY configs/ ./configs/

# Install dependencies using hatch
RUN hatch env create

# Default to CUDA environment
ENV HATCH_ENV=cuda

# Run training with hatch
ENTRYPOINT ["hatch", "run"]
CMD ["train-production"]