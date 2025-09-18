FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 AS base

RUN apt-get update && apt-get install -y \
    git \
    wget \
    tmux \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir hatch

WORKDIR /workspace

# Copy only dependency files first for better caching
COPY pyproject.toml .
# Create a minimal src/__init__.py for package installation
RUN mkdir -p src && echo '__version__ = "0.1.0"' > src/__init__.py

# Install dependencies - this layer will be cached unless pyproject.toml changes
RUN hatch env create

# Now copy the actual source code - changes here won't invalidate dependency cache
COPY src/ ./src/
COPY configs/ ./configs/

ENV HATCH_ENV=cuda

ENTRYPOINT ["hatch", "run", "cuda:train-prod"]