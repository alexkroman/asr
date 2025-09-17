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

COPY pyproject.toml .
COPY src/ ./src/
COPY configs/ ./configs/

RUN hatch env create

ENV HATCH_ENV=cuda

ENTRYPOINT ["hatch", "run", "cuda:train-prod"]