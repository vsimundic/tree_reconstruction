FROM python:3.10-slim-bookworm

LABEL maintainer="PEAR Project"
LABEL description="Tree reconstruction and pruning point detection from 3D point clouds of fruit trees"

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libusb-1.0-0 \
    libx11-6 \
    libxt6 \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    open3d \
    opencv-python \
    scipy \
    scikit-learn \
    matplotlib \
    trimesh \
    pyyaml \
    tqdm \
    pandas \
    plyfile

# Set working directory
WORKDIR /workspace

# Default command
# CMD ["/bin/bash"]
