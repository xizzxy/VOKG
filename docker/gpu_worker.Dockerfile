FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    gcc \
    g++ \
    ffmpeg \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Copy requirements
COPY requirements.txt requirements-gpu.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-gpu.txt

# Install SAM 2
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/sam2.git

# Copy application code
COPY backend ./backend

# Create temp directories
RUN mkdir -p /tmp/vokg /models

# Set CUDA environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Run Celery worker
CMD ["celery", "-A", "backend.core.celery_app", "worker", "--loglevel=info", "--queues=gpu", "--concurrency=1"]
