# GPU Worker Dockerfile
# For SAM 2 inference requiring CUDA support

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libpq5 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements-gpu.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir -r requirements-gpu.txt

# Copy application code
COPY backend ./backend

# Create model directory
RUN mkdir -p /models

# Create non-root user
RUN useradd -m -u 1000 vokg && chown -R vokg:vokg /app /models
USER vokg

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD celery -A backend.core.celery_app inspect ping -d celery@$HOSTNAME || exit 1

# Run GPU worker (SAM 2 inference only)
CMD ["celery", "-A", "backend.core.celery_app", "worker", \
     "--queues=gpu", \
     "--concurrency=1", \
     "--loglevel=info", \
     "--max-tasks-per-child=10", \
     "--task-events", \
     "--pool=solo"]
