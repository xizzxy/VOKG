# CPU Worker Dockerfile
# For frame extraction and general processing tasks

FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend ./backend
COPY alembic ./alembic
COPY alembic.ini .

# Create non-root user
RUN useradd -m -u 1000 vokg && chown -R vokg:vokg /app
USER vokg

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD celery -A backend.core.celery_app inspect ping -d celery@$HOSTNAME || exit 1

# Run CPU worker (frame extraction, interaction detection, graph generation)
CMD ["celery", "-A", "backend.core.celery_app", "worker", \
     "--queues=cpu,graph,default", \
     "--concurrency=4", \
     "--loglevel=info", \
     "--max-tasks-per-child=100", \
     "--task-events"]
