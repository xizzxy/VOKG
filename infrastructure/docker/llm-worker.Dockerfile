# LLM Worker Dockerfile
# For LLM reasoning tasks (OpenAI/Gemini API calls)

FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend ./backend

# Create non-root user
RUN useradd -m -u 1000 vokg && chown -R vokg:vokg /app
USER vokg

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD celery -A backend.core.celery_app inspect ping -d celery@$HOSTNAME || exit 1

# Run LLM worker (reasoning tasks only)
CMD ["celery", "-A", "backend.core.celery_app", "worker", \
     "--queues=llm", \
     "--concurrency=2", \
     "--loglevel=info", \
     "--max-tasks-per-child=50", \
     "--task-events"]
