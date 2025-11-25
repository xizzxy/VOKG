FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    ffmpeg \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend ./backend
COPY alembic ./alembic
COPY alembic.ini .

# Create temp directories
RUN mkdir -p /tmp/vokg/uploads

# Expose port
EXPOSE 8000

# Run API gateway
CMD ["uvicorn", "backend.api_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
