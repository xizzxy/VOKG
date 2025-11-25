#!/bin/bash
# VOKG Setup Script

set -e

echo "=========================================="
echo "  VOKG - Video Object Knowledge Graph"
echo "  Setup Script"
echo "=========================================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo " Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo " Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

echo " Docker found: $(docker --version)"
echo " Docker Compose found: $(docker-compose --version)"
echo ""

# Check NVIDIA GPU
echo "Checking NVIDIA GPU..."
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo " NVIDIA GPU detected"
    nvidia-smi | grep "CUDA Version"
else
    echo "  NVIDIA GPU not detected or nvidia-container-toolkit not installed"
    echo "   GPU workers will not function. See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
fi
echo ""

# Create .env if not exists
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env

    # Generate secure keys
    SECRET_KEY=$(openssl rand -hex 32)
    ENCRYPTION_KEY=$(openssl rand -hex 32)

    # Update .env with generated keys
    sed -i "s/your-secret-key-min-32-chars-change-this-in-production/$SECRET_KEY/" .env
    sed -i "s/your-encryption-key-min-32-chars-change-this/$ENCRYPTION_KEY/" .env

    echo " .env file created with secure keys"
    echo ""
    echo "  IMPORTANT: Edit .env and set:"
    echo "   - OPENAI_API_KEY or GEMINI_API_KEY"
    echo "   - Database passwords"
    echo ""
    read -p "Press Enter when you've updated .env file..."
else
    echo " .env file already exists"
fi
echo ""

# Download SAM 2 models
if [ ! -f models/sam2_hiera_large.pt ]; then
    echo "Downloading SAM 2 models..."
    bash scripts/download_sam2_models.sh
else
    echo " SAM 2 models already downloaded"
fi
echo ""

# Build Docker images
echo "Building Docker images..."
docker-compose build
echo " Docker images built"
echo ""

# Start services
echo "Starting services..."
docker-compose up -d postgres redis neo4j minio
echo " Waiting for databases to be ready..."
sleep 10
echo ""

# Run migrations
echo "Running database migrations..."
docker-compose run --rm api alembic upgrade head
echo " Migrations complete"
echo ""

# Start all services
echo "Starting all services..."
docker-compose up -d
echo ""

# Show status
echo "=========================================="
echo "  VOKG Setup Complete!"
echo "=========================================="
echo ""
echo "Services running:"
docker-compose ps
echo ""
echo "Access points:"
echo "  - API:          http://localhost:8000"
echo "  - API Docs:     http://localhost:8000/docs"
echo "  - Neo4j:        http://localhost:7474"
echo "  - MinIO:        http://localhost:9001"
echo "  - Flower:       http://localhost:5555"
echo ""
echo "Next steps:"
echo "  1. Create a user:"
echo "     curl -X POST http://localhost:8000/api/v1/auth/register \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"email\":\"user@example.com\",\"username\":\"testuser\",\"password\":\"SecurePass123\"}'"
echo ""
echo "  2. Upload a video (after login)"
echo "  3. Monitor processing at http://localhost:5555"
echo ""
echo "View logs:"
echo "  docker-compose logs -f"
echo ""
echo "Stop services:"
echo "  docker-compose down"
echo ""
