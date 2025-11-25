# VOKG - Video Object Knowledge Graph

A complete system that extracts objects from videos using SAM 2, tracks them across frames, detects interactions, builds a temporal knowledge graph, and generates AI-powered narratives explaining what happens in the video.

## 🎯 Overview

VOKG transforms raw videos into structured, queryable knowledge graphs with:
- **Object Detection & Tracking**: SAM 2 segments every object and tracks them across frames
- **Interaction Detection**: Spatial and temporal heuristics detect proximity, containment, chasing, following, etc.
- **Knowledge Graph**: Neo4j graph with objects as nodes and interactions/temporal/causal relationships as edges
- **LLM Reasoning**: GPT-4 or Gemini generates natural language narratives explaining the video
- **Real-time Updates**: WebSocket notifications for processing progress
- **REST API**: Full-featured API for video upload, graph queries, and narrative retrieval

## 🏗️ Architecture

```
┌─────────────┐
│   Frontend  │ (React + TypeScript)
└──────┬──────┘
       │ HTTP/WebSocket
┌──────▼──────────────────────────┐
│      API Gateway (FastAPI)       │
│  - Auth (JWT)                    │
│  - Rate Limiting                 │
│  - WebSocket Progress            │
└──────┬──────────────────────────┘
       │
┌──────▼─────────────────────────────────┐
│         Processing Pipeline            │
│  1. Video Ingestion                    │
│  2. Frame Extraction (FFmpeg)          │
│  3. SAM 2 Inference (GPU)              │
│  4. Interaction Detection (CPU)        │
│  5. Knowledge Graph Generation         │
│  6. LLM Reasoning (OpenAI/Gemini)      │
└──────┬─────────────────────────────────┘
       │
┌──────▼──────────┬──────────┬───────────┐
│   PostgreSQL    │  Neo4j   │   Redis   │
│   (Metadata)    │  (Graph) │  (Cache)  │
└─────────────────┴──────────┴───────────┘
       │
┌──────▼──────────┐
│  S3/MinIO       │
│  (Videos,       │
│   Frames,       │
│   Masks)        │
└─────────────────┘
```

## 📋 Prerequisites

- **Docker & Docker Compose**
- **NVIDIA GPU** (for SAM 2 inference)
- **NVIDIA Container Toolkit**
- **OpenAI API Key** or **Google Gemini API Key**

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/vokg.git
cd vokg
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set:
- `SECRET_KEY` (generate with `openssl rand -hex 32`)
- `ENCRYPTION_KEY` (generate with `openssl rand -hex 32`)
- `POSTGRES_PASSWORD`
- `NEO4J_PASSWORD`
- `OPENAI_API_KEY` (or `GEMINI_API_KEY`)

### 3. Download SAM 2 Model

```bash
mkdir -p models
cd models

# Download SAM 2 checkpoint (example for large model)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2_hiera_large.pt

# Download config
wget https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2_configs/sam2_hiera_l.yaml
```

### 4. Start Services

```bash
docker-compose up -d
```

This starts:
- PostgreSQL (port 5432)
- Neo4j (ports 7474, 7687)
- Redis (port 6379)
- MinIO (ports 9000, 9001)
- API Gateway (port 8000)
- Celery Workers (CPU, GPU, LLM, Graph)
- Flower (port 5555) - Celery monitoring

### 5. Run Database Migrations

```bash
docker-compose exec api alembic upgrade head
```

### 6. Create First User

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "username": "testuser",
    "password": "SecurePass123"
  }'
```

### 7. Upload Video

```bash
# Login to get token
TOKEN=$(curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123"
  }' | jq -r '.access_token')

# Upload video
curl -X POST http://localhost:8000/api/v1/videos/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@video.mp4" \
  -F "title=My Test Video" \
  -F "description=Testing VOKG"
```

### 8. Monitor Progress

```bash
# WebSocket connection (use wscat or similar)
wscat -c "ws://localhost:8000/ws/{video_id}?token=$TOKEN"

# Or check Celery Flower
open http://localhost:5555
```

### 9. Query Knowledge Graph

```bash
# Get full graph
curl http://localhost:8000/api/v1/graphs/{video_id} \
  -H "Authorization: Bearer $TOKEN"

# Get narrative
curl http://localhost:8000/api/v1/graphs/{video_id}/narrative \
  -H "Authorization: Bearer $TOKEN"

# Natural language query
curl -X POST http://localhost:8000/api/v1/queries \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What objects are moving?",
    "video_id": 1
  }'
```

## 📁 Project Structure

```
vokg-backend/
├── backend/
│   ├── api_gateway/          # FastAPI application
│   │   ├── main.py           # Entry point
│   │   ├── auth.py           # JWT authentication
│   │   ├── dependencies.py   # FastAPI dependencies
│   │   ├── middleware.py     # CORS, logging, errors
│   │   ├── websocket.py      # WebSocket manager
│   │   └── routes/           # API endpoints
│   │       ├── videos.py     # Video management
│   │       ├── graphs.py     # Knowledge graph queries
│   │       ├── queries.py    # Natural language queries
│   │       └── health.py     # Health checks
│   ├── services/             # Celery workers
│   │   ├── video_ingestion/  # Video validation & queueing
│   │   ├── frame_extraction/ # FFmpeg frame extraction
│   │   ├── sam_inference/    # SAM 2 GPU inference
│   │   ├── interaction_detection/ # Interaction heuristics
│   │   ├── graph_generator/  # Neo4j graph building
│   │   └── llm_reasoning/    # OpenAI/Gemini narrative generation
│   ├── database/
│   │   ├── models.py         # SQLAlchemy models
│   │   ├── postgres.py       # PostgreSQL connection
│   │   ├── neo4j_client.py   # Neo4j client
│   │   └── redis_client.py   # Redis client
│   ├── core/
│   │   ├── config.py         # Configuration management
│   │   ├── logging.py        # Structured logging
│   │   ├── celery_app.py     # Celery configuration
│   │   └── storage.py        # S3/MinIO client
│   └── utils/
│       ├── video_utils.py    # Video processing utilities
│       ├── encryption.py     # API key encryption
│       └── validators.py     # Input validation
├── docker/
│   ├── api_gateway.Dockerfile
│   ├── worker.Dockerfile
│   └── gpu_worker.Dockerfile
├── alembic/                  # Database migrations
├── docker-compose.yml
├── requirements.txt
├── requirements-gpu.txt
└── .env.example
```

## 🔧 Configuration

All configuration is done via environment variables in `.env`:

### Critical Settings

- `SECRET_KEY`: JWT signing key (min 32 chars)
- `ENCRYPTION_KEY`: API key encryption (min 32 chars)
- `OPENAI_API_KEY` or `GEMINI_API_KEY`: LLM API key
- `POSTGRES_PASSWORD`: Database password
- `NEO4J_PASSWORD`: Graph database password

### Video Processing

- `FRAME_EXTRACTION_FPS`: Frames per second to extract (default: 1)
- `FRAME_EXTRACTION_MAX_FRAMES`: Maximum frames (default: 1000)
- `SAM_POINTS_PER_SIDE`: SAM grid density (default: 32)

### Interaction Detection

- `INTERACTION_PROXIMITY_THRESHOLD`: Distance for proximity (pixels)
- `INTERACTION_TEMPORAL_WINDOW`: Frames for temporal analysis
- `TRACKING_IOU_THRESHOLD`: IoU threshold for tracking

### LLM

- `LLM_PROVIDER`: `openai` or `gemini`
- `OPENAI_MODEL`: Model name (e.g., `gpt-4o`)
- `LLM_MAX_RETRIES`: Number of critique/revision passes

## 🔌 API Endpoints

### Authentication

- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login (returns JWT)
- `GET /api/v1/auth/me` - Get current user info

### Videos

- `POST /api/v1/videos/upload` - Upload video
- `GET /api/v1/videos` - List videos
- `GET /api/v1/videos/{id}` - Get video details
- `PATCH /api/v1/videos/{id}` - Update video metadata
- `DELETE /api/v1/videos/{id}` - Delete video
- `POST /api/v1/videos/{id}/reprocess` - Restart processing

### Knowledge Graphs

- `GET /api/v1/graphs/{video_id}` - Get full graph
- `GET /api/v1/graphs/{video_id}/objects` - Get objects
- `GET /api/v1/graphs/{video_id}/interactions` - Get interactions
- `GET /api/v1/graphs/{video_id}/narrative` - Get LLM narrative
- `GET /api/v1/graphs/{video_id}/metrics` - Get graph metrics
- `GET /api/v1/graphs/{video_id}/download` - Download graph (JSON/GEXF/GraphML)

### Queries

- `POST /api/v1/queries` - Natural language query
- `GET /api/v1/queries/{video_id}/suggestions` - Query suggestions

### Health

- `GET /health` - Basic health check
- `GET /health/detailed` - Check all services
- `GET /health/ready` - Kubernetes readiness
- `GET /health/live` - Kubernetes liveness

### WebSocket

- `WS /ws/{video_id}?token={jwt}` - Real-time progress updates

## 🧪 Development

### Run Tests

```bash
docker-compose exec api pytest
```

### Access Database

```bash
# PostgreSQL
docker-compose exec postgres psql -U vokg -d vokg

# Neo4j Browser
open http://localhost:7474

# MinIO Console
open http://localhost:9001
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f worker-gpu
```

### Generate Database Migration

```bash
docker-compose exec api alembic revision --autogenerate -m "Description"
docker-compose exec api alembic upgrade head
```

## 🐛 Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If fails, install nvidia-container-toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### SAM 2 Model Not Found

Ensure models are in `./models/` directory and paths in `.env` are correct.

### Out of Memory

Reduce `SAM_BATCH_SIZE` or `SAM_POINTS_PER_SIDE` in `.env`.

### LLM API Errors

Check API key is valid and rate limits not exceeded.

## 📊 Monitoring

- **Celery Flower**: http://localhost:5555 - Task monitoring
- **Neo4j Browser**: http://localhost:7474 - Graph visualization
- **MinIO Console**: http://localhost:9001 - Storage management

## 🔐 Security Notes

- **NEVER** commit `.env` file
- Rotate `SECRET_KEY` and `ENCRYPTION_KEY` regularly
- Use strong passwords for databases
- Enable HTTPS in production
- Restrict CORS origins to your frontend domain
- Store API keys encrypted in database (already implemented)

## 📈 Performance Tips

1. **GPU Memory**: Adjust `SAM_BATCH_SIZE` based on GPU VRAM
2. **Frame Extraction**: Lower `FRAME_EXTRACTION_FPS` for faster processing
3. **Worker Concurrency**: Tune Celery worker counts in `docker-compose.yml`
4. **Database Indexes**: Already optimized in models
5. **Caching**: Redis caching enabled for graph queries

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📝 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- [SAM 2](https://github.com/facebookresearch/sam2) - Meta AI
- [FastAPI](https://fastapi.tiangolo.com/)
- [Neo4j](https://neo4j.com/)
- [Celery](https://docs.celeryq.dev/)

---

**Built with ❤️ for video understanding**
#   V O K G  
 