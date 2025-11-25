"""
Health check endpoints
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime

from backend.database.postgres import get_db
from backend.database.redis_client import get_redis_client, RedisClient
from backend.database.neo4j_client import get_neo4j_client, Neo4jClient
from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check():
    """
    Basic health check
    """
    return {
        "status": "healthy",
        "service": "vokg-api",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/detailed")
async def detailed_health_check(
    db: Session = Depends(get_db),
):
    """
    Detailed health check with dependency status
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {},
    }

    # Check PostgreSQL
    try:
        db.execute("SELECT 1")
        health_status["services"]["postgresql"] = "healthy"
    except Exception as e:
        logger.error(f"PostgreSQL health check failed: {e}")
        health_status["services"]["postgresql"] = "unhealthy"
        health_status["status"] = "degraded"

    # Check Redis
    try:
        redis = get_redis_client()
        redis.client.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        health_status["services"]["redis"] = "unhealthy"
        health_status["status"] = "degraded"

    # Check Neo4j
    try:
        neo4j = get_neo4j_client()
        neo4j.driver.verify_connectivity()
        health_status["services"]["neo4j"] = "healthy"
    except Exception as e:
        logger.error(f"Neo4j health check failed: {e}")
        health_status["services"]["neo4j"] = "unhealthy"
        health_status["status"] = "degraded"

    # Check Storage (S3/MinIO)
    try:
        from backend.core.storage import get_storage_client

        storage = get_storage_client()
        storage.client.head_bucket(Bucket=storage.bucket_name)
        health_status["services"]["storage"] = "healthy"
    except Exception as e:
        logger.error(f"Storage health check failed: {e}")
        health_status["services"]["storage"] = "unhealthy"
        health_status["status"] = "degraded"

    return health_status


@router.get("/ready")
async def readiness_check(db: Session = Depends(get_db)):
    """
    Kubernetes readiness probe
    """
    try:
        db.execute("SELECT 1")
        return {"status": "ready"}
    except Exception:
        return {"status": "not ready"}, 503


@router.get("/live")
async def liveness_check():
    """
    Kubernetes liveness probe
    """
    return {"status": "alive"}
