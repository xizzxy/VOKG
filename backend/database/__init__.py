"""
Database layer for VOKG
Includes PostgreSQL, Neo4j, and Redis clients
"""

from .postgres import get_db, engine, Base
from .neo4j_client import get_neo4j_client, Neo4jClient
from .redis_client import get_redis_client, RedisClient
from .models import (
    User,
    Video,
    ProcessingJob,
    Frame,
    DetectedObject,
    Interaction,
    GraphSnapshot,
)

__all__ = [
    "get_db",
    "engine",
    "Base",
    "get_neo4j_client",
    "Neo4jClient",
    "get_redis_client",
    "RedisClient",
    "User",
    "Video",
    "ProcessingJob",
    "Frame",
    "DetectedObject",
    "Interaction",
    "GraphSnapshot",
]
