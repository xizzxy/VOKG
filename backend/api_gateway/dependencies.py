"""
FastAPI dependencies for request handling
"""

from typing import Generator, Optional
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from backend.database.postgres import get_db
from backend.database.models import User
from backend.database.redis_client import get_redis_client, RedisClient
from backend.core.logging import get_logger
from .auth import get_user_from_token

logger = get_logger(__name__)

# Security scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """
    Get current authenticated user from JWT token

    Args:
        credentials: HTTP Bearer credentials
        db: Database session

    Returns:
        Current user

    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    user = get_user_from_token(db, token)
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get current active user

    Args:
        current_user: Current user from token

    Returns:
        Active user

    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )
    return current_user


async def get_current_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get current superuser

    Args:
        current_user: Current user from token

    Returns:
        Superuser

    Raises:
        HTTPException: If user is not a superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough privileges",
        )
    return current_user


def get_redis() -> RedisClient:
    """
    Get Redis client dependency
    """
    return get_redis_client()


async def verify_rate_limit(
    user: User = Depends(get_current_active_user),
    redis: RedisClient = Depends(get_redis),
) -> bool:
    """
    Verify user rate limit

    Args:
        user: Current user
        redis: Redis client

    Returns:
        True if within rate limit

    Raises:
        HTTPException: If rate limit exceeded
    """
    from backend.core.config import settings
    import time

    # Rate limit keys
    minute_key = f"rate_limit:user:{user.id}:minute"
    hour_key = f"rate_limit:user:{user.id}:hour"

    # Get current counts
    minute_count = redis.get(minute_key)
    hour_count = redis.get(hour_key)

    # Check limits
    if minute_count and int(minute_count) >= settings.RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded (per minute)",
        )

    if hour_count and int(hour_count) >= settings.RATE_LIMIT_PER_HOUR:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded (per hour)",
        )

    # Increment counters
    if not minute_count:
        redis.set(minute_key, "1", expire=60)
    else:
        redis.incr(minute_key)

    if not hour_count:
        redis.set(hour_key, "1", expire=3600)
    else:
        redis.incr(hour_key)

    return True
