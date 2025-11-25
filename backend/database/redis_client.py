"""
Redis client for caching and real-time data
"""

import json
from typing import Optional, Any
import redis
from redis.connection import ConnectionPool

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


class RedisClient:
    """
    Redis client wrapper with common operations
    """

    def __init__(self):
        """
        Initialize Redis connection pool
        """
        self.pool = ConnectionPool.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            max_connections=50,
        )
        self.client = redis.Redis(connection_pool=self.pool)
        self._test_connection()

    def _test_connection(self):
        """
        Test Redis connection
        """
        try:
            self.client.ping()
            logger.info("Redis connection established")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def get(self, key: str) -> Optional[str]:
        """
        Get value by key

        Args:
            key: Redis key

        Returns:
            Value or None
        """
        try:
            return self.client.get(key)
        except redis.RedisError as e:
            logger.error(f"Redis GET error: {e}", key=key)
            return None

    def set(
        self, key: str, value: str, expire: Optional[int] = None
    ) -> bool:
        """
        Set key-value pair

        Args:
            key: Redis key
            value: Value to store
            expire: Optional expiration in seconds

        Returns:
            True if successful
        """
        try:
            return self.client.set(key, value, ex=expire)
        except redis.RedisError as e:
            logger.error(f"Redis SET error: {e}", key=key)
            return False

    def delete(self, key: str) -> bool:
        """
        Delete key

        Args:
            key: Redis key

        Returns:
            True if deleted
        """
        try:
            return self.client.delete(key) > 0
        except redis.RedisError as e:
            logger.error(f"Redis DELETE error: {e}", key=key)
            return False

    def exists(self, key: str) -> bool:
        """
        Check if key exists

        Args:
            key: Redis key

        Returns:
            True if exists
        """
        try:
            return self.client.exists(key) > 0
        except redis.RedisError as e:
            logger.error(f"Redis EXISTS error: {e}", key=key)
            return False

    def get_json(self, key: str) -> Optional[Any]:
        """
        Get JSON value

        Args:
            key: Redis key

        Returns:
            Parsed JSON or None
        """
        value = self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}", key=key)
        return None

    def set_json(
        self, key: str, value: Any, expire: Optional[int] = None
    ) -> bool:
        """
        Set JSON value

        Args:
            key: Redis key
            value: JSON-serializable value
            expire: Optional expiration in seconds

        Returns:
            True if successful
        """
        try:
            json_str = json.dumps(value)
            return self.set(key, json_str, expire)
        except (json.JSONEncodeError, TypeError) as e:
            logger.error(f"JSON encode error: {e}", key=key)
            return False

    def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment key value

        Args:
            key: Redis key
            amount: Increment amount

        Returns:
            New value or None
        """
        try:
            return self.client.incr(key, amount)
        except redis.RedisError as e:
            logger.error(f"Redis INCR error: {e}", key=key)
            return None

    def expire(self, key: str, seconds: int) -> bool:
        """
        Set expiration on key

        Args:
            key: Redis key
            seconds: Expiration in seconds

        Returns:
            True if successful
        """
        try:
            return self.client.expire(key, seconds)
        except redis.RedisError as e:
            logger.error(f"Redis EXPIRE error: {e}", key=key)
            return False

    def ttl(self, key: str) -> Optional[int]:
        """
        Get time to live for key

        Args:
            key: Redis key

        Returns:
            TTL in seconds or None
        """
        try:
            return self.client.ttl(key)
        except redis.RedisError as e:
            logger.error(f"Redis TTL error: {e}", key=key)
            return None

    # List operations
    def lpush(self, key: str, *values: str) -> Optional[int]:
        """
        Push values to list head

        Args:
            key: Redis key
            *values: Values to push

        Returns:
            List length or None
        """
        try:
            return self.client.lpush(key, *values)
        except redis.RedisError as e:
            logger.error(f"Redis LPUSH error: {e}", key=key)
            return None

    def rpush(self, key: str, *values: str) -> Optional[int]:
        """
        Push values to list tail

        Args:
            key: Redis key
            *values: Values to push

        Returns:
            List length or None
        """
        try:
            return self.client.rpush(key, *values)
        except redis.RedisError as e:
            logger.error(f"Redis RPUSH error: {e}", key=key)
            return None

    def lrange(self, key: str, start: int = 0, end: int = -1) -> list[str]:
        """
        Get list range

        Args:
            key: Redis key
            start: Start index
            end: End index

        Returns:
            List of values
        """
        try:
            return self.client.lrange(key, start, end)
        except redis.RedisError as e:
            logger.error(f"Redis LRANGE error: {e}", key=key)
            return []

    # Hash operations
    def hset(self, name: str, key: str, value: str) -> bool:
        """
        Set hash field

        Args:
            name: Hash name
            key: Field key
            value: Field value

        Returns:
            True if new field
        """
        try:
            return self.client.hset(name, key, value) > 0
        except redis.RedisError as e:
            logger.error(f"Redis HSET error: {e}", name=name, key=key)
            return False

    def hget(self, name: str, key: str) -> Optional[str]:
        """
        Get hash field

        Args:
            name: Hash name
            key: Field key

        Returns:
            Field value or None
        """
        try:
            return self.client.hget(name, key)
        except redis.RedisError as e:
            logger.error(f"Redis HGET error: {e}", name=name, key=key)
            return None

    def hgetall(self, name: str) -> dict:
        """
        Get all hash fields

        Args:
            name: Hash name

        Returns:
            Dict of fields
        """
        try:
            return self.client.hgetall(name)
        except redis.RedisError as e:
            logger.error(f"Redis HGETALL error: {e}", name=name)
            return {}

    # Pub/Sub for WebSocket notifications
    def publish(self, channel: str, message: str) -> int:
        """
        Publish message to channel

        Args:
            channel: Channel name
            message: Message to publish

        Returns:
            Number of subscribers
        """
        try:
            return self.client.publish(channel, message)
        except redis.RedisError as e:
            logger.error(f"Redis PUBLISH error: {e}", channel=channel)
            return 0

    def publish_json(self, channel: str, data: Any) -> int:
        """
        Publish JSON message to channel

        Args:
            channel: Channel name
            data: JSON-serializable data

        Returns:
            Number of subscribers
        """
        try:
            message = json.dumps(data)
            return self.publish(channel, message)
        except (json.JSONEncodeError, TypeError) as e:
            logger.error(f"JSON encode error: {e}", channel=channel)
            return 0

    def close(self):
        """
        Close Redis connection
        """
        try:
            self.client.close()
            logger.info("Redis connection closed")
        except redis.RedisError as e:
            logger.error(f"Error closing Redis connection: {e}")


# Singleton instance
_redis_client: Optional[RedisClient] = None


def get_redis_client() -> RedisClient:
    """
    Get singleton Redis client instance
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client
