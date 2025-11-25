"""
WebSocket manager for real-time progress updates
"""

import json
import asyncio
from typing import Dict, Set
from fastapi import WebSocket, WebSocketDisconnect
from collections import defaultdict

from backend.core.logging import get_logger
from backend.database.redis_client import get_redis_client

logger = get_logger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for real-time updates
    """

    def __init__(self):
        """
        Initialize connection manager
        """
        # Map of video_id -> set of WebSocket connections
        self.active_connections: Dict[int, Set[WebSocket]] = defaultdict(set)
        # Map of WebSocket -> user_id
        self.connection_users: Dict[WebSocket, int] = {}

    async def connect(self, websocket: WebSocket, video_id: int, user_id: int):
        """
        Accept WebSocket connection and subscribe to video updates

        Args:
            websocket: WebSocket connection
            video_id: Video ID to subscribe to
            user_id: User ID
        """
        await websocket.accept()
        self.active_connections[video_id].add(websocket)
        self.connection_users[websocket] = user_id
        logger.info(
            f"WebSocket connected",
            video_id=video_id,
            user_id=user_id,
            total_connections=len(self.active_connections[video_id]),
        )

    def disconnect(self, websocket: WebSocket, video_id: int):
        """
        Remove WebSocket connection

        Args:
            websocket: WebSocket connection
            video_id: Video ID
        """
        self.active_connections[video_id].discard(websocket)
        user_id = self.connection_users.pop(websocket, None)

        # Clean up empty video entries
        if not self.active_connections[video_id]:
            del self.active_connections[video_id]

        logger.info(
            f"WebSocket disconnected",
            video_id=video_id,
            user_id=user_id,
            remaining_connections=len(self.active_connections.get(video_id, [])),
        )

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """
        Send message to specific WebSocket connection

        Args:
            message: Message data
            websocket: Target WebSocket
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")

    async def broadcast_to_video(self, video_id: int, message: dict):
        """
        Broadcast message to all connections watching a video

        Args:
            video_id: Video ID
            message: Message data
        """
        if video_id not in self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections[video_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to connection: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection, video_id)

    async def send_progress_update(
        self,
        video_id: int,
        job_type: str,
        progress: int,
        message: str = "",
        metadata: dict = None,
    ):
        """
        Send progress update for a processing job

        Args:
            video_id: Video ID
            job_type: Type of job (frame_extraction, sam_inference, etc.)
            progress: Progress percentage (0-100)
            message: Optional status message
            metadata: Optional additional data
        """
        update = {
            "type": "progress",
            "video_id": video_id,
            "job_type": job_type,
            "progress": progress,
            "message": message,
            "metadata": metadata or {},
        }
        await self.broadcast_to_video(video_id, update)

    async def send_error(self, video_id: int, job_type: str, error: str):
        """
        Send error notification

        Args:
            video_id: Video ID
            job_type: Type of job
            error: Error message
        """
        update = {
            "type": "error",
            "video_id": video_id,
            "job_type": job_type,
            "error": error,
        }
        await self.broadcast_to_video(video_id, update)

    async def send_completion(self, video_id: int, job_type: str, result: dict = None):
        """
        Send job completion notification

        Args:
            video_id: Video ID
            job_type: Type of job
            result: Optional result data
        """
        update = {
            "type": "complete",
            "video_id": video_id,
            "job_type": job_type,
            "result": result or {},
        }
        await self.broadcast_to_video(video_id, update)

    def get_connection_count(self, video_id: int) -> int:
        """
        Get number of active connections for a video

        Args:
            video_id: Video ID

        Returns:
            Connection count
        """
        return len(self.active_connections.get(video_id, []))


# Global connection manager instance
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, video_id: int, user_id: int):
    """
    WebSocket endpoint for video processing updates

    Args:
        websocket: WebSocket connection
        video_id: Video ID to subscribe to
        user_id: Authenticated user ID
    """
    await manager.connect(websocket, video_id, user_id)

    try:
        # Send initial connection confirmation
        await manager.send_personal_message(
            {
                "type": "connected",
                "video_id": video_id,
                "message": "Connected to video processing updates",
            },
            websocket,
        )

        # Keep connection alive and handle incoming messages
        while True:
            # Wait for messages (ping/pong, etc.)
            data = await websocket.receive_text()

            # Handle ping
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        manager.disconnect(websocket, video_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", video_id=video_id, user_id=user_id)
        manager.disconnect(websocket, video_id)


def get_connection_manager() -> ConnectionManager:
    """
    Get global connection manager instance
    """
    return manager
