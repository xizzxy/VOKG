"""
Video ingestion service
Coordinates video processing pipeline
"""

from datetime import datetime
from sqlalchemy.orm import Session

from backend.database.models import Video, ProcessingJob, ProcessingStatus
from backend.core.logging import get_logger
from backend.services.frame_extraction.worker import extract_frames_task
from backend.api_gateway.websocket import get_connection_manager

logger = get_logger(__name__)


def ingest_video(video_id: int, db: Session):
    """
    Start video processing pipeline

    Args:
        video_id: Video ID
        db: Database session
    """
    logger.info(f"Starting video ingestion", video_id=video_id)

    # Update video status
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        logger.error(f"Video not found", video_id=video_id)
        return

    video.status = ProcessingStatus.PROCESSING
    video.processing_started_at = datetime.utcnow()
    db.commit()

    # Create frame extraction job
    job = ProcessingJob(
        video_id=video_id,
        job_type="frame_extraction",
        status=ProcessingStatus.PENDING,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Queue frame extraction task
    try:
        task = extract_frames_task.apply_async(args=[video_id], queue="cpu")
        job.celery_task_id = task.id
        job.status = ProcessingStatus.PROCESSING
        job.started_at = datetime.utcnow()
        db.commit()

        logger.info(
            f"Frame extraction task queued",
            video_id=video_id,
            job_id=job.id,
            task_id=task.id,
        )

        # Notify via WebSocket
        ws_manager = get_connection_manager()
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.create_task(
            ws_manager.send_progress_update(
                video_id=video_id,
                job_type="frame_extraction",
                progress=0,
                message="Frame extraction started",
            )
        )

    except Exception as e:
        logger.error(f"Failed to queue frame extraction: {e}", video_id=video_id)
        job.status = ProcessingStatus.FAILED
        job.error_message = str(e)
        video.status = ProcessingStatus.FAILED
        video.error_message = str(e)
        db.commit()
        raise
