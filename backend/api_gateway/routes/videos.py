"""
Video upload and management endpoints
"""

import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from datetime import datetime

from backend.database.postgres import get_db
from backend.database.models import User, Video, ProcessingJob, ProcessingStatus
from backend.api_gateway.dependencies import get_current_active_user, verify_rate_limit
from backend.core.config import settings
from backend.core.storage import get_storage_client
from backend.core.logging import get_logger
from backend.utils.video_utils import validate_video_file, extract_video_metadata
from backend.services.video_ingestion.service import ingest_video

logger = get_logger(__name__)

router = APIRouter(prefix="/videos", tags=["videos"])


# Pydantic schemas
class VideoResponse(BaseModel):
    id: int
    title: str
    description: Optional[str]
    filename: str
    duration: Optional[float]
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    status: str
    total_frames: int
    processed_frames: int
    total_objects: int
    total_interactions: int
    created_at: datetime
    processing_started_at: Optional[datetime]
    processing_completed_at: Optional[datetime]
    error_message: Optional[str]

    class Config:
        from_attributes = True


class VideoListResponse(BaseModel):
    total: int
    videos: List[VideoResponse]


class VideoUpdateRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None


@router.post("/upload", response_model=VideoResponse, status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: UploadFile = File(...),
    title: str = Query(..., min_length=1, max_length=255),
    description: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    _rate_limit: bool = Depends(verify_rate_limit),
):
    """
    Upload a video file for processing

    Args:
        file: Video file upload
        title: Video title
        description: Optional video description
        db: Database session
        current_user: Authenticated user

    Returns:
        Created video metadata
    """
    logger.info(f"Video upload started", user_id=current_user.id, filename=file.filename)

    # Create temporary file
    temp_dir = Path("/tmp/vokg/uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / f"{datetime.utcnow().timestamp()}_{file.filename}"

    try:
        # Save uploaded file
        with temp_file.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Validate video
        is_valid, error = validate_video_file(temp_file)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid video file: {error}",
            )

        # Extract metadata
        metadata = extract_video_metadata(temp_file)

        # Create video record
        video = Video(
            user_id=current_user.id,
            title=title,
            description=description,
            filename=file.filename,
            storage_path="",  # Will be updated after upload to S3
            file_size=temp_file.stat().st_size,
            duration=metadata["duration"],
            width=metadata["width"],
            height=metadata["height"],
            fps=metadata["fps"],
            codec=metadata["codec"],
            status=ProcessingStatus.PENDING,
        )
        db.add(video)
        db.commit()
        db.refresh(video)

        # Upload to storage
        storage = get_storage_client()
        storage_path = storage.upload_video(str(video.id), temp_file)
        video.storage_path = storage_path
        db.commit()

        logger.info(
            f"Video uploaded successfully",
            video_id=video.id,
            user_id=current_user.id,
            filename=file.filename,
        )

        # Start ingestion process
        ingest_video(video.id, db)

        return video

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video upload failed: {e}", user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload video: {str(e)}",
        )
    finally:
        # Clean up temp file
        if temp_file.exists():
            temp_file.unlink()


@router.get("", response_model=VideoListResponse)
async def list_videos(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status_filter: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    List user's videos

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        status_filter: Optional status filter
        db: Database session
        current_user: Authenticated user

    Returns:
        List of videos
    """
    query = db.query(Video).filter(Video.user_id == current_user.id)

    if status_filter:
        try:
            status_enum = ProcessingStatus(status_filter)
            query = query.filter(Video.status == status_enum)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}",
            )

    total = query.count()
    videos = query.order_by(Video.created_at.desc()).offset(skip).limit(limit).all()

    return VideoListResponse(total=total, videos=videos)


@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get video details

    Args:
        video_id: Video ID
        db: Database session
        current_user: Authenticated user

    Returns:
        Video details
    """
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == current_user.id).first()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )

    return video


@router.patch("/{video_id}", response_model=VideoResponse)
async def update_video(
    video_id: int,
    update_data: VideoUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Update video metadata

    Args:
        video_id: Video ID
        update_data: Update data
        db: Database session
        current_user: Authenticated user

    Returns:
        Updated video
    """
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == current_user.id).first()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )

    # Update fields
    if update_data.title is not None:
        video.title = update_data.title
    if update_data.description is not None:
        video.description = update_data.description

    db.commit()
    db.refresh(video)

    logger.info(f"Video updated", video_id=video_id, user_id=current_user.id)

    return video


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video(
    video_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Delete video and all associated data

    Args:
        video_id: Video ID
        db: Database session
        current_user: Authenticated user
    """
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == current_user.id).first()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )

    # Delete from storage
    try:
        storage = get_storage_client()
        storage.delete_prefix(f"videos/{video_id}/")
        storage.delete_prefix(f"frames/{video_id}/")
        storage.delete_prefix(f"masks/{video_id}/")
        storage.delete_prefix(f"embeddings/{video_id}/")
        storage.delete_prefix(f"graphs/{video_id}/")
    except Exception as e:
        logger.warning(f"Failed to delete storage data: {e}", video_id=video_id)

    # Delete from Neo4j
    try:
        from backend.database.neo4j_client import get_neo4j_client

        neo4j = get_neo4j_client()
        neo4j.delete_video_graph(video_id)
    except Exception as e:
        logger.warning(f"Failed to delete Neo4j graph: {e}", video_id=video_id)

    # Delete from database (cascades to related records)
    db.delete(video)
    db.commit()

    logger.info(f"Video deleted", video_id=video_id, user_id=current_user.id)


@router.post("/{video_id}/reprocess", response_model=VideoResponse)
async def reprocess_video(
    video_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Restart video processing

    Args:
        video_id: Video ID
        db: Database session
        current_user: Authenticated user

    Returns:
        Video with updated status
    """
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == current_user.id).first()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )

    # Reset status
    video.status = ProcessingStatus.PENDING
    video.processing_started_at = None
    video.processing_completed_at = None
    video.error_message = None
    db.commit()

    # Start ingestion
    ingest_video(video_id, db)

    logger.info(f"Video reprocessing started", video_id=video_id, user_id=current_user.id)

    return video


@router.get("/{video_id}/presigned-url")
async def get_video_presigned_url(
    video_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get presigned URL for video download

    Args:
        video_id: Video ID
        db: Database session
        current_user: Authenticated user

    Returns:
        Presigned URL
    """
    video = db.query(Video).filter(Video.id == video_id, Video.user_id == current_user.id).first()

    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )

    storage = get_storage_client()
    url = storage.get_presigned_url(video.storage_path, expiration=3600)

    return {"url": url, "expires_in": 3600}
