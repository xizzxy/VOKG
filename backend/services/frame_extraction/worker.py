"""
Frame extraction Celery worker
Extracts frames from video using FFmpeg
"""

import subprocess
from pathlib import Path
from typing import List
import tempfile
import shutil

from backend.core.celery_app import celery_app, VOKGTask
from backend.core.config import settings
from backend.core.storage import get_storage_client
from backend.core.logging import get_logger
from backend.database.postgres import get_db_session
from backend.database.models import Video, Frame, ProcessingJob, ProcessingStatus
from backend.utils.video_utils import extract_video_metadata
from datetime import datetime

logger = get_logger(__name__)


@celery_app.task(base=VOKGTask, bind=True, name="extract_frames")
def extract_frames_task(self, video_id: int):
    """
    Extract frames from video

    Args:
        video_id: Video ID

    Returns:
        Dictionary with frame extraction results
    """
    logger.info(f"Starting frame extraction", video_id=video_id, task_id=self.request.id)

    db = get_db_session()
    storage = get_storage_client()
    temp_dir = Path(tempfile.mkdtemp(prefix=f"vokg_frames_{video_id}_"))

    try:
        # Get video from database
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise ValueError(f"Video {video_id} not found")

        # Update job status
        job = (
            db.query(ProcessingJob)
            .filter(
                ProcessingJob.video_id == video_id,
                ProcessingJob.job_type == "frame_extraction",
            )
            .first()
        )

        # Download video from storage
        logger.info(f"Downloading video from storage", video_id=video_id)
        video_path = temp_dir / "input_video.mp4"
        storage.download_file(video.storage_path, video_path)

        # Extract frames
        logger.info(f"Extracting frames", video_id=video_id)
        frame_paths = extract_frames(video_path, temp_dir, video.fps)

        total_frames = len(frame_paths)
        logger.info(f"Extracted {total_frames} frames", video_id=video_id)

        # Upload frames and create database records
        for idx, frame_path in enumerate(frame_paths):
            # Update progress
            progress = int((idx / total_frames) * 100)
            self.update_progress(idx + 1, total_frames, f"Uploading frame {idx + 1}/{total_frames}")

            # Calculate timestamp
            timestamp = (idx / video.fps) if video.fps > 0 else idx

            # Upload to storage
            storage_path = storage.upload_frame(video_id, idx, frame_path)

            # Create database record
            frame_record = Frame(
                video_id=video_id,
                frame_number=idx,
                timestamp=timestamp,
                storage_path=storage_path,
                width=video.width,
                height=video.height,
                is_keyframe=False,  # TODO: detect keyframes
            )
            db.add(frame_record)

            # Commit in batches
            if (idx + 1) % 50 == 0:
                db.commit()

        # Final commit
        db.commit()

        # Update video statistics
        video.total_frames = total_frames
        video.processed_frames = total_frames
        db.commit()

        # Update job status
        if job:
            job.status = ProcessingStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.result_data = {"total_frames": total_frames}
            db.commit()

        logger.info(f"Frame extraction completed", video_id=video_id, total_frames=total_frames)

        # Trigger next stage: SAM inference
        from backend.services.sam_inference.worker import sam_inference_task

        sam_task = sam_inference_task.apply_async(args=[video_id], queue="gpu")
        logger.info(f"SAM inference task queued", video_id=video_id, task_id=sam_task.id)

        # Create SAM job record
        sam_job = ProcessingJob(
            video_id=video_id,
            job_type="sam_inference",
            celery_task_id=sam_task.id,
            status=ProcessingStatus.PROCESSING,
            started_at=datetime.utcnow(),
        )
        db.add(sam_job)
        db.commit()

        return {
            "video_id": video_id,
            "total_frames": total_frames,
            "status": "completed",
        }

    except Exception as e:
        logger.error(f"Frame extraction failed: {e}", video_id=video_id, exc_info=True)

        # Update job status
        if job:
            job.status = ProcessingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            db.commit()

        # Update video status
        video.status = ProcessingStatus.FAILED
        video.error_message = f"Frame extraction failed: {str(e)}"
        db.commit()

        raise

    finally:
        # Cleanup
        db.close()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def extract_frames(
    video_path: Path, output_dir: Path, fps: float = None
) -> List[Path]:
    """
    Extract frames from video using FFmpeg

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        fps: FPS for extraction (None = keyframes only)

    Returns:
        List of frame file paths
    """
    output_pattern = output_dir / "frame_%06d.jpg"

    # Build FFmpeg command
    cmd = ["ffmpeg", "-i", str(video_path)]

    if fps is None or settings.FRAME_EXTRACTION_FPS is None:
        # Extract keyframes only
        cmd.extend(["-vf", "select='eq(pict_type,I)'", "-vsync", "vfr"])
    else:
        # Extract at specific FPS
        target_fps = settings.FRAME_EXTRACTION_FPS or fps
        cmd.extend(["-vf", f"fps={target_fps}"])

    # Quality and format settings
    cmd.extend(
        [
            "-q:v",
            str(settings.FRAME_OUTPUT_QUALITY),
            "-frames:v",
            str(settings.FRAME_EXTRACTION_MAX_FRAMES),
            str(output_pattern),
        ]
    )

    logger.debug(f"FFmpeg command: {' '.join(cmd)}")

    # Execute FFmpeg
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")

    # Get list of extracted frames
    frame_paths = sorted(output_dir.glob("frame_*.jpg"))

    return frame_paths
