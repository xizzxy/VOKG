"""
SAM 2 inference Celery worker
Runs Segment Anything Model 2 on frames
"""

import tempfile
import shutil
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
from PIL import Image
import io

from backend.core.celery_app import celery_app, VOKGTask
from backend.core.config import settings
from backend.core.storage import get_storage_client
from backend.core.logging import get_logger
from backend.database.postgres import get_db_session
from backend.database.models import Video, Frame, DetectedObject, ProcessingJob, ProcessingStatus
from backend.services.sam_inference.tracker import ObjectTracker, bbox_from_mask
from datetime import datetime

logger = get_logger(__name__)


@celery_app.task(base=VOKGTask, bind=True, name="sam_inference")
def sam_inference_task(self, video_id: int):
    """
    Run SAM 2 inference on all frames

    Args:
        video_id: Video ID

    Returns:
        Dictionary with inference results
    """
    logger.info(f"Starting SAM 2 inference", video_id=video_id, task_id=self.request.id)

    db = get_db_session()
    storage = get_storage_client()
    temp_dir = Path(tempfile.mkdtemp(prefix=f"vokg_sam_{video_id}_"))

    try:
        # Get video from database
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise ValueError(f"Video {video_id} not found")

        # Get frames
        frames = (
            db.query(Frame)
            .filter(Frame.video_id == video_id)
            .order_by(Frame.frame_number)
            .all()
        )

        if not frames:
            raise ValueError(f"No frames found for video {video_id}")

        logger.info(f"Processing {len(frames)} frames with SAM 2", video_id=video_id)

        # Initialize SAM 2 model
        sam_model = load_sam2_model()

        # Initialize object tracker
        tracker = ObjectTracker()

        total_objects = 0

        # Process each frame
        for idx, frame in enumerate(frames):
            # Update progress
            progress = int((idx / len(frames)) * 100)
            self.update_progress(
                idx + 1, len(frames), f"Processing frame {idx + 1}/{len(frames)}"
            )

            # Download frame
            frame_path = temp_dir / f"frame_{frame.frame_number}.jpg"
            storage.download_file(frame.storage_path, frame_path)

            # Run SAM 2 inference
            detections = run_sam2_on_frame(sam_model, frame_path)

            # Track objects across frames
            tracked_detections = tracker.update(detections)

            # Save detections to database
            for detection in tracked_detections:
                # Convert mask to PNG bytes
                mask_bytes = mask_to_png(detection["mask"])

                # Upload mask to storage
                mask_path = storage.upload_mask(
                    video_id, frame.frame_number, detection["object_id"], mask_bytes
                )

                # Create database record
                obj = DetectedObject(
                    video_id=video_id,
                    frame_id=frame.id,
                    object_id=detection["object_id"],
                    bbox_x1=detection["bbox"][0],
                    bbox_y1=detection["bbox"][1],
                    bbox_x2=detection["bbox"][2],
                    bbox_y2=detection["bbox"][3],
                    confidence=detection["confidence"],
                    area=detection["area"],
                    mask_storage_path=mask_path,
                    embedding_vector=detection.get("embedding"),
                )
                db.add(obj)
                total_objects += 1

            # Commit in batches
            if (idx + 1) % 10 == 0:
                db.commit()

        # Final commit
        db.commit()

        # Update video statistics
        video.total_objects = tracker.next_id  # Total unique objects
        db.commit()

        # Update job status
        job = (
            db.query(ProcessingJob)
            .filter(
                ProcessingJob.video_id == video_id,
                ProcessingJob.job_type == "sam_inference",
            )
            .first()
        )
        if job:
            job.status = ProcessingStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.result_data = {"total_objects": total_objects, "unique_objects": tracker.next_id}
            db.commit()

        logger.info(
            f"SAM 2 inference completed",
            video_id=video_id,
            total_detections=total_objects,
            unique_objects=tracker.next_id,
        )

        # Trigger next stage: Interaction detection
        from backend.services.interaction_detection.worker import interaction_detection_task

        interaction_task = interaction_detection_task.apply_async(args=[video_id], queue="cpu")
        logger.info(
            f"Interaction detection task queued", video_id=video_id, task_id=interaction_task.id
        )

        # Create interaction job record
        interaction_job = ProcessingJob(
            video_id=video_id,
            job_type="interaction_detection",
            celery_task_id=interaction_task.id,
            status=ProcessingStatus.PROCESSING,
            started_at=datetime.utcnow(),
        )
        db.add(interaction_job)
        db.commit()

        return {
            "video_id": video_id,
            "total_detections": total_objects,
            "unique_objects": tracker.next_id,
            "status": "completed",
        }

    except Exception as e:
        logger.error(f"SAM 2 inference failed: {e}", video_id=video_id, exc_info=True)

        # Update job status
        job = (
            db.query(ProcessingJob)
            .filter(
                ProcessingJob.video_id == video_id,
                ProcessingJob.job_type == "sam_inference",
            )
            .first()
        )
        if job:
            job.status = ProcessingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            db.commit()

        # Update video status
        video = db.query(Video).filter(Video.id == video_id).first()
        if video:
            video.status = ProcessingStatus.FAILED
            video.error_message = f"SAM inference failed: {str(e)}"
            db.commit()

        raise

    finally:
        # Cleanup
        db.close()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def load_sam2_model():
    """
    Load SAM 2 model

    Returns:
        SAM 2 model instance
    """
    try:
        # Import SAM 2 (assumes sam2 package is installed)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        # Build SAM 2 model
        sam2_checkpoint = settings.SAM_CHECKPOINT_PATH
        model_cfg = settings.SAM_CONFIG_PATH

        logger.info(f"Loading SAM 2 model from {sam2_checkpoint}")

        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=settings.SAM_DEVICE)
        predictor = SAM2ImagePredictor(sam2)

        logger.info("SAM 2 model loaded successfully")
        return predictor

    except ImportError:
        logger.error("SAM 2 not installed. Install with: pip install git+https://github.com/facebookresearch/sam2.git")
        raise
    except Exception as e:
        logger.error(f"Failed to load SAM 2 model: {e}")
        raise


def run_sam2_on_frame(model, frame_path: Path) -> List[Dict]:
    """
    Run SAM 2 on a single frame

    Args:
        model: SAM 2 predictor
        frame_path: Path to frame image

    Returns:
        List of detections with masks, bboxes, and embeddings
    """
    # Load image
    image = Image.open(frame_path).convert("RGB")
    image_np = np.array(image)

    # Set image in predictor
    model.set_image(image_np)

    # Generate automatic masks using point grid
    # SAM 2 automatic mask generation
    points_per_side = settings.SAM_POINTS_PER_SIDE
    pred_iou_thresh = settings.SAM_PRED_IOU_THRESH
    stability_score_thresh = settings.SAM_STABILITY_SCORE_THRESH

    # Generate point grid
    h, w = image_np.shape[:2]
    points = []
    for i in range(points_per_side):
        for j in range(points_per_side):
            x = (j + 0.5) * w / points_per_side
            y = (i + 0.5) * h / points_per_side
            points.append([x, y])

    points = np.array(points)
    labels = np.ones(len(points), dtype=int)

    # Predict masks
    masks, scores, _ = model.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
    )

    # Filter masks by quality
    detections = []
    for mask, score in zip(masks, scores):
        if score < pred_iou_thresh:
            continue

        # Extract bbox from mask
        bbox = bbox_from_mask(mask)
        area = np.sum(mask)

        if area < 100:  # Skip very small masks
            continue

        detection = {
            "mask": mask,
            "bbox": bbox,
            "confidence": float(score),
            "area": float(area),
            "embedding": None,  # TODO: Extract SAM embedding if needed
        }
        detections.append(detection)

    logger.debug(f"Detected {len(detections)} objects in frame", frame=frame_path.name)

    return detections


def mask_to_png(mask: np.ndarray) -> bytes:
    """
    Convert binary mask to PNG bytes

    Args:
        mask: Binary mask (H, W)

    Returns:
        PNG image bytes
    """
    # Convert to PIL Image
    mask_uint8 = (mask * 255).astype(np.uint8)
    img = Image.fromarray(mask_uint8, mode="L")

    # Save to bytes
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()
