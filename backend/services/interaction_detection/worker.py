"""
Interaction detection Celery worker
Detects interactions between tracked objects
"""

from collections import defaultdict
from typing import List, Dict

from backend.core.celery_app import celery_app, VOKGTask
from backend.core.logging import get_logger
from backend.database.postgres import get_db_session
from backend.database.models import Video, DetectedObject, Interaction, ProcessingJob, ProcessingStatus, Frame
from backend.services.interaction_detection.heuristics import (
    detect_proximity_interaction,
    detect_containment_interaction,
    detect_occlusion_interaction,
    detect_temporal_interactions,
)
from datetime import datetime

logger = get_logger(__name__)


@celery_app.task(base=VOKGTask, bind=True, name="interaction_detection")
def interaction_detection_task(self, video_id: int):
    """
    Detect interactions between objects

    Args:
        video_id: Video ID

    Returns:
        Dictionary with detection results
    """
    logger.info(
        f"Starting interaction detection", video_id=video_id, task_id=self.request.id
    )

    db = get_db_session()

    try:
        # Get video
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise ValueError(f"Video {video_id} not found")

        # Get all detected objects
        objects = (
            db.query(DetectedObject)
            .filter(DetectedObject.video_id == video_id)
            .join(Frame)
            .order_by(Frame.frame_number)
            .all()
        )

        if not objects:
            logger.warning(f"No objects found for video {video_id}")
            return {"video_id": video_id, "total_interactions": 0}

        logger.info(f"Detecting interactions for {len(objects)} objects", video_id=video_id)

        # Group objects by frame for spatial interactions
        objects_by_frame = defaultdict(list)
        for obj in objects:
            objects_by_frame[obj.frame.frame_number].append({
                "id": obj.id,
                "object_id": obj.object_id,
                "frame": obj.frame.frame_number,
                "bbox": [obj.bbox_x1, obj.bbox_y1, obj.bbox_x2, obj.bbox_y2],
                "confidence": obj.confidence,
                "area": obj.area,
            })

        # Group objects by tracking ID for temporal interactions
        objects_by_track = defaultdict(list)
        for obj in objects:
            objects_by_track[obj.object_id].append({
                "frame": obj.frame.frame_number,
                "bbox": [obj.bbox_x1, obj.bbox_y1, obj.bbox_x2, obj.bbox_y2],
                "confidence": obj.confidence,
            })

        total_interactions = 0
        total_frames = len(objects_by_frame)

        # Detect spatial interactions (frame by frame)
        for frame_idx, (frame_num, frame_objects) in enumerate(objects_by_frame.items()):
            # Update progress
            progress = int((frame_idx / total_frames) * 50)  # First 50% for spatial
            self.update_progress(
                frame_idx + 1,
                total_frames,
                f"Detecting spatial interactions: frame {frame_idx + 1}/{total_frames}",
            )

            # Check each pair of objects in the frame
            for i, obj1 in enumerate(frame_objects):
                for obj2 in frame_objects[i + 1 :]:
                    # Skip if same object
                    if obj1["object_id"] == obj2["object_id"]:
                        continue

                    # Detect proximity
                    proximity = detect_proximity_interaction(obj1, obj2)
                    if proximity:
                        save_interaction(
                            db,
                            video_id,
                            obj1["object_id"],
                            obj2["object_id"],
                            proximity["type"],
                            frame_num,
                            frame_num,
                            proximity["confidence"],
                            proximity["metadata"],
                        )
                        total_interactions += 1

                    # Detect containment
                    containment = detect_containment_interaction(obj1, obj2)
                    if containment:
                        save_interaction(
                            db,
                            video_id,
                            obj1["object_id"],
                            obj2["object_id"],
                            containment["type"],
                            frame_num,
                            frame_num,
                            containment["confidence"],
                            containment["metadata"],
                        )
                        total_interactions += 1

                    # Detect occlusion
                    occlusion = detect_occlusion_interaction(obj1, obj2)
                    if occlusion:
                        save_interaction(
                            db,
                            video_id,
                            obj1["object_id"],
                            obj2["object_id"],
                            occlusion["type"],
                            frame_num,
                            frame_num,
                            occlusion["confidence"],
                            occlusion["metadata"],
                        )
                        total_interactions += 1

            # Commit in batches
            if (frame_idx + 1) % 50 == 0:
                db.commit()

        # Commit spatial interactions
        db.commit()

        # Detect temporal interactions
        self.update_progress(
            total_frames, total_frames, "Detecting temporal interactions..."
        )

        temporal_interactions = detect_temporal_interactions(objects_by_track)

        for interaction in temporal_interactions:
            save_interaction(
                db,
                video_id,
                interaction["object_id_1"],
                interaction["object_id_2"],
                interaction["type"],
                interaction["start_frame"],
                interaction["end_frame"],
                interaction["confidence"],
                interaction["metadata"],
            )
            total_interactions += 1

        db.commit()

        # Update video statistics
        video.total_interactions = total_interactions
        db.commit()

        # Update job status
        job = (
            db.query(ProcessingJob)
            .filter(
                ProcessingJob.video_id == video_id,
                ProcessingJob.job_type == "interaction_detection",
            )
            .first()
        )
        if job:
            job.status = ProcessingStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.result_data = {"total_interactions": total_interactions}
            db.commit()

        logger.info(
            f"Interaction detection completed",
            video_id=video_id,
            total_interactions=total_interactions,
        )

        # Trigger next stage: Knowledge graph generation
        from backend.services.graph_generator.worker import graph_generation_task

        graph_task = graph_generation_task.apply_async(args=[video_id], queue="graph")
        logger.info(f"Graph generation task queued", video_id=video_id, task_id=graph_task.id)

        # Create graph job record
        graph_job = ProcessingJob(
            video_id=video_id,
            job_type="graph_generation",
            celery_task_id=graph_task.id,
            status=ProcessingStatus.PROCESSING,
            started_at=datetime.utcnow(),
        )
        db.add(graph_job)
        db.commit()

        return {
            "video_id": video_id,
            "total_interactions": total_interactions,
            "status": "completed",
        }

    except Exception as e:
        logger.error(f"Interaction detection failed: {e}", video_id=video_id, exc_info=True)

        # Update job status
        job = (
            db.query(ProcessingJob)
            .filter(
                ProcessingJob.video_id == video_id,
                ProcessingJob.job_type == "interaction_detection",
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
            video.error_message = f"Interaction detection failed: {str(e)}"
            db.commit()

        raise

    finally:
        db.close()


def save_interaction(
    db,
    video_id: int,
    object_id_1: int,
    object_id_2: int,
    interaction_type: str,
    start_frame: int,
    end_frame: int,
    confidence: float,
    metadata: dict,
):
    """
    Save interaction to database

    Args:
        db: Database session
        video_id: Video ID
        object_id_1: First object ID
        object_id_2: Second object ID
        interaction_type: Type of interaction
        start_frame: Start frame
        end_frame: End frame
        confidence: Confidence score
        metadata: Additional metadata
    """
    interaction = Interaction(
        video_id=video_id,
        object_id_1=object_id_1,
        object_id_2=object_id_2,
        interaction_type=interaction_type,
        start_frame=start_frame,
        end_frame=end_frame,
        confidence=confidence,
        metadata=metadata,
    )
    db.add(interaction)
