"""
Knowledge graph generation Celery worker
Builds Neo4j graph and triggers LLM reasoning
"""

import json
from datetime import datetime

from backend.core.celery_app import celery_app, VOKGTask
from backend.core.logging import get_logger
from backend.core.storage import get_storage_client
from backend.database.postgres import get_db_session
from backend.database.models import Video, GraphSnapshot, ProcessingJob, ProcessingStatus
from backend.services.graph_generator.builder import GraphBuilder

logger = get_logger(__name__)


@celery_app.task(base=VOKGTask, bind=True, name="graph_generation")
def graph_generation_task(self, video_id: int):
    """
    Generate knowledge graph

    Args:
        video_id: Video ID

    Returns:
        Dictionary with generation results
    """
    logger.info(f"Starting graph generation", video_id=video_id, task_id=self.request.id)

    db = get_db_session()
    storage = get_storage_client()

    try:
        # Get video
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise ValueError(f"Video {video_id} not found")

        # Build graph
        self.update_progress(0, 100, "Building knowledge graph...")
        builder = GraphBuilder(video_id)
        stats = builder.build_graph()

        self.update_progress(50, 100, "Exporting graph data...")

        # Export graph to JSON
        graph_json = builder.export_graph_json()
        builder.close()

        # Upload graph JSON to storage
        graph_storage_path = storage.upload_graph_data(video_id, graph_json)

        self.update_progress(75, 100, "Saving graph snapshot...")

        # Get current version
        latest_snapshot = (
            db.query(GraphSnapshot)
            .filter(GraphSnapshot.video_id == video_id)
            .order_by(GraphSnapshot.version.desc())
            .first()
        )
        version = (latest_snapshot.version + 1) if latest_snapshot else 1

        # Create graph snapshot
        snapshot = GraphSnapshot(
            video_id=video_id,
            version=version,
            storage_path=graph_storage_path,
            node_count=stats["node_count"],
            edge_count=stats["edge_count"],
            graph_metrics=stats.get("metrics", {}),
        )
        db.add(snapshot)
        db.commit()
        db.refresh(snapshot)

        # Update job status
        job = (
            db.query(ProcessingJob)
            .filter(
                ProcessingJob.video_id == video_id,
                ProcessingJob.job_type == "graph_generation",
            )
            .first()
        )
        if job:
            job.status = ProcessingStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.result_data = stats
            db.commit()

        logger.info(
            f"Graph generation completed",
            video_id=video_id,
            node_count=stats["node_count"],
            edge_count=stats["edge_count"],
        )

        self.update_progress(100, 100, "Graph generation complete")

        # Trigger next stage: LLM reasoning
        from backend.services.llm_reasoning.worker import llm_reasoning_task

        llm_task = llm_reasoning_task.apply_async(args=[video_id, snapshot.id], queue="llm")
        logger.info(f"LLM reasoning task queued", video_id=video_id, task_id=llm_task.id)

        # Create LLM job record
        llm_job = ProcessingJob(
            video_id=video_id,
            job_type="llm_reasoning",
            celery_task_id=llm_task.id,
            status=ProcessingStatus.PROCESSING,
            started_at=datetime.utcnow(),
        )
        db.add(llm_job)
        db.commit()

        return {
            "video_id": video_id,
            "snapshot_id": snapshot.id,
            "node_count": stats["node_count"],
            "edge_count": stats["edge_count"],
            "status": "completed",
        }

    except Exception as e:
        logger.error(f"Graph generation failed: {e}", video_id=video_id, exc_info=True)

        # Update job status
        job = (
            db.query(ProcessingJob)
            .filter(
                ProcessingJob.video_id == video_id,
                ProcessingJob.job_type == "graph_generation",
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
            video.error_message = f"Graph generation failed: {str(e)}"
            db.commit()

        raise

    finally:
        db.close()
