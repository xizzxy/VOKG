"""
Celery application configuration for distributed task processing
Manages GPU, CPU, and LLM queues
"""

from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure
from kombu import Queue, Exchange

from .config import settings
from .logging import get_logger

logger = get_logger(__name__)


# Create Celery app
celery_app = Celery(
    "vokg",
    broker=settings.celery_broker,
    backend=settings.celery_backend,
    include=[
        "backend.services.frame_extraction.worker",
        "backend.services.sam_inference.worker",
        "backend.services.interaction_detection.worker",
        "backend.services.graph_generator.worker",
        "backend.services.llm_reasoning.worker",
    ],
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=settings.CELERY_TASK_TRACK_STARTED,
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
    task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # Important for GPU tasks
    worker_max_tasks_per_child=50,  # Restart workers to prevent memory leaks
    # Result backend settings
    result_expires=3600,  # 1 hour
    result_backend_transport_options={
        "master_name": "vokg-master",
        "retry_on_timeout": True,
    },
    # Routing
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",
)

# Define task queues with priorities
celery_app.conf.task_queues = (
    # Default queue for miscellaneous tasks
    Queue("default", Exchange("default"), routing_key="default", priority=5),
    # CPU-intensive tasks (frame extraction, interaction detection)
    Queue("cpu", Exchange("cpu"), routing_key="cpu", priority=7),
    # GPU tasks (SAM inference) - highest priority
    Queue("gpu", Exchange("gpu"), routing_key="gpu", priority=10),
    # LLM reasoning tasks
    Queue("llm", Exchange("llm"), routing_key="llm", priority=6),
    # Graph generation tasks
    Queue("graph", Exchange("graph"), routing_key="graph", priority=8),
)

# Task routing
celery_app.conf.task_routes = {
    "backend.services.frame_extraction.worker.*": {"queue": "cpu"},
    "backend.services.sam_inference.worker.*": {"queue": "gpu"},
    "backend.services.interaction_detection.worker.*": {"queue": "cpu"},
    "backend.services.graph_generator.worker.*": {"queue": "graph"},
    "backend.services.llm_reasoning.worker.*": {"queue": "llm"},
}


# Signal handlers for logging
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **extra):
    """
    Log task start
    """
    logger.info(
        f"Task started: {task.name}",
        task_id=task_id,
        task_name=task.name,
        args=str(args)[:100],
        kwargs=str(kwargs)[:100],
    )


@task_postrun.connect
def task_postrun_handler(
    sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, **extra
):
    """
    Log task completion
    """
    logger.info(
        f"Task completed: {task.name}",
        task_id=task_id,
        task_name=task.name,
        result=str(retval)[:100],
    )


@task_failure.connect
def task_failure_handler(
    sender=None, task_id=None, exception=None, args=None, kwargs=None, traceback=None, **extra
):
    """
    Log task failure
    """
    logger.error(
        f"Task failed: {sender.name}",
        task_id=task_id,
        task_name=sender.name,
        exception=str(exception),
        traceback=str(traceback)[:500],
    )


# Task base class with common functionality
class VOKGTask(celery_app.Task):
    """
    Base task class with error handling and progress tracking
    """

    autoretry_for = (Exception,)
    retry_kwargs = {"max_retries": 3}
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """
        Handle task failure
        """
        logger.error(
            f"Task {self.name} failed",
            task_id=task_id,
            exception=str(exc),
            error_info=str(einfo)[:500],
        )
        super().on_failure(exc, task_id, args, kwargs, einfo)

    def update_progress(self, current: int, total: int, message: str = ""):
        """
        Update task progress for WebSocket notifications
        """
        self.update_state(
            state="PROGRESS",
            meta={
                "current": current,
                "total": total,
                "percent": int((current / total) * 100) if total > 0 else 0,
                "message": message,
            },
        )


# Export celery app
__all__ = ["celery_app", "VOKGTask"]
