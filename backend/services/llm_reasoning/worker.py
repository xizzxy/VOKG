"""
LLM reasoning Celery worker
Generates narrative and explanations using OpenAI or Gemini
"""

import json
from datetime import datetime
from typing import Optional

from backend.core.celery_app import celery_app, VOKGTask
from backend.core.config import settings
from backend.core.logging import get_logger
from backend.core.storage import get_storage_client
from backend.database.postgres import get_db_session
from backend.database.models import Video, GraphSnapshot, ProcessingJob, ProcessingStatus
from backend.services.llm_reasoning.prompts import (
    SYSTEM_PROMPT,
    create_analysis_prompt,
    create_critique_prompt,
    create_revision_prompt,
)

logger = get_logger(__name__)


@celery_app.task(base=VOKGTask, bind=True, name="llm_reasoning")
def llm_reasoning_task(self, video_id: int, snapshot_id: int):
    """
    Generate LLM narrative for video graph

    Args:
        video_id: Video ID
        snapshot_id: Graph snapshot ID

    Returns:
        Dictionary with reasoning results
    """
    logger.info(
        f"Starting LLM reasoning",
        video_id=video_id,
        snapshot_id=snapshot_id,
        task_id=self.request.id,
    )

    db = get_db_session()
    storage = get_storage_client()

    try:
        # Get video and snapshot
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise ValueError(f"Video {video_id} not found")

        snapshot = db.query(GraphSnapshot).filter(GraphSnapshot.id == snapshot_id).first()
        if not snapshot:
            raise ValueError(f"Snapshot {snapshot_id} not found")

        # Download graph data
        self.update_progress(0, 100, "Loading graph data...")
        graph_data_bytes = storage.download_fileobj(snapshot.storage_path)
        graph_data = json.loads(graph_data_bytes.read())

        # Prepare metadata
        video_metadata = {
            "duration": video.duration,
            "width": video.width,
            "height": video.height,
            "fps": video.fps,
            "total_frames": video.total_frames,
        }

        # Generate initial narrative
        self.update_progress(25, 100, "Generating narrative...")
        llm_client = get_llm_client()
        prompt = create_analysis_prompt(graph_data, video_metadata)

        narrative = llm_client.generate(SYSTEM_PROMPT, prompt)

        logger.info(f"Initial narrative generated ({len(narrative)} chars)", video_id=video_id)

        # Multi-pass self-critique (if enabled)
        if settings.LLM_MAX_RETRIES > 1:
            self.update_progress(60, 100, "Performing self-critique...")
            critique_prompt = create_critique_prompt(narrative)
            critique = llm_client.generate(SYSTEM_PROMPT, critique_prompt)

            self.update_progress(80, 100, "Revising narrative...")
            revision_prompt = create_revision_prompt(narrative, critique, graph_data)
            narrative = llm_client.generate(SYSTEM_PROMPT, revision_prompt)

            logger.info(
                f"Narrative revised after critique ({len(narrative)} chars)", video_id=video_id
            )

        # Save narrative to snapshot
        snapshot.llm_narrative = narrative
        snapshot.llm_model = settings.LLM_PROVIDER + ":" + (
            settings.OPENAI_MODEL if settings.LLM_PROVIDER == "openai" else settings.GEMINI_MODEL
        )
        db.commit()

        # Update video status to completed
        video.status = ProcessingStatus.COMPLETED
        video.processing_completed_at = datetime.utcnow()
        db.commit()

        # Update job status
        job = (
            db.query(ProcessingJob)
            .filter(
                ProcessingJob.video_id == video_id,
                ProcessingJob.job_type == "llm_reasoning",
            )
            .first()
        )
        if job:
            job.status = ProcessingStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.result_data = {"narrative_length": len(narrative)}
            db.commit()

        logger.info(f"LLM reasoning completed", video_id=video_id)

        self.update_progress(100, 100, "Processing complete!")

        return {
            "video_id": video_id,
            "snapshot_id": snapshot_id,
            "narrative_length": len(narrative),
            "status": "completed",
        }

    except Exception as e:
        logger.error(f"LLM reasoning failed: {e}", video_id=video_id, exc_info=True)

        # Update job status
        job = (
            db.query(ProcessingJob)
            .filter(
                ProcessingJob.video_id == video_id,
                ProcessingJob.job_type == "llm_reasoning",
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
            video.error_message = f"LLM reasoning failed: {str(e)}"
            db.commit()

        raise

    finally:
        db.close()


class LLMClient:
    """
    Abstract LLM client
    """

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text from prompts"""
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """
    OpenAI API client
    """

    def __init__(self):
        """Initialize OpenAI client"""
        import openai

        # Load API key from settings (loaded from environment variable)
        self.api_key = settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not configured")

        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = settings.OPENAI_MODEL

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text using OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=settings.OPENAI_MAX_TOKENS,
                temperature=settings.OPENAI_TEMPERATURE,
                timeout=settings.LLM_TIMEOUT,
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise


class GeminiClient(LLMClient):
    """
    Google Gemini API client
    """

    def __init__(self):
        """Initialize Gemini client"""
        import google.generativeai as genai

        # Load API key from settings (loaded from environment variable)
        self.api_key = settings.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not configured")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text using Gemini API"""
        try:
            # Gemini combines system and user prompts
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": settings.OPENAI_TEMPERATURE,  # Reuse temp setting
                    "max_output_tokens": settings.OPENAI_MAX_TOKENS,
                },
            )
            return response.text

        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise


def get_llm_client() -> LLMClient:
    """
    Get LLM client based on settings

    Returns:
        LLM client instance
    """
    if settings.LLM_PROVIDER == "openai":
        return OpenAIClient()
    elif settings.LLM_PROVIDER == "gemini":
        return GeminiClient()
    else:
        raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")
