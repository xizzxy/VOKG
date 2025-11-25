"""
SQLAlchemy models for PostgreSQL database
Stores metadata, processing jobs, and relationships
"""

from datetime import datetime
from typing import Optional
from enum import Enum as PyEnum

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    ForeignKey,
    Text,
    Float,
    JSON,
    Enum,
    Index,
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class ProcessingStatus(str, PyEnum):
    """Processing job status"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class User(Base):
    """
    User model for authentication and authorization
    """

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Encrypted API keys for LLM services
    openai_api_key_encrypted = Column(Text, nullable=True)
    gemini_api_key_encrypted = Column(Text, nullable=True)

    # Rate limiting
    api_calls_count = Column(Integer, default=0)
    last_api_call = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    videos = relationship("Video", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"


class Video(Base):
    """
    Video metadata and processing status
    """

    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    filename = Column(String(255), nullable=False)
    storage_path = Column(String(512), nullable=False)  # S3/MinIO path
    file_size = Column(Integer, nullable=False)  # bytes
    duration = Column(Float, nullable=True)  # seconds
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    fps = Column(Float, nullable=True)
    codec = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Processing status
    status = Column(
        Enum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False, index=True
    )
    processing_started_at = Column(DateTime(timezone=True), nullable=True)
    processing_completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)

    # Statistics
    total_frames = Column(Integer, default=0)
    processed_frames = Column(Integer, default=0)
    total_objects = Column(Integer, default=0)
    total_interactions = Column(Integer, default=0)

    # Relationships
    user = relationship("User", back_populates="videos")
    jobs = relationship("ProcessingJob", back_populates="video", cascade="all, delete-orphan")
    frames = relationship("Frame", back_populates="video", cascade="all, delete-orphan")
    graph_snapshots = relationship(
        "GraphSnapshot", back_populates="video", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("idx_video_user_status", "user_id", "status"),)

    def __repr__(self):
        return f"<Video(id={self.id}, title={self.title}, status={self.status})>"


class ProcessingJob(Base):
    """
    Individual processing jobs for pipeline stages
    """

    __tablename__ = "processing_jobs"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    job_type = Column(
        String(50), nullable=False, index=True
    )  # frame_extraction, sam_inference, etc.
    celery_task_id = Column(String(255), unique=True, index=True, nullable=True)
    status = Column(
        Enum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False, index=True
    )
    progress = Column(Integer, default=0)  # 0-100
    error_message = Column(Text, nullable=True)
    result_data = Column(JSON, nullable=True)  # Store task results
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    video = relationship("Video", back_populates="jobs")

    __table_args__ = (
        Index("idx_job_video_type", "video_id", "job_type"),
        Index("idx_job_status", "status"),
    )

    def __repr__(self):
        return f"<ProcessingJob(id={self.id}, type={self.job_type}, status={self.status})>"


class Frame(Base):
    """
    Extracted video frames
    """

    __tablename__ = "frames"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)  # seconds
    storage_path = Column(String(512), nullable=False)  # S3/MinIO path
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    is_keyframe = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    video = relationship("Video", back_populates="frames")
    detected_objects = relationship(
        "DetectedObject", back_populates="frame", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_frame_video_number", "video_id", "frame_number", unique=True),
        Index("idx_frame_keyframe", "video_id", "is_keyframe"),
    )

    def __repr__(self):
        return f"<Frame(id={self.id}, video_id={self.video_id}, frame={self.frame_number})>"


class DetectedObject(Base):
    """
    Objects detected by SAM in each frame
    """

    __tablename__ = "detected_objects"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    frame_id = Column(Integer, ForeignKey("frames.id"), nullable=False)
    object_id = Column(Integer, nullable=False)  # Tracking ID across frames
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    area = Column(Float, nullable=False)  # pixels
    mask_storage_path = Column(String(512), nullable=True)  # S3/MinIO path to mask PNG
    embedding_vector = Column(JSON, nullable=True)  # SAM embedding (stored as list)
    clip_label = Column(String(100), nullable=True)  # Optional CLIP classification
    clip_confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    video_id_fk = Column(Integer, ForeignKey("videos.id"))
    frame = relationship("Frame", back_populates="detected_objects")

    __table_args__ = (
        Index("idx_object_video_id", "video_id", "object_id"),
        Index("idx_object_frame", "frame_id"),
    )

    def __repr__(self):
        return f"<DetectedObject(id={self.id}, object_id={self.object_id}, frame={self.frame_id})>"


class Interaction(Base):
    """
    Detected interactions between objects
    """

    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    object_id_1 = Column(Integer, nullable=False)
    object_id_2 = Column(Integer, nullable=False)
    interaction_type = Column(String(50), nullable=False)  # proximity, containment, chase, etc.
    start_frame = Column(Integer, nullable=False)
    end_frame = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    metadata = Column(JSON, nullable=True)  # Additional interaction details
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("idx_interaction_video", "video_id"),)

    def __repr__(self):
        return f"<Interaction(id={self.id}, type={self.interaction_type}, objects={self.object_id_1}-{self.object_id_2})>"


class GraphSnapshot(Base):
    """
    Knowledge graph snapshots for versioning
    """

    __tablename__ = "graph_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    version = Column(Integer, nullable=False)
    storage_path = Column(String(512), nullable=False)  # S3/MinIO path to graph JSON
    node_count = Column(Integer, nullable=False)
    edge_count = Column(Integer, nullable=False)
    graph_metrics = Column(JSON, nullable=True)  # Centrality, density, etc.
    llm_narrative = Column(Text, nullable=True)  # Generated narrative
    llm_model = Column(String(50), nullable=True)  # Model used for reasoning
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    video = relationship("Video", back_populates="graph_snapshots")

    __table_args__ = (
        Index("idx_graph_video_version", "video_id", "version", unique=True),
    )

    def __repr__(self):
        return f"<GraphSnapshot(id={self.id}, video_id={self.video_id}, version={self.version})>"
