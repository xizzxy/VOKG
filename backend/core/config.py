"""
Configuration management for VOKG backend
Loads all environment variables and validates settings
"""

from functools import lru_cache
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from enviroment variables
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application
    APP_NAME: str = "VOKG"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # API Gateway
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    # Security
    SECRET_KEY: str = Field(..., min_length=32)
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ENCRYPTION_KEY: str = Field(..., min_length=32)

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000

    # PostgreSQL
    POSTGRES_USER: str = "vokg"
    POSTGRES_PASSWORD: str = Field(..., min_length=8)
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "vokg"

    @property
    def POSTGRES_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def POSTGRES_ASYNC_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = Field(..., min_length=8)

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # Celery
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None
    CELERY_TASK_TRACK_STARTED: bool = True
    CELERY_TASK_TIME_LIMIT: int = 3600
    CELERY_TASK_SOFT_TIME_LIMIT: int = 3300

    @property
    def celery_broker(self) -> str:
        return self.CELERY_BROKER_URL or self.REDIS_URL

    @property
    def celery_backend(self) -> str:
        return self.CELERY_RESULT_BACKEND or self.REDIS_URL

    # Storage (S3/MinIO)
    STORAGE_TYPE: str = "minio"  # minio or s3
    S3_ENDPOINT_URL: Optional[str] = "http://localhost:9000"
    S3_ACCESS_KEY_ID: str = Field(..., min_length=3)
    S3_SECRET_ACCESS_KEY: str = Field(..., min_length=8)
    S3_BUCKET_NAME: str = "vokg"
    S3_REGION: str = "us-east-1"

    # Video Processing
    MAX_VIDEO_SIZE_MB: int = 500
    ALLOWED_VIDEO_FORMATS: str = "mp4,avi,mov,mkv,webm"
    FRAME_EXTRACTION_FPS: Optional[float] = None  # None = keyframes only
    FRAME_EXTRACTION_MAX_FRAMES: int = 1000
    FRAME_OUTPUT_FORMAT: str = "jpg"
    FRAME_OUTPUT_QUALITY: int = 85

    @field_validator("ALLOWED_VIDEO_FORMATS")
    @classmethod
    def parse_video_formats(cls, v: str) -> str:
        return v.lower()

    @property
    def allowed_formats_list(self) -> list[str]:
        return [f.strip() for f in self.ALLOWED_VIDEO_FORMATS.split(",")]

    # SAM 2 Configuration
    SAM_MODEL_TYPE: str = "vit_h"  # vit_h, vit_l, vit_b
    SAM_CHECKPOINT_PATH: str = "/models/sam2_hiera_large.pt"
    SAM_CONFIG_PATH: str = "/models/sam2_hiera_l.yaml"
    SAM_DEVICE: str = "cuda"
    SAM_BATCH_SIZE: int = 4
    SAM_POINTS_PER_SIDE: int = 32
    SAM_PRED_IOU_THRESH: float = 0.88
    SAM_STABILITY_SCORE_THRESH: float = 0.95

    # CLIP Configuration (optional classification)
    USE_CLIP: bool = False
    CLIP_MODEL_NAME: str = "ViT-B/32"

    # Object Tracking
    TRACKING_IOU_THRESHOLD: float = 0.3
    TRACKING_MAX_AGE: int = 5
    TRACKING_MIN_HITS: int = 3

    # Interaction Detection
    INTERACTION_TEMPORAL_WINDOW: int = 30  # frames
    INTERACTION_PROXIMITY_THRESHOLD: float = 50.0  # pixels
    INTERACTION_IOU_THRESHOLD: float = 0.1
    INTERACTION_VELOCITY_THRESHOLD: float = 5.0  # pixels/frame

    # Knowledge Graph
    GRAPH_MIN_CONFIDENCE: float = 0.5
    GRAPH_MAX_NODES: int = 10000
    GRAPH_MAX_EDGES: int = 50000

    # LLM Configuration
    LLM_PROVIDER: str = "openai"  # openai or gemini
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_MAX_TOKENS: int = 4096
    OPENAI_TEMPERATURE: float = 0.7

    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-pro"

    LLM_MAX_RETRIES: int = 3
    LLM_TIMEOUT: int = 120

    @field_validator("LLM_PROVIDER")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        if v not in ["openai", "gemini"]:
            raise ValueError("LLM_PROVIDER must be 'openai' or 'gemini'")
        return v

    # WebSocket
    WS_MESSAGE_QUEUE_SIZE: int = 100
    WS_HEARTBEAT_INTERVAL: int = 30

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json or text
    LOG_FILE: Optional[str] = None

    # Monitoring
    SENTRY_DSN: Optional[str] = None
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instanc
    """
    return Settings()


# Export settings instace
settings = get_settings()
