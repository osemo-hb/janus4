"""
Janus3 Configuration

Centralized settings using pydantic-settings for environment variable support.
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    DATABASE_URL: str = "postgresql://janus:janus_dev@localhost:5432/janus3_db"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Redis Streams Configuration
    STM_STREAM_PREFIX: str = "stm"
    CORTEX_CONSUMER_GROUP: str = "cortex_workers"
    STM_MAX_LEN: int = 1000  # XTRIM MAXLEN for stream retention

    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # Adaptive Threshold Configuration
    # Pre-calculated global average velocities for cold start (Fix 4)
    # These represent typical semantic drift values from training data
    COLD_START_VELOCITIES: List[float] = [
        0.32, 0.28, 0.35, 0.41, 0.38, 0.29, 0.33, 0.40,
        0.36, 0.31, 0.42, 0.37, 0.34, 0.30, 0.39, 0.27,
        0.43, 0.35, 0.32, 0.38
    ]
    ADAPTIVE_THRESHOLD_WINDOW: int = 50
    ADAPTIVE_THRESHOLD_MULTIPLIER: float = 1.5  # std multiplier for boundary detection

    # Retrieval Configuration
    RETRIEVAL_TOP_K_ENTITIES: int = 10
    RETRIEVAL_TOP_K_EPISODES: int = 5
    STM_CONTEXT_TURNS: int = 10

    # Consolidation Configuration
    MIN_TURNS_FOR_CONSOLIDATION: int = 5
    CONSOLIDATION_IDLE_TIMEOUT_MS: int = 60000  # 60 seconds for crash recovery

    # OpenAI
    OPENAI_API_KEY: str = ""
    GENERATION_MODEL: str = "gpt-4o-mini"

    # GLiNER Entity Extraction
    GLINER_MODEL: str = "urchade/gliner_base"
    GLINER_LABELS: List[str] = [
        "person", "organization", "location", "concept", "product", "event"
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton instance
settings = Settings()
