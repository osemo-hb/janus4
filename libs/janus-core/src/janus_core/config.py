"""
Janus Core Configuration
"""

from pydantic_settings import BaseSettings
from typing import List


class CoreSettings(BaseSettings):
    """Core library settings loaded from environment variables."""

    # Environment
    ENVIRONMENT: str = "development"

    # Database (PostgreSQL + pgvector)
    DATABASE_URL: str = "postgresql://janus:janus_dev@localhost:5433/janus3_db"

    # Redis
    REDIS_URL: str = "redis://localhost:6380/0"
    STM_STREAM_PREFIX: str = "stm"
    CORTEX_CONSUMER_GROUP: str = "cortex_workers"
    STM_MAX_LEN: int = 1000
    THRESHOLD_REDIS_TTL: int = 604800  # 7 days

    # Embeddings (OpenAI)
    OPENAI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIM: int = 1536

    # Adaptive Threshold
    COLD_START_VELOCITIES: List[float] = [
        0.32, 0.28, 0.35, 0.41, 0.38, 0.29, 0.33, 0.40,
        0.36, 0.31, 0.42, 0.37, 0.34, 0.30, 0.39, 0.27,
        0.43, 0.35, 0.32, 0.38
    ]
    ADAPTIVE_THRESHOLD_WINDOW: int = 50
    ADAPTIVE_THRESHOLD_MULTIPLIER: float = 1.5

    # Retrieval
    RETRIEVAL_TOP_K_ENTITIES: int = 10
    RETRIEVAL_TOP_K_EPISODES: int = 5
    STM_CONTEXT_TURNS: int = 10

    # Consolidation
    MIN_TURNS_FOR_CONSOLIDATION: int = 5
    CONSOLIDATION_IDLE_TIMEOUT_MS: int = 60000
    MAX_BUFFER_SIZE: int = 100
    MAX_CONCURRENT_LLM_CALLS: int = 5

    # LLM / OpenAI
    GENERATION_MODEL: str = "gpt-4o-mini"
    LLM_MODEL: str = "gpt-4o-mini"
    EXTRACTION_MODEL: str = "gpt-4o-mini"

    # Tavily (web search)
    TAVILY_API_KEY: str = ""

    # Text truncation limits
    TEXT_TRUNCATION_LIMIT: int = 4000
    RERANK_TRUNCATION_LIMIT: int = 300
    TOPIC_TRUNCATION_LIMIT: int = 200
    QUERY_TRUNCATION_LIMIT: int = 500

    # Topic Detection
    TOPIC_DETECTION_ENABLED: bool = True
    TOPIC_DETECTION_MODEL: str = "gpt-4o-mini"
    TOPIC_DETECTION_CONTEXT_TURNS: int = 4

    # LLM Reranker
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "gpt-4o-mini"
    RERANKER_TOP_K: int = 5
    RERANKER_CANDIDATE_MULTIPLIER: int = 3

    # Memory Distillation
    DISTILLATION_ENABLED: bool = True
    DISTILLATION_MODEL: str = "gpt-4o-mini"
    DISTILLATION_MIN_EPISODES: int = 5
    DISTILLATION_SIMILARITY_THRESHOLD: float = 0.7

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = CoreSettings()
