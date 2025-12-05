"""
Janus 3.5 Configuration

Centralized settings using pydantic-settings for environment variable support.
Simplified for Janus 3.5 - Redis Streams + PostgreSQL only.
"""

from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ============================================================
    # Environment
    # ============================================================
    ENVIRONMENT: str = "development"

    # ============================================================
    # Database (PostgreSQL + pgvector)
    # ============================================================
    DATABASE_URL: str = "postgresql://janus:janus_dev@localhost:5432/janus3_db"

    # ============================================================
    # Redis
    # ============================================================
    REDIS_URL: str = "redis://localhost:6379/0"

    # Redis Streams Configuration (STM - Short Term Memory)
    STM_STREAM_PREFIX: str = "stm"
    CORTEX_CONSUMER_GROUP: str = "cortex_workers"
    STM_MAX_LEN: int = 1000  # XTRIM MAXLEN for stream retention

    # Threshold Store (Redis)
    THRESHOLD_REDIS_TTL: int = 604800  # 7 days in seconds

    # ============================================================
    # Embeddings (OpenAI text-embedding-3-large)
    # ============================================================
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIM: int = 1536  # Configurable: 256, 512, 1024, 1536, 3072

    # ============================================================
    # Adaptive Threshold Configuration
    # ============================================================
    # Pre-calculated global average velocities for cold start
    # These represent typical semantic drift values from training data
    COLD_START_VELOCITIES: List[float] = [
        0.32, 0.28, 0.35, 0.41, 0.38, 0.29, 0.33, 0.40,
        0.36, 0.31, 0.42, 0.37, 0.34, 0.30, 0.39, 0.27,
        0.43, 0.35, 0.32, 0.38
    ]
    ADAPTIVE_THRESHOLD_WINDOW: int = 50
    ADAPTIVE_THRESHOLD_MULTIPLIER: float = 1.5  # std multiplier for boundary detection

    # ============================================================
    # Retrieval Configuration
    # ============================================================
    RETRIEVAL_TOP_K_ENTITIES: int = 10
    RETRIEVAL_TOP_K_EPISODES: int = 5
    STM_CONTEXT_TURNS: int = 10

    # ============================================================
    # Consolidation Configuration
    # ============================================================
    MIN_TURNS_FOR_CONSOLIDATION: int = 5
    CONSOLIDATION_IDLE_TIMEOUT_MS: int = 60000  # 60 seconds for crash recovery
    MAX_BUFFER_SIZE: int = 100  # Max turns buffered per session before forced consolidation
    MAX_CONCURRENT_CONSOLIDATIONS: int = 10
    MAX_CONCURRENT_LLM_CALLS: int = 5
    CONSOLIDATION_WORKERS: int = 4

    # ============================================================
    # LLM / OpenAI
    # ============================================================
    OPENAI_API_KEY: str = ""
    GENERATION_MODEL: str = "gpt-4o-mini"
    LLM_MODEL: str = "gpt-4o-mini"
    EXTRACTION_MODEL: str = "gpt-4o-mini"  # For unified extractor

    # ============================================================
    # Tavily (Web Search)
    # ============================================================
    TAVILY_API_KEY: str = ""
    TAVILY_ENABLED: bool = True  # Feature flag to enable/disable web search
    TAVILY_DEFAULT_SEARCH_DEPTH: str = "basic"  # "basic" or "advanced"
    TAVILY_MAX_RESULTS: int = 5
    TAVILY_MAX_TOOL_ITERATIONS: int = 3  # Max agentic loop iterations

    # ============================================================
    # Topic Detection (LLM-based boundary detection)
    # ============================================================
    TOPIC_DETECTION_ENABLED: bool = True
    TOPIC_DETECTION_MODEL: str = "gpt-4o-mini"
    TOPIC_DETECTION_CONTEXT_TURNS: int = 4  # Recent turns for context

    # ============================================================
    # LLM Reranker
    # ============================================================
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "gpt-4o-mini"
    RERANKER_TOP_K: int = 5
    RERANKER_CANDIDATE_MULTIPLIER: int = 3  # Fetch 3x candidates for reranking

    # ============================================================
    # Memory Distillation
    # ============================================================
    DISTILLATION_ENABLED: bool = True
    DISTILLATION_MODEL: str = "gpt-4o-mini"
    DISTILLATION_MIN_EPISODES: int = 5  # Min episodes to trigger distillation
    DISTILLATION_SIMILARITY_THRESHOLD: float = 0.7  # Cluster similarity

    # ============================================================
    # Observability
    # ============================================================
    # OpenTelemetry
    OTEL_ENABLED: bool = True
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = None  # e.g., "localhost:4317"
    OTEL_SERVICE_NAME: str = "janus3"
    OTEL_CONSOLE_EXPORT: bool = False  # Enable for debugging

    # Prometheus
    PROMETHEUS_ENABLED: bool = True
    PROMETHEUS_MULTIPROC_DIR: Optional[str] = None  # For multiprocess mode

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton instance
settings = Settings()
