"""
FastAPI Dependency Injection - Janus 3.5

Manages singletons and dependency injection for the API layer.

Simplified for Janus 3.5:
- Redis for STM (Redis Streams)
- PostgreSQL for LTM (episodes, entities, facts)
- Unified extractor for entity/fact extraction
- OpenTelemetry tracing

GPT-5.1 Upgrades:
- Topic detector for LLM-based boundary detection
- LLM reranker for relevance scoring
- Memory distiller for compression
"""

import logging
from functools import lru_cache
from typing import Dict, Optional
from uuid import UUID

import redis.asyncio as redis

from janus_core.config import settings
from janus_core.embedder import Embedder
from janus_core.unified_extractor import UnifiedExtractor
from janus_core.retrieval import HybridRetriever
from janus_core.threshold import AdaptiveThreshold
from janus_core.threshold_store import ThresholdStore
from janus_core.topic_detector import TopicDetector
from janus_core.reranker import LLMReranker
from janus_core.distillation import MemoryDistiller
from janus_core.db.connection import close_db_pool, get_db_pool, init_db_pool
from janus_core.db.stm import STMManager
from janus_core.llm.service import LLMService
from janus_app.services.tavily import TavilyService
from janus_app.services.agentic import AgenticChatHandler

logger = logging.getLogger(__name__)

# Global state
_redis_client: Optional[redis.Redis] = None
_threshold_cache: Dict[UUID, AdaptiveThreshold] = {}
_threshold_store: Optional[ThresholdStore] = None
_tavily_service: Optional[TavilyService] = None
_topic_detector: Optional[TopicDetector] = None
_reranker: Optional[LLMReranker] = None
_distiller: Optional[MemoryDistiller] = None


async def get_redis() -> redis.Redis:
    """
    Get async Redis client.

    Creates client if not exists.

    Returns:
        Async Redis client instance.
    """
    global _redis_client

    if _redis_client is None:
        _redis_client = redis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )

    return _redis_client


async def close_redis():
    """Close Redis client."""
    global _redis_client

    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None


async def get_threshold_store() -> ThresholdStore:
    """
    Get Redis-backed threshold store (singleton).

    Returns:
        ThresholdStore instance for atomic threshold updates.
    """
    global _threshold_store

    if _threshold_store is None:
        redis_client = await get_redis()
        _threshold_store = ThresholdStore(redis_client)

    return _threshold_store


@lru_cache()
def get_embedder() -> Embedder:
    """
    Get embedder service (singleton).

    Returns:
        Embedder instance.
    """
    return Embedder()


@lru_cache()
def get_unified_extractor() -> UnifiedExtractor:
    """
    Get unified extractor service (singleton).

    Returns:
        UnifiedExtractor instance for entity/fact/summary extraction.
    """
    return UnifiedExtractor()


@lru_cache()
def get_llm_service() -> LLMService:
    """
    Get LLM service (singleton).

    Returns:
        LLMService instance.
    """
    return LLMService()


def get_tavily_service() -> TavilyService:
    """
    Get Tavily service (singleton).

    Returns:
        TavilyService instance for web search.
    """
    global _tavily_service

    if _tavily_service is None:
        _tavily_service = TavilyService()

    return _tavily_service


def get_agentic_handler() -> AgenticChatHandler:
    """
    Get agentic chat handler.

    Creates AgenticChatHandler with LLM and Tavily services.

    Returns:
        AgenticChatHandler instance for tool-enabled chat.
    """
    llm = get_llm_service()
    tavily = get_tavily_service()
    return AgenticChatHandler(llm.client, tavily)


def get_topic_detector() -> TopicDetector:
    """
    Get topic detector (singleton).

    Returns:
        TopicDetector instance for LLM-based boundary detection.
    """
    global _topic_detector

    if _topic_detector is None:
        _topic_detector = TopicDetector()

    return _topic_detector


def get_reranker() -> LLMReranker:
    """
    Get LLM reranker (singleton).

    Returns:
        LLMReranker instance for relevance scoring.
    """
    global _reranker

    if _reranker is None:
        _reranker = LLMReranker()

    return _reranker


def get_distiller() -> MemoryDistiller:
    """
    Get memory distiller (singleton).

    Returns:
        MemoryDistiller instance for memory compression.
    """
    global _distiller

    if _distiller is None:
        _distiller = MemoryDistiller()

    return _distiller


async def get_retriever() -> HybridRetriever:
    """
    Get hybrid retriever with all dependencies.

    GPT-5.1: Injects reranker if enabled.

    Returns:
        HybridRetriever instance.
    """
    db_pool = await get_db_pool()
    redis_client = await get_redis()
    embedder = get_embedder()

    # Inject reranker if enabled
    reranker = get_reranker() if settings.RERANKER_ENABLED else None

    return HybridRetriever(db_pool, redis_client, embedder, reranker)


async def get_stm_manager() -> STMManager:
    """
    Get STM manager.

    Returns:
        STMManager instance.
    """
    redis_client = await get_redis()
    return STMManager(redis_client)


def get_threshold(session_id: UUID) -> AdaptiveThreshold:
    """
    Get or create adaptive threshold for a session.

    Args:
        session_id: Session UUID.

    Returns:
        AdaptiveThreshold instance for the session.
    """
    if session_id not in _threshold_cache:
        _threshold_cache[session_id] = AdaptiveThreshold(str(session_id))

    return _threshold_cache[session_id]


def clear_threshold(session_id: UUID):
    """
    Remove threshold from cache (on session delete).

    Args:
        session_id: Session UUID.
    """
    _threshold_cache.pop(session_id, None)


async def startup():
    """
    Application startup - initialize all services.
    """
    logger.info("Initializing services (Janus 3.5)...")

    # Initialize database pool
    await init_db_pool()
    logger.info("Database pool initialized")

    # Initialize Redis client
    await get_redis()
    logger.info("Redis client initialized")

    # Pre-load embedding model
    get_embedder()
    logger.info("Embedder loaded")

    # Initialize threshold store
    await get_threshold_store()
    logger.info("Threshold store initialized")

    # Initialize Tavily service (if API key configured)
    if settings.TAVILY_API_KEY:
        get_tavily_service()
        logger.info("Tavily service initialized")
    else:
        logger.warning("Tavily API key not configured - web search disabled")

    # Setup tracing
    from janus_app.infra.tracing import configure_tracing
    configure_tracing()
    logger.info("OpenTelemetry tracing initialized")

    logger.info("All services initialized")


async def shutdown():
    """
    Application shutdown - cleanup all services.
    """
    global _tavily_service

    logger.info("Shutting down services...")

    # Close Tavily HTTP client
    if _tavily_service is not None:
        await _tavily_service.close()
        _tavily_service = None
        logger.info("Tavily service closed")

    # Close database pool
    await close_db_pool()
    logger.info("Database pool closed")

    # Close Redis client
    await close_redis()
    logger.info("Redis client closed")

    # Clear threshold cache
    _threshold_cache.clear()

    # Shutdown tracing
    from janus3.infra.tracing import shutdown_tracing
    shutdown_tracing()

    logger.info("All services shut down")
