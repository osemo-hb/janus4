"""
FastAPI Dependency Injection

Manages singletons and dependency injection for the API layer.
"""

from functools import lru_cache
from typing import Optional, Dict
from uuid import UUID

import asyncpg
import redis.asyncio as redis

from config import settings
from core.embedder import Embedder
from core.entity_linker import EntityLinker
from core.retrieval import HybridRetriever
from core.threshold import AdaptiveThreshold
from db.connection import init_db_pool, get_db_pool, close_db_pool
from db.stm import STMManager
from services.llm_service import LLMService

# Global state
_redis_client: Optional[redis.Redis] = None
_threshold_cache: Dict[UUID, AdaptiveThreshold] = {}


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
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )

    return _redis_client


async def close_redis():
    """Close Redis client."""
    global _redis_client

    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None


@lru_cache()
def get_embedder() -> Embedder:
    """
    Get embedder service (singleton).

    Returns:
        Embedder instance.
    """
    return Embedder()


@lru_cache()
def get_entity_linker() -> EntityLinker:
    """
    Get entity linker service (singleton).

    Returns:
        EntityLinker instance.
    """
    return EntityLinker()


@lru_cache()
def get_llm_service() -> LLMService:
    """
    Get LLM service (singleton).

    Returns:
        LLMService instance.
    """
    return LLMService()


async def get_retriever() -> HybridRetriever:
    """
    Get hybrid retriever with all dependencies.

    Returns:
        HybridRetriever instance.
    """
    db_pool = await get_db_pool()
    redis_client = await get_redis()
    embedder = get_embedder()

    return HybridRetriever(db_pool, redis_client, embedder)


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
    print("Initializing services...")

    # Initialize database pool
    await init_db_pool()
    print("Database pool initialized.")

    # Initialize Redis client
    await get_redis()
    print("Redis client initialized.")

    # Pre-load embedding model (optional, can be lazy)
    get_embedder()
    print("Embedder loaded.")

    print("All services initialized.")


async def shutdown():
    """
    Application shutdown - cleanup all services.
    """
    print("Shutting down services...")

    # Close database pool
    await close_db_pool()
    print("Database pool closed.")

    # Close Redis client
    await close_redis()
    print("Redis client closed.")

    # Clear threshold cache
    _threshold_cache.clear()

    print("All services shut down.")
