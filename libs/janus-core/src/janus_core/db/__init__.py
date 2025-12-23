"""
Database layer for Janus Core.

Provides repository pattern implementations for PostgreSQL (with pgvector)
and Redis (Streams for STM).
"""

from janus_core.db.connection import init_db_pool, get_db_pool, close_db_pool
from janus_core.db.stm import STMManager
from janus_core.db.episodes import EpisodeRepository
from janus_core.db.facts import EntityRepository, FactRepository
from janus_core.db.distilled import DistilledMemoryRepository

__all__ = [
    "init_db_pool",
    "get_db_pool",
    "close_db_pool",
    "STMManager",
    "EpisodeRepository",
    "EntityRepository",
    "FactRepository",
    "DistilledMemoryRepository",
]
