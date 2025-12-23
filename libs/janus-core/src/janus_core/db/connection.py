"""
Database Connection Pool Management

Uses asyncpg for async PostgreSQL connections with pgvector support.
"""

import asyncpg
from typing import Optional
from pgvector.asyncpg import register_vector

from janus_core.config import settings

# Global connection pool
_pool: Optional[asyncpg.Pool] = None


async def init_db_pool() -> asyncpg.Pool:
    """
    Initialize the database connection pool.

    Returns:
        asyncpg.Pool: The connection pool instance.
    """
    global _pool

    if _pool is not None:
        return _pool

    _pool = await asyncpg.create_pool(
        settings.DATABASE_URL,
        min_size=5,
        max_size=20,
        command_timeout=60,
        init=_init_connection  # Register pgvector on each connection
    )

    return _pool


async def _init_connection(conn: asyncpg.Connection):
    """Initialize each connection with pgvector support."""
    await register_vector(conn)


async def close_db_pool():
    """Close the database connection pool."""
    global _pool

    if _pool is not None:
        await _pool.close()
        _pool = None


async def get_db_pool() -> asyncpg.Pool:
    """
    Get the database connection pool.

    Raises:
        RuntimeError: If pool is not initialized.

    Returns:
        asyncpg.Pool: The connection pool instance.
    """
    if _pool is None:
        raise RuntimeError(
            "Database pool not initialized. Call init_db_pool() first."
        )
    return _pool
