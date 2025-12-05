"""
Distilled Memory Repository for Janus 3.5

PostgreSQL repository for storing and searching distilled memories.
"""

import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID

import asyncpg

from janus3.core.distillation import DistilledMemory
from janus3.config import settings

logger = logging.getLogger(__name__)


class DistilledMemoryRepository:
    """
    Repository for distilled memory storage and retrieval.

    Provides CRUD operations and vector search for distilled memories.
    """

    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize repository with database pool.

        Args:
            pool: asyncpg connection pool.
        """
        self.pool = pool

    async def store(self, memory: DistilledMemory) -> UUID:
        """
        Store a distilled memory.

        Args:
            memory: DistilledMemory to store.

        Returns:
            UUID of the stored memory.
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO distilled_memories (
                    id, session_id, source_episode_ids, source_fact_ids,
                    compressed_content, topic_cluster,
                    time_range_start, time_range_end,
                    embedding, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                memory.id,
                memory.session_id,
                memory.source_episode_ids,
                memory.source_fact_ids,
                memory.compressed_content,
                memory.topic_cluster,
                memory.time_range[0],
                memory.time_range[1],
                str(memory.embedding),  # pgvector accepts string format
                memory.created_at
            )

        logger.debug(f"Stored distilled memory {memory.id}")
        return memory.id

    async def get_by_id(self, memory_id: UUID) -> Optional[DistilledMemory]:
        """
        Get a distilled memory by ID.

        Args:
            memory_id: UUID of the memory.

        Returns:
            DistilledMemory or None if not found.
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, session_id, source_episode_ids, source_fact_ids,
                       compressed_content, topic_cluster,
                       time_range_start, time_range_end,
                       embedding, created_at
                FROM distilled_memories
                WHERE id = $1
                """,
                memory_id
            )

        if not row:
            return None

        return self._row_to_memory(row)

    async def get_by_session(
        self,
        session_id: UUID,
        limit: int = 20
    ) -> List[DistilledMemory]:
        """
        Get distilled memories for a session.

        Args:
            session_id: Session UUID.
            limit: Maximum number of memories to return.

        Returns:
            List of DistilledMemory objects.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, session_id, source_episode_ids, source_fact_ids,
                       compressed_content, topic_cluster,
                       time_range_start, time_range_end,
                       embedding, created_at
                FROM distilled_memories
                WHERE session_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                session_id,
                limit
            )

        return [self._row_to_memory(row) for row in rows]

    async def search_similar(
        self,
        query_vec: List[float],
        session_id: UUID,
        limit: int = 5,
        min_similarity: float = 0.5
    ) -> List[tuple]:
        """
        Vector similarity search on distilled memories.

        Args:
            query_vec: Query embedding vector.
            session_id: Session UUID to search within.
            limit: Maximum number of results.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            List of (DistilledMemory, similarity_score) tuples.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, session_id, source_episode_ids, source_fact_ids,
                       compressed_content, topic_cluster,
                       time_range_start, time_range_end,
                       embedding, created_at,
                       1 - (embedding <=> $1::vector) as similarity
                FROM distilled_memories
                WHERE session_id = $2
                  AND 1 - (embedding <=> $1::vector) >= $3
                ORDER BY embedding <=> $1::vector
                LIMIT $4
                """,
                str(query_vec),
                session_id,
                min_similarity,
                limit
            )

        results = []
        for row in rows:
            memory = self._row_to_memory(row)
            similarity = row['similarity']
            results.append((memory, similarity))

        return results

    async def get_by_topic(
        self,
        session_id: UUID,
        topic_cluster: str,
        limit: int = 10
    ) -> List[DistilledMemory]:
        """
        Get distilled memories by topic cluster.

        Args:
            session_id: Session UUID.
            topic_cluster: Topic cluster label.
            limit: Maximum number of results.

        Returns:
            List of DistilledMemory objects.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, session_id, source_episode_ids, source_fact_ids,
                       compressed_content, topic_cluster,
                       time_range_start, time_range_end,
                       embedding, created_at
                FROM distilled_memories
                WHERE session_id = $1
                  AND topic_cluster = $2
                ORDER BY created_at DESC
                LIMIT $3
                """,
                session_id,
                topic_cluster,
                limit
            )

        return [self._row_to_memory(row) for row in rows]

    async def delete(self, memory_id: UUID) -> bool:
        """
        Delete a distilled memory.

        Args:
            memory_id: UUID of the memory to delete.

        Returns:
            True if deleted, False if not found.
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM distilled_memories WHERE id = $1",
                memory_id
            )

        return "DELETE 1" in result

    async def delete_by_session(self, session_id: UUID) -> int:
        """
        Delete all distilled memories for a session.

        Args:
            session_id: Session UUID.

        Returns:
            Number of deleted memories.
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM distilled_memories WHERE session_id = $1",
                session_id
            )

        # Parse "DELETE N" result
        count = int(result.split()[-1]) if result else 0
        logger.debug(f"Deleted {count} distilled memories for session {session_id}")
        return count

    async def count_by_session(self, session_id: UUID) -> int:
        """
        Count distilled memories for a session.

        Args:
            session_id: Session UUID.

        Returns:
            Number of distilled memories.
        """
        async with self.pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM distilled_memories WHERE session_id = $1",
                session_id
            )

        return count or 0

    def _row_to_memory(self, row: asyncpg.Record) -> DistilledMemory:
        """Convert database row to DistilledMemory object."""
        # Parse embedding from string/list
        embedding = row['embedding']
        if isinstance(embedding, str):
            # Parse "[1.0, 2.0, ...]" format
            embedding = [float(x) for x in embedding.strip('[]').split(',')]

        return DistilledMemory(
            id=row['id'],
            session_id=row['session_id'],
            source_episode_ids=list(row['source_episode_ids'] or []),
            source_fact_ids=list(row['source_fact_ids'] or []),
            compressed_content=row['compressed_content'],
            topic_cluster=row['topic_cluster'] or '',
            time_range=(
                row['time_range_start'] or datetime.utcnow(),
                row['time_range_end'] or datetime.utcnow()
            ),
            embedding=embedding,
            created_at=row['created_at']
        )
