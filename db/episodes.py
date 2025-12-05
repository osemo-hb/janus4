"""
Episode Repository

Handles CRUD operations for episodes and episode_vectors tables.
Implements multi-vector storage with proper HNSW indexing.
"""

from typing import List, Optional, Tuple
from uuid import UUID, uuid4
from datetime import datetime

import asyncpg

from core.models import Episode, EpisodeVector, VectorType


class EpisodeRepository:
    """
    Repository for episode operations.

    Manages episodes (LTM) with multi-vector storage.
    Uses child table pattern for proper HNSW indexing.
    """

    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize repository.

        Args:
            pool: asyncpg connection pool.
        """
        self.pool = pool

    async def create_episode(
        self,
        session_id: UUID,
        summary: str,
        turn_start: int,
        turn_end: int,
        summary_vector: List[float],
        centroid_vector: Optional[List[float]] = None,
        user_turn_vectors: Optional[List[Tuple[int, List[float]]]] = None,
        compressed_state: Optional[str] = None,
        topic_label: Optional[str] = None
    ) -> Episode:
        """
        Create episode with vectors in a transaction.

        Args:
            session_id: Session UUID.
            summary: Episode summary text.
            turn_start: First turn index in episode.
            turn_end: Last turn index in episode.
            summary_vector: Embedding of the summary.
            centroid_vector: Optional centroid of user turn vectors.
            user_turn_vectors: Optional list of (turn_index, vector) tuples.
            compressed_state: Optional 20-40 word abstraction (GPT-5.1).
            topic_label: Optional 2-3 word topic label (GPT-5.1).

        Returns:
            Created Episode object.
        """
        episode_id = uuid4()
        vectors = []

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Insert episode with optional compressed_state and topic_label
                await conn.execute("""
                    INSERT INTO episodes (id, session_id, summary, turn_start, turn_end,
                                          compressed_state, topic_label)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, episode_id, session_id, summary, turn_start, turn_end,
                    compressed_state, topic_label)

                # Insert summary vector
                summary_vec_id = uuid4()
                await conn.execute("""
                    INSERT INTO episode_vectors (id, episode_id, vector_type, embedding)
                    VALUES ($1, $2, $3, $4::vector)
                """, summary_vec_id, episode_id, VectorType.SUMMARY.value, summary_vector)

                vectors.append(EpisodeVector(
                    id=summary_vec_id,
                    episode_id=episode_id,
                    vector_type=VectorType.SUMMARY,
                    embedding=summary_vector
                ))

                # Insert centroid vector if provided
                if centroid_vector:
                    centroid_vec_id = uuid4()
                    await conn.execute("""
                        INSERT INTO episode_vectors (id, episode_id, vector_type, embedding)
                        VALUES ($1, $2, $3, $4::vector)
                    """, centroid_vec_id, episode_id, VectorType.CENTROID.value, centroid_vector)

                    vectors.append(EpisodeVector(
                        id=centroid_vec_id,
                        episode_id=episode_id,
                        vector_type=VectorType.CENTROID,
                        embedding=centroid_vector
                    ))

                # Insert user turn vectors if provided
                if user_turn_vectors:
                    for turn_index, vector in user_turn_vectors:
                        turn_vec_id = uuid4()
                        await conn.execute("""
                            INSERT INTO episode_vectors
                            (id, episode_id, vector_type, turn_index, embedding)
                            VALUES ($1, $2, $3, $4, $5::vector)
                        """, turn_vec_id, episode_id, VectorType.USER_TURN.value,
                             turn_index, vector)

                        vectors.append(EpisodeVector(
                            id=turn_vec_id,
                            episode_id=episode_id,
                            vector_type=VectorType.USER_TURN,
                            turn_index=turn_index,
                            embedding=vector
                        ))

        return Episode(
            id=episode_id,
            session_id=session_id,
            summary=summary,
            turn_start=turn_start,
            turn_end=turn_end,
            created_at=datetime.utcnow(),
            vectors=vectors
        )

    async def search_by_summary(
        self,
        session_id: UUID,
        query_vector: List[float],
        limit: int = 5,
        min_similarity: float = 0.4
    ) -> List[Tuple[Episode, float]]:
        """
        Search episodes by summary vector similarity.

        Uses the idx_vec_summary partial index for O(log N) search.

        Args:
            session_id: Session UUID.
            query_vector: Query embedding.
            limit: Maximum results.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            List of (Episode, similarity) tuples.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    e.id, e.session_id, e.summary, e.turn_start, e.turn_end, e.created_at,
                    1 - (ev.embedding <=> $1::vector) as similarity
                FROM episodes e
                JOIN episode_vectors ev ON e.id = ev.episode_id
                WHERE e.session_id = $2
                  AND ev.vector_type = 'summary'
                  AND 1 - (ev.embedding <=> $1::vector) > $3
                ORDER BY ev.embedding <=> $1::vector
                LIMIT $4
            """, query_vector, session_id, min_similarity, limit)

            return [
                (
                    Episode(
                        id=row["id"],
                        session_id=row["session_id"],
                        summary=row["summary"],
                        turn_start=row["turn_start"],
                        turn_end=row["turn_end"],
                        created_at=row["created_at"],
                        vectors=[]
                    ),
                    row["similarity"]
                )
                for row in rows
            ]

    async def search_by_user_turn(
        self,
        session_id: UUID,
        query_vector: List[float],
        limit: int = 5,
        min_similarity: float = 0.4
    ) -> List[Tuple[Episode, float, int]]:
        """
        Search episodes by user turn vector similarity.

        Uses the idx_vec_turns partial index for granular retrieval.
        Useful for "that specific thing I mentioned" queries.

        Args:
            session_id: Session UUID.
            query_vector: Query embedding.
            limit: Maximum results.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            List of (Episode, similarity, turn_index) tuples.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    e.id, e.session_id, e.summary, e.turn_start, e.turn_end, e.created_at,
                    ev.turn_index,
                    1 - (ev.embedding <=> $1::vector) as similarity
                FROM episodes e
                JOIN episode_vectors ev ON e.id = ev.episode_id
                WHERE e.session_id = $2
                  AND ev.vector_type = 'user_turn'
                  AND 1 - (ev.embedding <=> $1::vector) > $3
                ORDER BY ev.embedding <=> $1::vector
                LIMIT $4
            """, query_vector, session_id, min_similarity, limit)

            return [
                (
                    Episode(
                        id=row["id"],
                        session_id=row["session_id"],
                        summary=row["summary"],
                        turn_start=row["turn_start"],
                        turn_end=row["turn_end"],
                        created_at=row["created_at"],
                        vectors=[]
                    ),
                    row["similarity"],
                    row["turn_index"]
                )
                for row in rows
            ]

    async def get_by_session(
        self,
        session_id: UUID,
        limit: int = 50
    ) -> List[Episode]:
        """
        Get all episodes for a session, ordered by creation time.

        Args:
            session_id: Session UUID.
            limit: Maximum results.

        Returns:
            List of Episode objects.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, session_id, summary, turn_start, turn_end, created_at
                FROM episodes
                WHERE session_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, session_id, limit)

            return [
                Episode(
                    id=row["id"],
                    session_id=row["session_id"],
                    summary=row["summary"],
                    turn_start=row["turn_start"],
                    turn_end=row["turn_end"],
                    created_at=row["created_at"],
                    vectors=[]
                )
                for row in rows
            ]

    async def get_by_id(self, episode_id: UUID) -> Optional[Episode]:
        """
        Get episode by ID with all vectors.

        Args:
            episode_id: Episode UUID.

        Returns:
            Episode object or None.
        """
        async with self.pool.acquire() as conn:
            # Get episode
            row = await conn.fetchrow("""
                SELECT id, session_id, summary, turn_start, turn_end, created_at
                FROM episodes
                WHERE id = $1
            """, episode_id)

            if not row:
                return None

            # Get vectors
            vector_rows = await conn.fetch("""
                SELECT id, episode_id, vector_type, turn_index, embedding
                FROM episode_vectors
                WHERE episode_id = $1
            """, episode_id)

            vectors = [
                EpisodeVector(
                    id=vr["id"],
                    episode_id=vr["episode_id"],
                    vector_type=VectorType(vr["vector_type"]),
                    turn_index=vr["turn_index"],
                    embedding=list(vr["embedding"])
                )
                for vr in vector_rows
            ]

            return Episode(
                id=row["id"],
                session_id=row["session_id"],
                summary=row["summary"],
                turn_start=row["turn_start"],
                turn_end=row["turn_end"],
                created_at=row["created_at"],
                vectors=vectors
            )

    async def count_by_session(self, session_id: UUID) -> int:
        """Get episode count for a session."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval("""
                SELECT COUNT(*) FROM episodes WHERE session_id = $1
            """, session_id)

    async def delete_by_session(self, session_id: UUID) -> int:
        """Delete all episodes for a session. Returns count deleted."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM episodes WHERE session_id = $1
            """, session_id)
            # Extract count from "DELETE N"
            return int(result.split()[-1])

    async def get_by_topic(
        self,
        session_id: UUID,
        topic_label: str,
        limit: int = 20
    ) -> List[Episode]:
        """
        Get episodes with a specific topic label (GPT-5.1).

        Args:
            session_id: Session UUID.
            topic_label: Topic label to filter by.
            limit: Maximum results.

        Returns:
            List of Episode objects with matching topic.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, session_id, summary, turn_start, turn_end, created_at,
                       compressed_state, topic_label
                FROM episodes
                WHERE session_id = $1 AND topic_label = $2
                ORDER BY created_at DESC
                LIMIT $3
            """, session_id, topic_label, limit)

            return [
                Episode(
                    id=row["id"],
                    session_id=row["session_id"],
                    summary=row["summary"],
                    turn_start=row["turn_start"],
                    turn_end=row["turn_end"],
                    created_at=row["created_at"],
                    vectors=[],
                    compressed_state=row.get("compressed_state"),
                    topic_label=row.get("topic_label")
                )
                for row in rows
            ]
