"""
Fact Repository - Janus 3.5

Simplified facts with upsert (no versioning).
Uses ON CONFLICT DO UPDATE for simple overwrite semantics.
"""

from typing import List, Optional
from uuid import UUID, uuid4
from datetime import datetime

import asyncpg

from core.models import Fact, Entity


class FactRepository:
    """
    Repository for fact operations.

    Janus 3.5: Simple upsert with ON CONFLICT DO UPDATE.
    No versioning - facts are overwritten when updated.
    """

    def __init__(self, pool: asyncpg.Pool):
        """
        Initialize repository.

        Args:
            pool: asyncpg connection pool.
        """
        self.pool = pool

    async def upsert_fact(
        self,
        session_id: UUID,
        subject_entity_id: UUID,
        predicate: str,
        object_value: str,
        confidence: float = 0.9,
        source_episode_id: Optional[UUID] = None
    ) -> Fact:
        """
        Upsert a fact with simple overwrite semantics.

        Uses ON CONFLICT DO UPDATE on (session_id, subject_entity_id, predicate).
        New value overwrites old value - no version history.

        Args:
            session_id: Session UUID.
            subject_entity_id: Entity UUID for the subject.
            predicate: Relationship type.
            object_value: The fact value.
            confidence: Confidence score (0.0-1.0).
            source_episode_id: Optional source episode.

        Returns:
            Created or updated Fact object.
        """
        async with self.pool.acquire() as conn:
            fact_id = uuid4()
            row = await conn.fetchrow("""
                INSERT INTO facts
                (id, session_id, subject_entity_id, predicate, object,
                 confidence, source_episode_id, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                ON CONFLICT (session_id, subject_entity_id, predicate)
                DO UPDATE SET
                    object = EXCLUDED.object,
                    confidence = EXCLUDED.confidence,
                    source_episode_id = EXCLUDED.source_episode_id,
                    updated_at = NOW()
                RETURNING id, session_id, subject_entity_id, predicate, object,
                          confidence, source_episode_id, created_at, updated_at
            """, fact_id, session_id, subject_entity_id, predicate,
                 object_value, confidence, source_episode_id)

            return Fact(
                id=row["id"],
                session_id=row["session_id"],
                subject_entity_id=row["subject_entity_id"],
                predicate=row["predicate"],
                object=row["object"],
                confidence=row["confidence"],
                source_episode_id=row["source_episode_id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"]
            )

    async def get_facts_for_entities(
        self,
        session_id: UUID,
        entity_ids: List[UUID]
    ) -> List[Fact]:
        """
        Get facts for a list of entities.

        Args:
            session_id: Session UUID.
            entity_ids: List of entity UUIDs.

        Returns:
            List of Fact objects.
        """
        if not entity_ids:
            return []

        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, session_id, subject_entity_id, predicate, object,
                       confidence, source_episode_id, created_at, updated_at
                FROM facts
                WHERE session_id = $1
                  AND subject_entity_id = ANY($2)
                ORDER BY created_at DESC
            """, session_id, entity_ids)

            return [
                Fact(
                    id=row["id"],
                    session_id=row["session_id"],
                    subject_entity_id=row["subject_entity_id"],
                    predicate=row["predicate"],
                    object=row["object"],
                    confidence=row["confidence"],
                    source_episode_id=row["source_episode_id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"]
                )
                for row in rows
            ]

    async def get_all_facts(
        self,
        session_id: UUID,
        limit: int = 100
    ) -> List[Fact]:
        """
        Get all facts for a session.

        Args:
            session_id: Session UUID.
            limit: Maximum results.

        Returns:
            List of Fact objects.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, session_id, subject_entity_id, predicate, object,
                       confidence, source_episode_id, created_at, updated_at
                FROM facts
                WHERE session_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, session_id, limit)

            return [
                Fact(
                    id=row["id"],
                    session_id=row["session_id"],
                    subject_entity_id=row["subject_entity_id"],
                    predicate=row["predicate"],
                    object=row["object"],
                    confidence=row["confidence"],
                    source_episode_id=row["source_episode_id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"]
                )
                for row in rows
            ]

    async def delete_fact(
        self,
        fact_id: UUID
    ) -> bool:
        """
        Delete a fact by ID.

        Args:
            fact_id: Fact UUID.

        Returns:
            True if fact was deleted.
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM facts WHERE id = $1
            """, fact_id)
            return "DELETE 1" in result

    async def count_by_session(self, session_id: UUID) -> int:
        """Get fact count for a session."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval("""
                SELECT COUNT(*) FROM facts WHERE session_id = $1
            """, session_id)

    async def delete_by_session(self, session_id: UUID) -> int:
        """Delete all facts for a session. Returns count deleted."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM facts WHERE session_id = $1
            """, session_id)
            return int(result.split()[-1])


class EntityRepository:
    """
    Repository for entity operations.
    """

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def get_or_create(
        self,
        session_id: UUID,
        name: str,
        entity_type: str,
        embedding: List[float]
    ) -> Entity:
        """
        Get existing entity or create new one.

        Uses normalized_name for deduplication.

        Args:
            session_id: Session UUID.
            name: Entity name.
            entity_type: Entity type (person, location, etc.).
            embedding: Entity embedding vector.

        Returns:
            Entity object.
        """
        normalized_name = name.lower().strip()

        async with self.pool.acquire() as conn:
            # Try to find existing entity
            row = await conn.fetchrow("""
                SELECT id, session_id, name, normalized_name, entity_type, embedding, created_at
                FROM entities
                WHERE session_id = $1 AND normalized_name = $2
            """, session_id, normalized_name)

            if row:
                return Entity(
                    id=row["id"],
                    session_id=row["session_id"],
                    name=row["name"],
                    normalized_name=row["normalized_name"],
                    entity_type=row["entity_type"],
                    embedding=list(row["embedding"]),
                    created_at=row["created_at"]
                )

            # Create new entity
            entity_id = uuid4()
            await conn.execute("""
                INSERT INTO entities
                (id, session_id, name, normalized_name, entity_type, embedding)
                VALUES ($1, $2, $3, $4, $5, $6::vector)
            """, entity_id, session_id, name, normalized_name, entity_type, embedding)

            return Entity(
                id=entity_id,
                session_id=session_id,
                name=name,
                normalized_name=normalized_name,
                entity_type=entity_type,
                embedding=embedding,
                created_at=datetime.utcnow()
            )

    async def search_by_vector(
        self,
        session_id: UUID,
        query_vector: List[float],
        limit: int = 10,
        min_similarity: float = 0.4
    ) -> List[tuple[Entity, float]]:
        """
        Search entities by vector similarity.

        Args:
            session_id: Session UUID.
            query_vector: Query embedding.
            limit: Maximum results.
            min_similarity: Minimum cosine similarity.

        Returns:
            List of (Entity, similarity) tuples.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, session_id, name, normalized_name, entity_type, embedding, created_at,
                       1 - (embedding <=> $1::vector) as similarity
                FROM entities
                WHERE session_id = $2
                  AND 1 - (embedding <=> $1::vector) > $3
                ORDER BY embedding <=> $1::vector
                LIMIT $4
            """, query_vector, session_id, min_similarity, limit)

            return [
                (
                    Entity(
                        id=row["id"],
                        session_id=row["session_id"],
                        name=row["name"],
                        normalized_name=row["normalized_name"],
                        entity_type=row["entity_type"],
                        embedding=list(row["embedding"]),
                        created_at=row["created_at"]
                    ),
                    row["similarity"]
                )
                for row in rows
            ]

    async def get_by_ids(
        self,
        entity_ids: List[UUID]
    ) -> List[Entity]:
        """Get entities by IDs."""
        if not entity_ids:
            return []

        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, session_id, name, normalized_name, entity_type, embedding, created_at
                FROM entities
                WHERE id = ANY($1)
            """, entity_ids)

            return [
                Entity(
                    id=row["id"],
                    session_id=row["session_id"],
                    name=row["name"],
                    normalized_name=row["normalized_name"],
                    entity_type=row["entity_type"],
                    embedding=list(row["embedding"]),
                    created_at=row["created_at"]
                )
                for row in rows
            ]

    async def count_by_session(self, session_id: UUID) -> int:
        """Get entity count for a session."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval("""
                SELECT COUNT(*) FROM entities WHERE session_id = $1
            """, session_id)
