"""
Fact Repository

Handles versioned facts in the knowledge graph.
Implements soft-delete versioning with is_current flag.
"""

from typing import List, Optional
from uuid import UUID, uuid4
from datetime import datetime

import asyncpg

from core.models import Fact, Entity


class FactRepository:
    """
    Repository for fact operations.

    Manages versioned facts with automatic version incrementing
    and soft-delete via is_current flag.
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
        Upsert a fact with versioning.

        If a fact exists with same subject+predicate:
        - Mark old fact as not current
        - Create new fact with incremented version

        Args:
            session_id: Session UUID.
            subject_entity_id: Entity UUID for the subject.
            predicate: Relationship type.
            object_value: The fact value.
            confidence: Confidence score (0.0-1.0).
            source_episode_id: Optional source episode.

        Returns:
            Created Fact object.
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # Mark existing current fact as superseded
                await conn.execute("""
                    UPDATE facts
                    SET is_current = FALSE
                    WHERE session_id = $1
                      AND subject_entity_id = $2
                      AND predicate = $3
                      AND is_current = TRUE
                """, session_id, subject_entity_id, predicate)

                # Get max version for this subject+predicate
                max_version = await conn.fetchval("""
                    SELECT COALESCE(MAX(version), 0)
                    FROM facts
                    WHERE session_id = $1
                      AND subject_entity_id = $2
                      AND predicate = $3
                """, session_id, subject_entity_id, predicate)

                # Insert new version
                fact_id = uuid4()
                new_version = max_version + 1

                await conn.execute("""
                    INSERT INTO facts
                    (id, session_id, subject_entity_id, predicate, object,
                     version, confidence, is_current, source_episode_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, TRUE, $8)
                """, fact_id, session_id, subject_entity_id, predicate,
                     object_value, new_version, confidence, source_episode_id)

                return Fact(
                    id=fact_id,
                    session_id=session_id,
                    subject_entity_id=subject_entity_id,
                    predicate=predicate,
                    object=object_value,
                    version=new_version,
                    confidence=confidence,
                    is_current=True,
                    source_episode_id=source_episode_id,
                    created_at=datetime.utcnow()
                )

    async def get_facts_for_entities(
        self,
        session_id: UUID,
        entity_ids: List[UUID],
        current_only: bool = True
    ) -> List[Fact]:
        """
        Get facts for a list of entities.

        Args:
            session_id: Session UUID.
            entity_ids: List of entity UUIDs.
            current_only: If True, only return current facts.

        Returns:
            List of Fact objects.
        """
        if not entity_ids:
            return []

        async with self.pool.acquire() as conn:
            if current_only:
                rows = await conn.fetch("""
                    SELECT id, session_id, subject_entity_id, predicate, object,
                           version, confidence, is_current, source_episode_id, created_at
                    FROM facts
                    WHERE session_id = $1
                      AND subject_entity_id = ANY($2)
                      AND is_current = TRUE
                    ORDER BY created_at DESC
                """, session_id, entity_ids)
            else:
                rows = await conn.fetch("""
                    SELECT id, session_id, subject_entity_id, predicate, object,
                           version, confidence, is_current, source_episode_id, created_at
                    FROM facts
                    WHERE session_id = $1
                      AND subject_entity_id = ANY($2)
                    ORDER BY version DESC
                """, session_id, entity_ids)

            return [
                Fact(
                    id=row["id"],
                    session_id=row["session_id"],
                    subject_entity_id=row["subject_entity_id"],
                    predicate=row["predicate"],
                    object=row["object"],
                    version=row["version"],
                    confidence=row["confidence"],
                    is_current=row["is_current"],
                    source_episode_id=row["source_episode_id"],
                    created_at=row["created_at"]
                )
                for row in rows
            ]

    async def get_all_current_facts(
        self,
        session_id: UUID,
        limit: int = 100
    ) -> List[Fact]:
        """
        Get all current facts for a session.

        Args:
            session_id: Session UUID.
            limit: Maximum results.

        Returns:
            List of current Fact objects.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, session_id, subject_entity_id, predicate, object,
                       version, confidence, is_current, source_episode_id, created_at
                FROM facts
                WHERE session_id = $1
                  AND is_current = TRUE
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
                    version=row["version"],
                    confidence=row["confidence"],
                    is_current=row["is_current"],
                    source_episode_id=row["source_episode_id"],
                    created_at=row["created_at"]
                )
                for row in rows
            ]

    async def retract_fact(
        self,
        fact_id: UUID
    ) -> bool:
        """
        Retract a fact (mark as not current).

        Args:
            fact_id: Fact UUID.

        Returns:
            True if fact was retracted.
        """
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE facts
                SET is_current = FALSE
                WHERE id = $1
            """, fact_id)
            return "UPDATE 1" in result

    async def get_fact_history(
        self,
        session_id: UUID,
        subject_entity_id: UUID,
        predicate: str
    ) -> List[Fact]:
        """
        Get version history for a specific fact.

        Args:
            session_id: Session UUID.
            subject_entity_id: Entity UUID.
            predicate: Fact predicate.

        Returns:
            List of Fact objects ordered by version descending.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, session_id, subject_entity_id, predicate, object,
                       version, confidence, is_current, source_episode_id, created_at
                FROM facts
                WHERE session_id = $1
                  AND subject_entity_id = $2
                  AND predicate = $3
                ORDER BY version DESC
            """, session_id, subject_entity_id, predicate)

            return [
                Fact(
                    id=row["id"],
                    session_id=row["session_id"],
                    subject_entity_id=row["subject_entity_id"],
                    predicate=row["predicate"],
                    object=row["object"],
                    version=row["version"],
                    confidence=row["confidence"],
                    is_current=row["is_current"],
                    source_episode_id=row["source_episode_id"],
                    created_at=row["created_at"]
                )
                for row in rows
            ]

    async def count_by_session(self, session_id: UUID, current_only: bool = True) -> int:
        """Get fact count for a session."""
        async with self.pool.acquire() as conn:
            if current_only:
                return await conn.fetchval("""
                    SELECT COUNT(*) FROM facts
                    WHERE session_id = $1 AND is_current = TRUE
                """, session_id)
            else:
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
