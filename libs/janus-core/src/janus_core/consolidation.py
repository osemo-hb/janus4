"""
Consolidation Service - Janus 3.5

Handles memory consolidation: converts buffered turns into episodes,
extracts entities/facts using unified LLM extractor, and persists to PostgreSQL.

This is the core of the Cortex pipeline, triggered by:
- Buffer overflow (too many turns in STM)
- Topic boundary signals (semantic shift detected)
- Periodic consolidation (time-based)

GPT-5.1 Upgrades:
- Compressed state and topic label for episodes
- Canonical predicates for facts
- Memory distillation for compression

Usage:
    service = await ConsolidationService.get_instance()
    result = await service.consolidate(session_id, turns, reason)
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np

from janus_core.config import settings
from janus_core.embedder import Embedder
from janus_core.unified_extractor import UnifiedExtractor, ExtractedEntity, ExtractedFact
from janus_core.distillation import MemoryDistiller
from janus_core.models import ConsolidationResult
from janus_core.db.connection import get_db_pool
from janus_core.db.episodes import EpisodeRepository
from janus_core.db.facts import EntityRepository, FactRepository
from janus_core.db.distilled import DistilledMemoryRepository
from janus_core.tracing import tracer

logger = logging.getLogger(__name__)


class ConsolidationService:
    """
    Memory consolidation service - Janus 3.5.

    Singleton pattern - use get_instance() to obtain the service.

    Simplified consolidation flow:
    1. Format turns into text block
    2. Unified extraction (entities + facts + summary in one LLM call)
    3. Generate embeddings for episode
    4. Create episode with vectors (PostgreSQL)
    5. Persist entities/facts (PostgreSQL with simple upsert)
    """

    _instance: Optional["ConsolidationService"] = None
    _initialized: bool = False

    def __new__(cls) -> "ConsolidationService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def initialize(self) -> None:
        """Initialize all dependencies."""
        if ConsolidationService._initialized:
            return

        logger.info("Initializing ConsolidationService (Janus 3.5 with GPT-5.1 upgrades)")

        # Initialize embedder
        self.embedder = Embedder()

        # Initialize unified extractor (replaces EntityLinker + HybridFactExtractor)
        self.extractor = UnifiedExtractor()

        # Initialize memory distiller
        self.distiller = MemoryDistiller()

        # Get database pool
        self.db_pool = await get_db_pool()

        # Rate limiting semaphore for LLM calls
        self.llm_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_LLM_CALLS)

        ConsolidationService._initialized = True
        logger.info("ConsolidationService initialized")

    @classmethod
    async def get_instance(cls) -> "ConsolidationService":
        """Get or create the singleton instance."""
        instance = cls()
        if not cls._initialized:
            await instance.initialize()
        return instance

    async def consolidate(
        self,
        session_id: UUID,
        turns: List[Dict[str, Any]],
        source_episode_reason: str = "buffer_overflow",
    ) -> Optional[ConsolidationResult]:
        """
        Consolidate turns into a long-term memory episode.

        Args:
            session_id: Session UUID
            turns: List of turn dicts with 'role', 'content', 'turn_index'
            source_episode_reason: Why consolidation was triggered

        Returns:
            ConsolidationResult on success, None on failure
        """
        if len(turns) < settings.MIN_TURNS_FOR_CONSOLIDATION:
            logger.debug(
                f"Not enough turns for consolidation: {len(turns)} < "
                f"{settings.MIN_TURNS_FOR_CONSOLIDATION}"
            )
            return None

        with tracer.start_as_current_span("consolidate") as span:
            span.set_attribute("session_id", str(session_id))
            span.set_attribute("turn_count", len(turns))
            span.set_attribute("reason", source_episode_reason)

            try:
                return await self._do_consolidate(
                    session_id, turns, source_episode_reason
                )
            except Exception as e:
                logger.error(
                    f"Consolidation failed for session {str(session_id)[:8]}: {e}",
                    exc_info=True,
                )
                raise

    async def _do_consolidate(
        self,
        session_id: UUID,
        turns: List[Dict[str, Any]],
        reason: str,
    ) -> ConsolidationResult:
        """Internal consolidation logic - Janus 3.5 with GPT-5.1 upgrades."""
        start_time = datetime.utcnow()

        # 1. Format conversation text
        with tracer.start_as_current_span("format_text"):
            text_block = self._format_turns(turns)
            turn_indices = [t.get("turn_index", 0) for t in turns]
            turn_start = min(turn_indices) if turn_indices else 0
            turn_end = max(turn_indices) if turn_indices else 0

        # 2. Unified extraction (entities + facts + summary + compressed state)
        with tracer.start_as_current_span("unified_extraction") as span:
            async with self.llm_semaphore:
                extraction = await self.extractor.extract(text_block)
            span.set_attribute("entity_count", len(extraction.entities))
            span.set_attribute("fact_count", len(extraction.facts))

            # Extract compressed state and topic label (GPT-5.1)
            compressed_state = None
            topic_label = None
            if extraction.episode_state:
                compressed_state = extraction.episode_state.compressed
                topic_label = extraction.episode_state.topic_label
                span.set_attribute("topic_label", topic_label or "unknown")

        # 3. Generate embeddings
        with tracer.start_as_current_span("generate_embeddings"):
            # Use extracted summary or generate fallback
            summary = extraction.summary or f"Conversation segment ({len(turns)} turns)"
            summary_vec = await self.embedder.encode(summary)

            # Calculate centroid of user turn vectors
            centroid_vec = await self._calculate_centroid(turns)

            # Get individual user turn vectors (first 3 for anchoring)
            user_turn_vectors = await self._get_user_turn_vectors(turns)

        # 4. Create episode in PostgreSQL (with compressed state + topic label)
        with tracer.start_as_current_span("create_episode") as span:
            episode_repo = EpisodeRepository(self.db_pool)

            episode = await episode_repo.create_episode(
                session_id=session_id,
                summary=summary,
                turn_start=turn_start,
                turn_end=turn_end,
                summary_vector=summary_vec,
                centroid_vector=centroid_vec,
                user_turn_vectors=user_turn_vectors,
                compressed_state=compressed_state,  # GPT-5.1
                topic_label=topic_label,  # GPT-5.1
            )
            span.set_attribute("episode_id", str(episode.id))

        # 5. Persist entities and facts to PostgreSQL (with canonical predicates)
        with tracer.start_as_current_span("persist_entities_facts"):
            entities_created, facts_created = await self._persist_to_postgres(
                session_id=session_id,
                episode_id=episode.id,
                entities=extraction.entities,
                facts=extraction.facts,
            )

        # 6. Check if distillation should be triggered (GPT-5.1)
        if settings.DISTILLATION_ENABLED and topic_label:
            await self._maybe_distill(session_id, topic_label)

        duration = (datetime.utcnow() - start_time).total_seconds()

        logger.info(
            f"Consolidated session {str(session_id)[:8]}: "
            f"episode={episode.id}, "
            f"entities={entities_created}, "
            f"facts={facts_created}, "
            f"topic={topic_label or 'unknown'}, "
            f"duration={duration:.2f}s"
        )

        return ConsolidationResult(
            episode_id=episode.id,
            summary=summary,
            facts_extracted=facts_created,
            entities_created=entities_created,
            turns_processed=len(turns),
        )

    def _format_turns(self, turns: List[Dict[str, Any]]) -> str:
        """Format turns into conversation text block."""
        return "\n".join([
            f"{t.get('role', 'unknown')}: {t.get('content', '')}"
            for t in turns
        ])

    async def _calculate_centroid(
        self, turns: List[Dict[str, Any]]
    ) -> Optional[List[float]]:
        """Calculate centroid of user turn vectors."""
        user_contents = [
            t.get("content", "")
            for t in turns
            if t.get("role") == "user" and t.get("content")
        ]

        if not user_contents:
            return None

        # Batch embed user turns
        vectors = await self.embedder.encode_batch(user_contents)

        if not vectors:
            return None

        # Calculate centroid
        centroid = np.mean(vectors, axis=0)
        return centroid.tolist()

    async def _get_user_turn_vectors(
        self, turns: List[Dict[str, Any]]
    ) -> List[Tuple[int, List[float]]]:
        """Get embeddings for first 3 user turns for granular anchoring."""
        user_turns = [
            (t.get("turn_index", i), t.get("content", ""))
            for i, t in enumerate(turns)
            if t.get("role") == "user" and t.get("content")
        ][:3]

        if not user_turns:
            return []

        result = []
        for turn_index, content in user_turns:
            vec = await self.embedder.encode(content)
            result.append((turn_index, vec))

        return result

    async def _persist_to_postgres(
        self,
        session_id: UUID,
        episode_id: UUID,
        entities: List[ExtractedEntity],
        facts: List[ExtractedFact],
    ) -> tuple[int, int]:
        """Persist entities and facts to PostgreSQL with simple upsert."""
        entity_repo = EntityRepository(self.db_pool)
        fact_repo = FactRepository(self.db_pool)

        entities_created = 0
        facts_created = 0

        # Create a mapping of entity names to their IDs
        entity_map: Dict[str, UUID] = {}

        # 1. Create entities
        for e in entities:
            # Generate embedding for entity
            embedding = await self.embedder.encode(e.name)

            entity = await entity_repo.get_or_create(
                session_id=session_id,
                name=e.name,
                entity_type=e.entity_type,
                embedding=embedding,
            )
            entity_map[e.name.lower().strip()] = entity.id
            entities_created += 1

        # 2. Create/upsert facts (with canonical predicates - GPT-5.1)
        for f in facts:
            # Find or create subject entity
            subject_key = f.subject.lower().strip()
            if subject_key not in entity_map:
                # Create entity for the subject if not already extracted
                embedding = await self.embedder.encode(f.subject)
                entity = await entity_repo.get_or_create(
                    session_id=session_id,
                    name=f.subject,
                    entity_type="concept",
                    embedding=embedding,
                )
                entity_map[subject_key] = entity.id
                entities_created += 1

            subject_entity_id = entity_map[subject_key]

            # Upsert fact (with canonical predicate and source span - GPT-5.1)
            await fact_repo.upsert_fact(
                session_id=session_id,
                subject_entity_id=subject_entity_id,
                predicate=f.predicate,
                object_value=f.object_value,
                confidence=0.9,  # Default confidence from unified extractor
                source_episode_id=episode_id,
                canonical_predicate=f.canonical_predicate,  # GPT-5.1
                source_span=f.source_span,  # GPT-5.1
            )
            facts_created += 1

        return entities_created, facts_created

    async def _maybe_distill(self, session_id: UUID, topic_label: str) -> None:
        """Check if distillation should be triggered for this topic cluster."""
        if not settings.DISTILLATION_ENABLED:
            return

        try:
            await self._check_and_distill(session_id, topic_label)
        except Exception as e:
            logger.warning(f"Distillation check failed: {e}")

    async def _check_and_distill(self, session_id: UUID, topic_label: str) -> None:
        """Check episode count and trigger distillation if threshold met."""
        with tracer.start_as_current_span("check_distillation"):
            episode_repo = EpisodeRepository(self.db_pool)
            episodes = await episode_repo.get_by_topic(
                session_id=session_id,
                topic_label=topic_label,
                limit=settings.DISTILLATION_MIN_EPISODES + 5
            )

            if len(episodes) < settings.DISTILLATION_MIN_EPISODES:
                return

            logger.info(
                f"Triggering distillation for topic '{topic_label}' "
                f"with {len(episodes)} episodes"
            )

            episode_dicts = [
                {
                    'id': e.id,
                    'summary': e.summary,
                    'compressed_state': getattr(e, 'compressed_state', None),
                    'created_at': e.created_at
                }
                for e in episodes
            ]

            distilled = await self.distiller.distill_episodes(episode_dicts, session_id)
            if not distilled:
                return

            distilled_repo = DistilledMemoryRepository(self.db_pool)
            await distilled_repo.store(distilled)
            logger.info(f"Created distilled memory {distilled.id} for topic '{topic_label}'")
