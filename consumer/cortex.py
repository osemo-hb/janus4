"""
Cortex Consumer

Native asyncio Redis Streams consumer for memory consolidation.
NO RQ - Redis Streams handles job queuing, retries, and crash recovery natively.

Fix 5: Uses XREADGROUP/XACK/XAUTOCLAIM pattern.
"""

import asyncio
import json
import logging
import os
import signal
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from uuid import UUID
import numpy as np

import redis.asyncio as redis
import asyncpg

from config import settings
from core.embedder import Embedder
from core.entity_linker import EntityLinker
from core.models import Turn, ConsolidationResult, ExtractedFact
from db.connection import init_db_pool, close_db_pool, get_db_pool
from db.stm import STMManager
from db.episodes import EpisodeRepository
from db.facts import FactRepository, EntityRepository
from services.llm_service import LLMService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cortex")


class CortexConsumer:
    """
    Native Redis Streams consumer for memory consolidation.

    Fix 5: NO RQ - Uses native asyncio loop with XREADGROUP.
    - Polls streams with blocking read (reduces CPU spin)
    - Acknowledges messages with XACK after successful processing
    - Recovers abandoned messages with XAUTOCLAIM

    The consumer monitors multiple session streams and consolidates
    memory when triggered by topic boundaries or periodic checks.
    """

    def __init__(self):
        """Initialize the Cortex consumer."""
        self.consumer_name = f"consumer-{os.getpid()}"
        self.consumer_group = settings.CORTEX_CONSUMER_GROUP

        # State
        self.running = True
        self.redis: Optional[redis.Redis] = None
        self.db_pool: Optional[asyncpg.Pool] = None

        # Services (initialized on run)
        self.embedder: Optional[Embedder] = None
        self.entity_linker: Optional[EntityLinker] = None
        self.llm: Optional[LLMService] = None

        # Track active sessions
        self.active_sessions: Set[UUID] = set()

    async def initialize(self):
        """Initialize all services and connections."""
        logger.info(f"Initializing Cortex Consumer [{self.consumer_name}]")

        # Initialize database
        self.db_pool = await init_db_pool()
        logger.info("Database pool initialized")

        # Initialize Redis
        self.redis = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        logger.info("Redis client initialized")

        # Initialize ML services
        self.embedder = Embedder()
        logger.info("Embedder loaded")

        self.entity_linker = EntityLinker()
        # Note: GLiNER loads lazily on first use

        self.llm = LLMService()
        logger.info("LLM service initialized")

        logger.info("Cortex Consumer initialization complete")

    async def cleanup(self):
        """Cleanup resources on shutdown."""
        logger.info("Cleaning up Cortex Consumer...")

        if self.redis:
            await self.redis.close()

        if self.db_pool:
            await close_db_pool()

        logger.info("Cortex Consumer cleanup complete")

    async def run(self):
        """
        Main consumer loop.

        1. Poll for new sessions with activity
        2. Check each session's STM for consolidation triggers
        3. Process and consolidate when needed
        4. Recover abandoned work periodically
        """
        await self.initialize()

        logger.info(f"Cortex Consumer [{self.consumer_name}] started")

        try:
            while self.running:
                try:
                    # 1. Discover active sessions
                    await self._discover_sessions()

                    # 2. Process each active session
                    for session_id in list(self.active_sessions):
                        await self._process_session(session_id)

                    # 3. Check for abandoned work (crash recovery)
                    await self._recover_abandoned()

                    # 4. Small sleep to prevent tight loop
                    await asyncio.sleep(1)

                except Exception as e:
                    logger.error(f"Consumer loop error: {e}", exc_info=True)
                    await asyncio.sleep(5)  # Back off on error

        except asyncio.CancelledError:
            logger.info("Consumer received cancellation")
        finally:
            await self.cleanup()

    async def _discover_sessions(self):
        """
        Discover sessions with recent activity.

        Queries database for sessions with activity in last 24 hours.
        """
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id FROM sessions
                WHERE last_activity > NOW() - INTERVAL '24 hours'
            """)

            self.active_sessions = {row["id"] for row in rows}

    async def _process_session(self, session_id: UUID):
        """
        Process a single session for consolidation.

        Checks if consolidation is needed and performs it.
        """
        stm = STMManager(self.redis)

        # Get STM length
        stm_length = await stm.get_stream_length(session_id)

        # Check if we have enough turns to consolidate
        if stm_length < settings.MIN_TURNS_FOR_CONSOLIDATION:
            return

        # Get all turns from STM
        turns = await stm.get_all_turns(session_id)

        if len(turns) < settings.MIN_TURNS_FOR_CONSOLIDATION:
            return

        # Check if we should consolidate (time-based or triggered)
        should_consolidate = await self._should_consolidate(session_id, turns)

        if should_consolidate:
            logger.info(f"Consolidating session {session_id} ({len(turns)} turns)")
            await self._consolidate(session_id, turns)

    async def _should_consolidate(
        self,
        session_id: UUID,
        turns: List[Turn]
    ) -> bool:
        """
        Determine if consolidation should occur.

        Triggers:
        - STM has more than threshold turns
        - Consolidation signal received (topic boundary)
        - Time since last consolidation exceeds threshold
        """
        # Check for consolidation signal via pub/sub
        # (In production, this would use a more sophisticated trigger mechanism)

        # For now, consolidate if we have enough turns
        return len(turns) >= settings.MIN_TURNS_FOR_CONSOLIDATION * 2

    async def _consolidate(
        self,
        session_id: UUID,
        turns: List[Turn]
    ) -> Optional[ConsolidationResult]:
        """
        Consolidate STM turns into an LTM episode.

        1. Generate summary using LLM
        2. Extract facts using LLM
        3. Create episode with vectors
        4. Create/update entities and facts
        5. Acknowledge processed turns
        """
        try:
            # 1. Format conversation for LLM
            text_block = "\n".join([
                f"{t.role}: {t.content}"
                for t in turns
            ])

            # 2. Extract summary and facts using LLM
            extraction = await self.llm.extract_facts(text_block)
            summary = extraction.get("summary", "Conversation segment")
            raw_facts = extraction.get("facts", [])

            # 3. Create summary embedding
            summary_vec = await self.embedder.encode(summary)

            # 4. Calculate centroid of user turn vectors
            user_vectors = [t.vector for t in turns if t.role == "user"]
            if user_vectors:
                centroid_vec = np.mean(user_vectors, axis=0).tolist()
            else:
                centroid_vec = None

            # 5. Prepare user turn vectors (first 3 for anchoring)
            user_turn_vectors = [
                (i, turns[i].vector)
                for i, t in enumerate(turns)
                if t.role == "user"
            ][:3]

            # 6. Calculate turn range
            turn_start = min(
                int(t.id.split("-")[0]) if "-" in t.id else 0
                for t in turns
            )
            turn_end = max(
                int(t.id.split("-")[0]) if "-" in t.id else 0
                for t in turns
            )

            # 7. Create episode
            episode_repo = EpisodeRepository(self.db_pool)
            episode = await episode_repo.create_episode(
                session_id=session_id,
                summary=summary,
                turn_start=turn_start,
                turn_end=turn_end,
                summary_vector=summary_vec,
                centroid_vector=centroid_vec,
                user_turn_vectors=user_turn_vectors
            )

            # 8. Process extracted facts
            entity_repo = EntityRepository(self.db_pool)
            fact_repo = FactRepository(self.db_pool)

            entities_created = 0
            facts_created = 0

            for raw_fact in raw_facts:
                subject = raw_fact.get("subject", "").strip()
                predicate = raw_fact.get("predicate", "").strip()
                obj = raw_fact.get("object", "").strip()
                confidence = raw_fact.get("confidence", 0.9)

                if not subject or not predicate or not obj:
                    continue

                # Get or create subject entity
                subject_vec = await self.embedder.encode(subject)
                entity = await entity_repo.get_or_create(
                    session_id=session_id,
                    name=subject,
                    entity_type="concept",  # Could infer from GLiNER
                    embedding=subject_vec
                )
                entities_created += 1

                # Create fact
                await fact_repo.upsert_fact(
                    session_id=session_id,
                    subject_entity_id=entity.id,
                    predicate=predicate,
                    object_value=obj,
                    confidence=confidence,
                    source_episode_id=episode.id
                )
                facts_created += 1

            # 9. Acknowledge processed turns
            stm = STMManager(self.redis)
            entry_ids = [t.id for t in turns]
            await stm.acknowledge(session_id, entry_ids)

            logger.info(
                f"Consolidated session {session_id}: "
                f"episode={episode.id}, "
                f"entities={entities_created}, "
                f"facts={facts_created}"
            )

            return ConsolidationResult(
                episode_id=episode.id,
                summary=summary,
                facts_extracted=facts_created,
                entities_created=entities_created,
                turns_processed=len(turns)
            )

        except Exception as e:
            logger.error(
                f"Consolidation failed for session {session_id}: {e}",
                exc_info=True
            )
            return None

    async def _recover_abandoned(self):
        """
        Recover messages abandoned by crashed consumers.

        Uses XAUTOCLAIM to claim messages that have been pending
        longer than the idle timeout.
        """
        stm = STMManager(self.redis)

        for session_id in self.active_sessions:
            try:
                claimed = await stm.claim_abandoned(
                    session_id=session_id,
                    consumer_name=self.consumer_name,
                    min_idle_ms=settings.CONSOLIDATION_IDLE_TIMEOUT_MS
                )

                if claimed:
                    logger.info(
                        f"Claimed {len(claimed)} abandoned entries "
                        f"for session {session_id}"
                    )
                    # Re-acknowledge claimed entries
                    # (they were already processed or we'll process them next cycle)
                    entry_ids = [entry_id for entry_id, _ in claimed]
                    await stm.acknowledge(session_id, entry_ids)

            except Exception as e:
                logger.warning(
                    f"Failed to recover abandoned for {session_id}: {e}"
                )

    def stop(self):
        """Signal the consumer to stop."""
        logger.info("Stop signal received")
        self.running = False


def create_consumer() -> CortexConsumer:
    """Factory function to create a consumer instance."""
    return CortexConsumer()
