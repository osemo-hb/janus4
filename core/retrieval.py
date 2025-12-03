"""
Hybrid Retrieval System

Implements Fix 2: Parallelized hot path for context building.
Uses asyncio.gather to run entity and episode searches concurrently.
"""

import asyncio
import time
from typing import List, Tuple, Optional
from uuid import UUID
from dataclasses import dataclass

import asyncpg
import redis.asyncio as redis

from core.models import Entity, Episode, Fact, Turn, RetrievalStats
from core.embedder import Embedder
from db.stm import STMManager
from db.episodes import EpisodeRepository
from db.facts import FactRepository, EntityRepository
from config import settings


@dataclass
class RetrievalContext:
    """Aggregated context from all retrieval sources."""
    entities: List[Tuple[Entity, float]]  # (entity, similarity)
    episodes: List[Tuple[Episode, float]]  # (episode, similarity)
    facts: List[Fact]
    stm_turns: List[Turn]
    stats: RetrievalStats


class HybridRetriever:
    """
    Hybrid retrieval combining vector search and graph traversal.

    Fix 2: Parallelized hot path
    - Entity search and episode search run in PARALLEL
    - Graph traversal runs AFTER (depends on entity results)
    """

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        redis_client: redis.Redis,
        embedder: Embedder
    ):
        """
        Initialize retriever.

        Args:
            db_pool: asyncpg connection pool.
            redis_client: Async Redis client.
            embedder: Embedding service.
        """
        self.db_pool = db_pool
        self.redis = redis_client
        self.embedder = embedder

        # Initialize repositories
        self.stm = STMManager(redis_client)
        self.episode_repo = EpisodeRepository(db_pool)
        self.fact_repo = FactRepository(db_pool)
        self.entity_repo = EntityRepository(db_pool)

    async def build_context(
        self,
        query: str,
        query_vec: List[float],
        session_id: UUID
    ) -> RetrievalContext:
        """
        Build context for LLM response generation.

        Fix 2: Parallelized hot path
        1. Run entity search and episode search in PARALLEL
        2. Run graph traversal AFTER (depends on entities)
        3. Get STM turns

        Args:
            query: User query text.
            query_vec: Query embedding vector.
            session_id: Session UUID.

        Returns:
            RetrievalContext with all retrieved data.
        """
        start_time = time.time()

        # Step 1: PARALLEL - Entity and Episode search
        entities_task = self._search_entities(query_vec, session_id)
        episodes_task = self._search_episodes(query_vec, session_id)

        entities, episodes = await asyncio.gather(
            entities_task,
            episodes_task
        )

        # Step 2: SEQUENTIAL - Graph traversal (depends on entities)
        entity_ids = [e.id for e, _ in entities]
        facts = await self._search_graph(entity_ids, session_id)

        # Step 3: Get recent STM turns
        stm_turns = await self._get_stm_turns(session_id)

        elapsed_ms = (time.time() - start_time) * 1000

        stats = RetrievalStats(
            entities_found=len(entities),
            episodes_retrieved=len(episodes),
            facts_retrieved=len(facts),
            stm_turns=len(stm_turns),
            retrieval_time_ms=elapsed_ms
        )

        return RetrievalContext(
            entities=entities,
            episodes=episodes,
            facts=facts,
            stm_turns=stm_turns,
            stats=stats
        )

    async def _search_entities(
        self,
        query_vec: List[float],
        session_id: UUID
    ) -> List[Tuple[Entity, float]]:
        """
        Vector similarity search on entities.

        Args:
            query_vec: Query embedding.
            session_id: Session UUID.

        Returns:
            List of (Entity, similarity) tuples.
        """
        return await self.entity_repo.search_by_vector(
            session_id=session_id,
            query_vector=query_vec,
            limit=settings.RETRIEVAL_TOP_K_ENTITIES,
            min_similarity=0.4
        )

    async def _search_episodes(
        self,
        query_vec: List[float],
        session_id: UUID
    ) -> List[Tuple[Episode, float]]:
        """
        Vector similarity search on episodes (using summary vectors).

        Uses the idx_vec_summary partial index for O(log N) search.

        Args:
            query_vec: Query embedding.
            session_id: Session UUID.

        Returns:
            List of (Episode, similarity) tuples.
        """
        return await self.episode_repo.search_by_summary(
            session_id=session_id,
            query_vector=query_vec,
            limit=settings.RETRIEVAL_TOP_K_EPISODES,
            min_similarity=0.4
        )

    async def _search_graph(
        self,
        entity_ids: List[UUID],
        session_id: UUID
    ) -> List[Fact]:
        """
        Traverse knowledge graph from entities to facts.

        Args:
            entity_ids: List of entity UUIDs.
            session_id: Session UUID.

        Returns:
            List of current Fact objects.
        """
        if not entity_ids:
            return []

        return await self.fact_repo.get_facts_for_entities(
            session_id=session_id,
            entity_ids=entity_ids,
            current_only=True
        )

    async def _get_stm_turns(
        self,
        session_id: UUID
    ) -> List[Turn]:
        """
        Get recent turns from Redis Stream STM.

        Args:
            session_id: Session UUID.

        Returns:
            List of Turn objects in chronological order.
        """
        return await self.stm.get_recent_turns(
            session_id=session_id,
            count=settings.STM_CONTEXT_TURNS
        )

    def format_context(self, context: RetrievalContext) -> str:
        """
        Format retrieved context into a string for LLM.

        Args:
            context: RetrievalContext object.

        Returns:
            Formatted context string.
        """
        parts = []

        # Knowledge graph facts
        if context.facts:
            parts.append("--- KNOWLEDGE GRAPH ---")
            entity_map = {e.id: e.name for e, _ in context.entities}
            for fact in context.facts:
                entity_name = entity_map.get(fact.subject_entity_id, "Unknown")
                parts.append(f"{entity_name} {fact.predicate} {fact.object}")

        # Episode summaries
        if context.episodes:
            parts.append("\n--- RELEVANT EPISODES ---")
            for episode, similarity in context.episodes:
                parts.append(
                    f"[Turns {episode.turn_start}-{episode.turn_end}]: {episode.summary}"
                )

        # Recent STM turns
        if context.stm_turns:
            parts.append("\n--- CURRENT CONVERSATION ---")
            for turn in context.stm_turns[-10:]:  # Last 10 turns
                role_display = "User" if turn.role == "user" else "Assistant"
                parts.append(f"{role_display}: {turn.content}")

        return "\n".join(parts)


class ContextBuilder:
    """
    Helper class for building context strings for different purposes.
    """

    @staticmethod
    def build_system_prompt(context: RetrievalContext, base_prompt: str = None) -> str:
        """
        Build a system prompt with context.

        Args:
            context: RetrievalContext object.
            base_prompt: Optional base system prompt.

        Returns:
            Complete system prompt string.
        """
        if base_prompt is None:
            base_prompt = "You are Janus, an AI assistant with persistent memory."

        parts = [base_prompt]
        parts.append("\nUse the following context to inform your response:\n")

        # Format knowledge graph
        if context.facts:
            parts.append("KNOWN FACTS:")
            entity_map = {e.id: e.name for e, _ in context.entities}
            for fact in context.facts:
                entity_name = entity_map.get(fact.subject_entity_id, "Unknown")
                parts.append(f"- {entity_name} {fact.predicate}: {fact.object}")

        # Format episodes
        if context.episodes:
            parts.append("\nRELEVANT PAST CONVERSATIONS:")
            for episode, _ in context.episodes[:3]:  # Top 3 episodes
                parts.append(f"- {episode.summary}")

        # Format recent conversation
        if context.stm_turns:
            parts.append("\nCURRENT CONVERSATION:")
            for turn in context.stm_turns[-6:]:  # Last 6 turns
                role = "User" if turn.role == "user" else "You"
                # Truncate long messages
                content = turn.content[:200] + "..." if len(turn.content) > 200 else turn.content
                parts.append(f"{role}: {content}")

        return "\n".join(parts)

    @staticmethod
    def build_consolidation_prompt(turns: List[Turn]) -> str:
        """
        Build a prompt for memory consolidation.

        Args:
            turns: List of Turn objects to consolidate.

        Returns:
            Consolidation prompt string.
        """
        text_block = "\n".join([
            f"{t.role}: {t.content}"
            for t in turns
        ])

        return f"""Analyze this conversation segment and extract key information.

CONVERSATION:
{text_block}

Return JSON with:
1. "summary": A concise narrative summary of the conversation (1-2 sentences).
2. "facts": A list of permanent facts learned. Each fact should be an object with:
   - "subject": The entity the fact is about
   - "predicate": The relationship/property
   - "object": The value
   - "confidence": How certain (0.0-1.0)

Focus on facts that would be useful for future conversations:
- User preferences and interests
- Names, relationships, locations
- Technical details mentioned
- Commitments or plans

Example output:
{{
  "summary": "User discussed their vegetarian wife and asked about high-protein dinner options.",
  "facts": [
    {{"subject": "User's wife", "predicate": "dietary_preference", "object": "vegetarian", "confidence": 0.95}},
    {{"subject": "User", "predicate": "planning", "object": "dinner party", "confidence": 0.8}}
  ]
}}"""
