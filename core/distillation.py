"""
Memory Distillation for Janus 3.5

Compresses multiple episodes/facts into composite representations
for efficient long-term context management.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple
from uuid import UUID, uuid4

from janus3.services.llm_service import LLMService
from janus3.core.embedder import Embedder
from janus3.config import settings

logger = logging.getLogger(__name__)


@dataclass
class DistilledMemory:
    """A distilled memory composite."""
    id: UUID
    session_id: UUID
    source_episode_ids: List[UUID] = field(default_factory=list)
    source_fact_ids: List[UUID] = field(default_factory=list)
    compressed_content: str  # 50-100 word synthesis
    topic_cluster: str
    time_range: Tuple[datetime, datetime] = field(default_factory=lambda: (datetime.utcnow(), datetime.utcnow()))
    embedding: List[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


EPISODE_DISTILLATION_PROMPT = """Synthesize these related conversation episodes into a single coherent summary.

Episodes (chronologically ordered):
{episodes}

Requirements:
- Preserve key facts, relationships, and temporal order
- Maximum {max_words} words
- Maintain important details while removing redundancy
- Focus on information useful for future conversations

Return JSON:
{{
    "synthesis": "Your synthesized summary here...",
    "topic_cluster": "2-3 word topic label",
    "key_entities": ["Entity1", "Entity2"]
}}"""


FACT_DISTILLATION_PROMPT = """Synthesize these facts about {entity_name} into a brief narrative profile.

Facts:
{facts}

Requirements:
- Create a natural language profile
- Maximum {max_words} words
- Highlight the most important attributes
- Make it useful for understanding who/what {entity_name} is

Return JSON:
{{
    "profile": "Your profile summary here...",
    "key_attributes": ["attr1", "attr2", "attr3"]
}}"""


class MemoryDistiller:
    """
    Compresses multiple memories into composite representations.

    Uses LLM to synthesize related episodes or facts about an entity
    into more compact representations.
    """

    _instance: Optional["MemoryDistiller"] = None

    def __new__(cls) -> "MemoryDistiller":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.llm = LLMService()
            self.embedder = Embedder()
            self._initialized = True

    async def distill_episodes(
        self,
        episodes: List[dict],
        session_id: UUID,
        max_output_words: int = 100
    ) -> Optional[DistilledMemory]:
        """
        Compress multiple related episodes into single representation.

        Args:
            episodes: List of episode dicts with 'id', 'summary'/'compressed_state', 'created_at'.
            session_id: Session these episodes belong to.
            max_output_words: Target length for compressed output.

        Returns:
            DistilledMemory with synthesized content, or None if distillation fails.
        """
        if not episodes:
            return None

        if len(episodes) < settings.DISTILLATION_MIN_EPISODES:
            logger.debug(f"Not enough episodes for distillation: {len(episodes)}")
            return None

        if not settings.DISTILLATION_ENABLED:
            return None

        if not self.llm.has_api:
            logger.debug("LLM API unavailable, skipping distillation")
            return None

        try:
            # Format episodes chronologically
            sorted_episodes = sorted(episodes, key=lambda e: e.get('created_at', datetime.min))
            episode_texts = []

            for e in sorted_episodes:
                date_str = ""
                if 'created_at' in e:
                    date_str = f"[{e['created_at'].strftime('%Y-%m-%d')}] "

                # Prefer compressed_state if available
                content = e.get('compressed_state') or e.get('summary', '')
                episode_texts.append(f"{date_str}{content}")

            prompt = EPISODE_DISTILLATION_PROMPT.format(
                episodes="\n".join(f"- {t}" for t in episode_texts),
                max_words=max_output_words
            )

            response = await self.llm.generate_json(
                system_prompt="You are a memory synthesizer. Be concise and accurate.",
                user_content=prompt,
                temperature=0.2,
            )

            result = json.loads(response)
            synthesis = result.get("synthesis", "").strip()
            topic_cluster = result.get("topic_cluster", "general").strip()

            if not synthesis:
                logger.warning("Empty synthesis from distillation")
                return None

            # Generate embedding for the synthesis
            embedding = await self.embedder.encode(synthesis)

            # Calculate time range
            times = [e.get('created_at') for e in sorted_episodes if e.get('created_at')]
            time_range = (
                min(times) if times else datetime.utcnow(),
                max(times) if times else datetime.utcnow()
            )

            return DistilledMemory(
                id=uuid4(),
                session_id=session_id,
                source_episode_ids=[e.get('id') for e in episodes if e.get('id')],
                source_fact_ids=[],
                compressed_content=synthesis,
                topic_cluster=topic_cluster,
                time_range=time_range,
                embedding=embedding,
                created_at=datetime.utcnow()
            )

        except Exception as e:
            logger.warning(f"Episode distillation failed: {e}")
            return None

    async def distill_facts(
        self,
        facts: List[dict],
        entity_name: str,
        session_id: UUID,
        max_output_words: int = 50
    ) -> Optional[DistilledMemory]:
        """
        Compress facts about an entity into narrative form.

        Args:
            facts: List of fact dicts with 'id', 'subject', 'predicate', 'object', etc.
            entity_name: The entity these facts are about.
            session_id: Session these facts belong to.
            max_output_words: Target length for compressed output.

        Returns:
            DistilledMemory with synthesized profile, or None if distillation fails.
        """
        if not facts:
            return None

        if not settings.DISTILLATION_ENABLED:
            return None

        if not self.llm.has_api:
            logger.debug("LLM API unavailable, skipping distillation")
            return None

        try:
            # Format facts
            fact_texts = []
            for f in facts:
                subject = f.get('subject', '')
                predicate = f.get('canonical_predicate') or f.get('predicate', '')
                obj = f.get('object', '') or f.get('object_value', '')
                fact_texts.append(f"- {subject} {predicate} {obj}")

            prompt = FACT_DISTILLATION_PROMPT.format(
                entity_name=entity_name,
                facts="\n".join(fact_texts),
                max_words=max_output_words
            )

            response = await self.llm.generate_json(
                system_prompt="You are a fact synthesizer. Create natural language profiles.",
                user_content=prompt,
                temperature=0.2,
            )

            result = json.loads(response)
            profile = result.get("profile", "").strip()

            if not profile:
                logger.warning("Empty profile from fact distillation")
                return None

            # Generate embedding
            embedding = await self.embedder.encode(profile)

            # Calculate time range from facts
            times = [f.get('created_at') for f in facts if f.get('created_at')]
            time_range = (
                min(times) if times else datetime.utcnow(),
                max(times) if times else datetime.utcnow()
            )

            return DistilledMemory(
                id=uuid4(),
                session_id=session_id,
                source_episode_ids=[],
                source_fact_ids=[f.get('id') for f in facts if f.get('id')],
                compressed_content=profile,
                topic_cluster=entity_name,
                time_range=time_range,
                embedding=embedding,
                created_at=datetime.utcnow()
            )

        except Exception as e:
            logger.warning(f"Fact distillation failed: {e}")
            return None

    def should_distill(self, episode_count: int, topic_label: str = None) -> bool:
        """
        Check if distillation should be triggered.

        Args:
            episode_count: Number of episodes in the cluster.
            topic_label: Optional topic label for the cluster.

        Returns:
            True if distillation should be performed.
        """
        if not settings.DISTILLATION_ENABLED:
            return False

        return episode_count >= settings.DISTILLATION_MIN_EPISODES


def get_memory_distiller() -> MemoryDistiller:
    """Get memory distiller instance (dependency injection helper)."""
    return MemoryDistiller()
