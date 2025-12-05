"""
LLM Reranker for Janus 3.5

Uses a small model (gpt-4o-mini) to rerank retrieved memories
based on relevance to the user's query.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, TypeVar

from janus3.services.llm_service import LLMService
from janus3.config import settings

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class RankedItem:
    """A reranked item with scores and explanation."""
    item: Any
    original_score: float
    rerank_score: float  # 0.0 to 1.0
    relevance_reason: str


RERANK_PROMPT = """Rate each memory's relevance to the query on a scale of 0-10.

Query: "{query}"

Memories to evaluate:
{memories}

For each memory, assess:
- How directly relevant is it to answering the query?
- Does it provide useful context or information?
- Is it recent and applicable?

Return a JSON array with scores for each memory:
[
    {{"index": 0, "score": 8, "reason": "Directly addresses the question about..."}},
    {{"index": 1, "score": 3, "reason": "Tangentially related but not specific..."}}
]

Be strict: only give high scores (7+) to highly relevant memories.
Give low scores (0-3) to irrelevant or outdated information."""


class LLMReranker:
    """
    LLM-based reranking for retrieved memories.

    Takes a set of retrieved items and reranks them based on
    relevance to the user's query using LLM judgment.
    """

    _instance: Optional["LLMReranker"] = None

    def __new__(cls) -> "LLMReranker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.llm = LLMService()
            self._initialized = True

    async def rerank(
        self,
        query: str,
        items: List[T],
        item_formatter: Callable[[T], str],
        top_k: Optional[int] = None,
        original_scores: Optional[List[float]] = None
    ) -> List[RankedItem]:
        """
        Rerank items using LLM relevance scoring.

        Args:
            query: The user's query for relevance scoring.
            items: List of items to rerank (episodes, facts, etc.).
            item_formatter: Function to convert item to text for LLM.
            top_k: Number of top items to return (default: settings.RERANKER_TOP_K).
            original_scores: Optional original scores from vector search.

        Returns:
            List of top-k RankedItem objects sorted by rerank score.
        """
        if not items:
            return []

        if not settings.RERANKER_ENABLED:
            # Pass through without reranking
            return self._passthrough(items, original_scores)

        if not self.llm.has_api:
            logger.debug("LLM API unavailable, returning original order")
            return self._passthrough(items, original_scores)

        top_k = top_k or settings.RERANKER_TOP_K

        try:
            # Format memories for LLM
            formatted_memories = self._format_memories(items, item_formatter)

            prompt = RERANK_PROMPT.format(
                query=query[:500],  # Truncate long queries
                memories=formatted_memories
            )

            response = await self.llm.generate_json(
                system_prompt="You are a memory relevance scorer. Be precise and strict.",
                user_content=prompt,
                temperature=0.1,  # Low temperature for consistent scoring
            )

            ranked = self._parse_response(
                response, items, original_scores or [0.0] * len(items)
            )

            # Sort by rerank score and take top-k
            ranked.sort(key=lambda x: x.rerank_score, reverse=True)
            return ranked[:top_k]

        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return self._passthrough(items, original_scores)[:top_k]

    def _format_memories(
        self,
        items: List[Any],
        item_formatter: Callable[[Any], str]
    ) -> str:
        """Format items as numbered list for LLM."""
        lines = []
        for i, item in enumerate(items):
            text = item_formatter(item)
            # Truncate individual items
            if len(text) > 300:
                text = text[:300] + "..."
            lines.append(f"[{i}] {text}")
        return "\n".join(lines)

    def _parse_response(
        self,
        response: str,
        items: List[Any],
        original_scores: List[float]
    ) -> List[RankedItem]:
        """Parse LLM JSON response into RankedItem list."""
        try:
            scores = json.loads(response)

            # Build result list
            ranked = []
            seen_indices = set()

            for score_item in scores:
                idx = score_item.get("index", -1)
                score = score_item.get("score", 0)
                reason = score_item.get("reason", "")

                # Validate index
                if idx < 0 or idx >= len(items) or idx in seen_indices:
                    continue

                seen_indices.add(idx)

                # Normalize score to 0-1 range
                normalized_score = max(0.0, min(1.0, score / 10.0))

                ranked.append(RankedItem(
                    item=items[idx],
                    original_score=original_scores[idx] if idx < len(original_scores) else 0.0,
                    rerank_score=normalized_score,
                    relevance_reason=reason
                ))

            # Add any items that weren't scored (with score 0)
            for i, item in enumerate(items):
                if i not in seen_indices:
                    ranked.append(RankedItem(
                        item=item,
                        original_score=original_scores[i] if i < len(original_scores) else 0.0,
                        rerank_score=0.0,
                        relevance_reason="Not scored by reranker"
                    ))

            return ranked

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse rerank response: {e}")
            return self._passthrough(items, original_scores)

    def _passthrough(
        self,
        items: List[Any],
        original_scores: Optional[List[float]] = None
    ) -> List[RankedItem]:
        """Return items in original order as RankedItems."""
        scores = original_scores or [0.0] * len(items)
        return [
            RankedItem(
                item=item,
                original_score=scores[i] if i < len(scores) else 0.0,
                rerank_score=scores[i] if i < len(scores) else 0.0,
                relevance_reason="Passthrough (no reranking)"
            )
            for i, item in enumerate(items)
        ]


def get_reranker() -> LLMReranker:
    """Get reranker instance (dependency injection helper)."""
    return LLMReranker()
