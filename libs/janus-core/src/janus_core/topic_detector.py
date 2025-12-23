"""
LLM-Based Topic Detection for Janus 3.5

Uses gpt-4o-mini to determine if a message starts a new topic.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from janus_core.llm.service import LLMService
from janus_core.config import settings
from janus_core.patterns import singleton, parse_json_response
from janus_core.prompts import TOPIC_DETECTION_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class TopicBoundaryResult:
    """Result of topic boundary detection."""
    is_boundary: bool
    confidence: float  # 0.0 to 1.0
    topic_label: str  # 2-3 word topic label
    reasoning: Optional[str] = None


@dataclass
class Turn:
    """Simplified turn representation for topic detection."""
    role: str
    content: str


@singleton
class TopicDetector:
    """LLM-based topic boundary detection."""

    def __init__(self):
        self.llm = LLMService()

    async def detect_boundary(
        self,
        recent_turns: List[Turn],
        current_message: str
    ) -> TopicBoundaryResult:
        """
        Detect if the current message starts a new topic.

        Args:
            recent_turns: Recent conversation turns for context.
            current_message: The new message to evaluate.

        Returns:
            TopicBoundaryResult with boundary decision and metadata.
        """
        if not settings.TOPIC_DETECTION_ENABLED:
            # Return non-boundary when disabled
            return TopicBoundaryResult(
                is_boundary=False,
                confidence=0.0,
                topic_label="unknown",
                reasoning="Topic detection disabled"
            )

        if not self.llm.has_api:
            logger.debug("LLM API unavailable, returning default non-boundary")
            return self._mock_result()

        # If no recent turns, this is the first message (new topic by definition)
        if not recent_turns:
            return TopicBoundaryResult(
                is_boundary=True,
                confidence=1.0,
                topic_label="conversation start",
                reasoning="First message in conversation"
            )

        try:
            # Format context from recent turns
            context = self._format_context(recent_turns)

            prompt = TOPIC_DETECTION_PROMPT.format(
                context=context,
                new_message=current_message[:settings.QUERY_TRUNCATION_LIMIT]
            )

            response = await self.llm.generate_json(
                system_prompt="You are a conversation analyst. Detect topic boundaries accurately.",
                user_content=prompt,
                temperature=0.1,  # Low temperature for consistent detection
            )

            return self._parse_response(response)

        except Exception as e:
            logger.warning(f"Topic detection failed: {e}")
            return TopicBoundaryResult(
                is_boundary=False,
                confidence=0.0,
                topic_label="unknown",
                reasoning=f"Detection error: {str(e)}"
            )

    def _format_context(self, turns: List[Turn]) -> str:
        """Format recent turns into context string."""
        # Limit to configured number of turns
        max_turns = settings.TOPIC_DETECTION_CONTEXT_TURNS
        recent = turns[-max_turns:] if len(turns) > max_turns else turns

        lines = []
        for turn in recent:
            role = "User" if turn.role == "user" else "Assistant"
            content = turn.content[:settings.TOPIC_TRUNCATION_LIMIT] + "..." if len(turn.content) > settings.TOPIC_TRUNCATION_LIMIT else turn.content
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _parse_response(self, response: str) -> TopicBoundaryResult:
        """Parse LLM JSON response into TopicBoundaryResult."""
        fallback = TopicBoundaryResult(
            is_boundary=False,
            confidence=0.0,
            topic_label="unknown",
            reasoning="Parse error"
        )
        data = parse_json_response(response, None, "Topic detection response")
        if data is None:
            return fallback

        is_boundary = bool(data.get("is_new_topic", False))
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        topic_label = str(data.get("topic_label", "unknown")).strip()
        reasoning = data.get("reasoning", "")

        return TopicBoundaryResult(
            is_boundary=is_boundary,
            confidence=confidence,
            topic_label=topic_label,
            reasoning=reasoning
        )

    def _mock_result(self) -> TopicBoundaryResult:
        """Return mock result when API is unavailable."""
        return TopicBoundaryResult(
            is_boundary=False,
            confidence=0.5,
            topic_label="mock topic",
            reasoning="API unavailable - returning default"
        )


def get_topic_detector() -> TopicDetector:
    """Get topic detector instance (dependency injection helper)."""
    return TopicDetector()
