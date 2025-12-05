"""
LLM-Based Topic Detection for Janus 3.5

Replaces semantic-velocity cosine distance with LLM-based boundary detection.
Uses a small model (gpt-4o-mini) to determine if a message starts a new topic.
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Optional

from janus3.services.llm_service import LLMService
from janus3.config import settings

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
    role: str  # "user" or "assistant"
    content: str


TOPIC_DETECTION_PROMPT = """Analyze if the new message starts a NEW TOPIC or continues the current conversation topic.

Recent conversation:
{context}

New message: "{new_message}"

A NEW TOPIC is indicated when:
- The subject matter changes significantly
- A completely new question or request is introduced
- The conversation shifts to an unrelated area
- There's a clear break in the flow of discussion

CONTINUATION is indicated when:
- The message follows up on the current discussion
- It asks for clarification about recent topics
- It provides additional information about what was discussed
- It responds to or builds upon recent messages

Return JSON:
{{
    "is_new_topic": true/false,
    "confidence": 0.0-1.0,
    "topic_label": "2-3 word label for the current/new topic",
    "reasoning": "Brief explanation"
}}"""


class TopicDetector:
    """
    LLM-based topic boundary detection.

    Replaces semantic velocity algorithm with LLM judgment
    for more accurate topic boundary detection.
    """

    _instance: Optional["TopicDetector"] = None

    def __new__(cls) -> "TopicDetector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.llm = LLMService()
            self._initialized = True

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
                new_message=current_message[:500]  # Truncate for efficiency
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
            # Truncate long messages
            content = turn.content[:200] + "..." if len(turn.content) > 200 else turn.content
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _parse_response(self, response: str) -> TopicBoundaryResult:
        """Parse LLM JSON response into TopicBoundaryResult."""
        try:
            data = json.loads(response)

            is_boundary = bool(data.get("is_new_topic", False))
            confidence = float(data.get("confidence", 0.5))
            topic_label = str(data.get("topic_label", "unknown")).strip()
            reasoning = data.get("reasoning", "")

            # Clamp confidence to valid range
            confidence = max(0.0, min(1.0, confidence))

            return TopicBoundaryResult(
                is_boundary=is_boundary,
                confidence=confidence,
                topic_label=topic_label,
                reasoning=reasoning
            )

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse topic detection response: {e}")
            return TopicBoundaryResult(
                is_boundary=False,
                confidence=0.0,
                topic_label="unknown",
                reasoning=f"Parse error: {str(e)}"
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
