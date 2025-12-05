"""
Tests for LLM-Based Topic Detection (GPT-5.1 Upgrade 2)
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.topic_detector import (
    TopicDetector,
    TopicBoundaryResult,
    Turn,
    get_topic_detector,
)
from config import settings


class TestTopicBoundaryResult:
    """Test TopicBoundaryResult dataclass."""

    def test_creation(self):
        """Test basic creation of TopicBoundaryResult."""
        result = TopicBoundaryResult(
            is_boundary=True,
            confidence=0.85,
            topic_label="python debugging",
            reasoning="New topic introduced"
        )

        assert result.is_boundary is True
        assert result.confidence == 0.85
        assert result.topic_label == "python debugging"
        assert result.reasoning == "New topic introduced"

    def test_optional_reasoning(self):
        """Test that reasoning is optional."""
        result = TopicBoundaryResult(
            is_boundary=False,
            confidence=0.5,
            topic_label="general chat"
        )

        assert result.reasoning is None


class TestTurn:
    """Test Turn dataclass."""

    def test_user_turn(self):
        """Test user turn creation."""
        turn = Turn(role="user", content="Hello, how are you?")

        assert turn.role == "user"
        assert turn.content == "Hello, how are you?"

    def test_assistant_turn(self):
        """Test assistant turn creation."""
        turn = Turn(role="assistant", content="I'm doing great!")

        assert turn.role == "assistant"
        assert turn.content == "I'm doing great!"


class TestTopicDetector:
    """Test TopicDetector class."""

    def test_singleton_pattern(self):
        """TopicDetector should be a singleton."""
        detector1 = TopicDetector()
        detector2 = TopicDetector()

        assert detector1 is detector2

    def test_get_topic_detector_helper(self):
        """get_topic_detector should return the singleton instance."""
        detector = get_topic_detector()

        assert isinstance(detector, TopicDetector)
        assert detector is TopicDetector()

    @pytest.mark.asyncio
    async def test_empty_turns_is_boundary(self):
        """First message (no recent turns) should be marked as boundary."""
        detector = TopicDetector()

        result = await detector.detect_boundary(
            recent_turns=[],
            current_message="Hello, I need help with Python."
        )

        assert result.is_boundary is True
        assert result.confidence == 1.0
        assert "conversation start" in result.topic_label.lower()

    @pytest.mark.asyncio
    async def test_disabled_returns_non_boundary(self):
        """When disabled, should return non-boundary result."""
        detector = TopicDetector()
        original = settings.TOPIC_DETECTION_ENABLED

        try:
            settings.TOPIC_DETECTION_ENABLED = False

            result = await detector.detect_boundary(
                recent_turns=[Turn(role="user", content="Test")],
                current_message="Another test"
            )

            assert result.is_boundary is False
            assert "disabled" in result.reasoning.lower()
        finally:
            settings.TOPIC_DETECTION_ENABLED = original

    def test_format_context(self):
        """Test context formatting for LLM prompt."""
        detector = TopicDetector()

        turns = [
            Turn(role="user", content="What is Python?"),
            Turn(role="assistant", content="Python is a programming language."),
            Turn(role="user", content="Can you show an example?"),
        ]

        context = detector._format_context(turns)

        assert "User: What is Python?" in context
        assert "Assistant: Python is a programming language." in context
        assert "User: Can you show an example?" in context

    def test_format_context_truncates_long_messages(self):
        """Long messages should be truncated in context."""
        detector = TopicDetector()

        long_content = "A" * 500  # 500 characters
        turns = [Turn(role="user", content=long_content)]

        context = detector._format_context(turns)

        # Should be truncated to 200 chars + "..."
        assert len(context) < 250
        assert "..." in context

    def test_format_context_respects_turn_limit(self):
        """Context should respect TOPIC_DETECTION_CONTEXT_TURNS limit."""
        detector = TopicDetector()
        max_turns = settings.TOPIC_DETECTION_CONTEXT_TURNS

        # Create more turns than the limit
        turns = [
            Turn(role="user", content=f"Message {i}")
            for i in range(max_turns + 5)
        ]

        context = detector._format_context(turns)

        # Should only include the last max_turns messages
        assert f"Message {max_turns + 4}" in context  # Last message
        assert f"Message 0" not in context  # First message should be excluded

    def test_parse_response_valid_json(self):
        """Test parsing valid JSON response."""
        detector = TopicDetector()

        response = '''
        {
            "is_new_topic": true,
            "confidence": 0.8,
            "topic_label": "database design",
            "reasoning": "Shifted from Python to databases"
        }
        '''

        result = detector._parse_response(response)

        assert result.is_boundary is True
        assert result.confidence == 0.8
        assert result.topic_label == "database design"
        assert "databases" in result.reasoning.lower()

    def test_parse_response_clamps_confidence(self):
        """Confidence should be clamped to 0.0-1.0 range."""
        detector = TopicDetector()

        # Test over 1.0
        response_high = '{"is_new_topic": true, "confidence": 1.5, "topic_label": "test"}'
        result_high = detector._parse_response(response_high)
        assert result_high.confidence == 1.0

        # Test below 0.0
        response_low = '{"is_new_topic": false, "confidence": -0.5, "topic_label": "test"}'
        result_low = detector._parse_response(response_low)
        assert result_low.confidence == 0.0

    def test_parse_response_invalid_json(self):
        """Invalid JSON should return safe default result."""
        detector = TopicDetector()

        result = detector._parse_response("not valid json {{{")

        assert result.is_boundary is False
        assert result.confidence == 0.0
        assert result.topic_label == "unknown"
        assert "error" in result.reasoning.lower()

    def test_parse_response_missing_fields(self):
        """Missing fields should use defaults."""
        detector = TopicDetector()

        response = '{"is_new_topic": true}'  # Missing confidence, topic_label

        result = detector._parse_response(response)

        assert result.is_boundary is True
        assert result.confidence == 0.5  # Default
        assert result.topic_label == "unknown"  # Default

    def test_mock_result(self):
        """Test mock result generation."""
        detector = TopicDetector()

        result = detector._mock_result()

        assert result.is_boundary is False
        assert result.confidence == 0.5
        assert result.topic_label == "mock topic"
        assert "unavailable" in result.reasoning.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
