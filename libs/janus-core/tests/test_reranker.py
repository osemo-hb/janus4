"""
Tests for LLM Reranker (GPT-5.1 Upgrade 5)
"""

import pytest

from janus_core.reranker import LLMReranker, RankedItem, get_reranker
from janus_core.config import settings


class TestRankedItem:
    """Test RankedItem dataclass."""

    def test_creation(self):
        """Test basic creation of RankedItem."""
        item = RankedItem(
            item={"id": "test", "content": "Some memory"},
            original_score=0.85,
            rerank_score=0.92,
            relevance_reason="Directly relevant to query"
        )

        assert item.item["id"] == "test"
        assert item.original_score == 0.85
        assert item.rerank_score == 0.92
        assert "relevant" in item.relevance_reason.lower()


class TestLLMReranker:
    """Test LLMReranker class."""

    def test_singleton_pattern(self):
        """LLMReranker should be a singleton."""
        reranker1 = LLMReranker()
        reranker2 = LLMReranker()

        assert reranker1 is reranker2

    def test_get_reranker_helper(self):
        """get_reranker should return the singleton instance."""
        reranker = get_reranker()

        assert isinstance(reranker, LLMReranker)
        assert reranker is LLMReranker()

    @pytest.mark.asyncio
    async def test_empty_items_returns_empty(self):
        """Empty input should return empty output."""
        reranker = LLMReranker()

        result = await reranker.rerank(
            query="test query",
            items=[],
            item_formatter=str
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_disabled_passthrough(self):
        """When disabled, should pass through items unchanged."""
        reranker = LLMReranker()
        original = settings.RERANKER_ENABLED

        try:
            settings.RERANKER_ENABLED = False

            items = ["item1", "item2", "item3"]
            scores = [0.9, 0.7, 0.5]

            result = await reranker.rerank(
                query="test query",
                items=items,
                item_formatter=str,
                original_scores=scores
            )

            assert len(result) == 3
            assert result[0].item == "item1"
            assert result[0].original_score == 0.9
            assert "passthrough" in result[0].relevance_reason.lower()
        finally:
            settings.RERANKER_ENABLED = original

    def test_format_memories(self):
        """Test memory formatting for LLM prompt."""
        reranker = LLMReranker()

        items = ["Memory one", "Memory two", "Memory three"]

        formatted = reranker._format_memories(items, str)

        assert "[0] Memory one" in formatted
        assert "[1] Memory two" in formatted
        assert "[2] Memory three" in formatted

    def test_format_memories_truncates(self):
        """Long items should be truncated."""
        reranker = LLMReranker()

        long_item = "A" * 500  # 500 chars
        items = [long_item]

        formatted = reranker._format_memories(items, str)

        # Should be truncated to 300 + "..."
        assert len(formatted) < 350
        assert "..." in formatted

    def test_parse_response_valid(self):
        """Test parsing valid rerank response."""
        reranker = LLMReranker()

        items = ["item0", "item1", "item2"]
        original_scores = [0.5, 0.6, 0.7]

        response = '''
        [
            {"index": 0, "score": 8, "reason": "Very relevant"},
            {"index": 2, "score": 6, "reason": "Somewhat relevant"},
            {"index": 1, "score": 2, "reason": "Not very relevant"}
        ]
        '''

        result = reranker._parse_response(response, items, original_scores)

        assert len(result) == 3

        # Check that scores are normalized (0-1 range)
        for r in result:
            assert 0.0 <= r.rerank_score <= 1.0

        # Find item0 result
        item0_result = next(r for r in result if r.item == "item0")
        assert item0_result.rerank_score == 0.8  # 8/10
        assert item0_result.original_score == 0.5

    def test_parse_response_skips_invalid_indices(self):
        """Invalid indices should be skipped."""
        reranker = LLMReranker()

        items = ["item0", "item1"]
        original_scores = [0.5, 0.6]

        response = '''
        [
            {"index": 0, "score": 8, "reason": "Valid"},
            {"index": 99, "score": 9, "reason": "Invalid index"},
            {"index": -1, "score": 9, "reason": "Negative index"}
        ]
        '''

        result = reranker._parse_response(response, items, original_scores)

        # Should have 2 results (one valid from response, one unscored)
        assert len(result) == 2

        # item0 should have score from response
        item0_result = next(r for r in result if r.item == "item0")
        assert item0_result.rerank_score == 0.8

        # item1 should be unscored (score 0)
        item1_result = next(r for r in result if r.item == "item1")
        assert item1_result.rerank_score == 0.0

    def test_parse_response_skips_duplicate_indices(self):
        """Duplicate indices should be skipped."""
        reranker = LLMReranker()

        items = ["item0", "item1"]
        original_scores = [0.5, 0.6]

        response = '''
        [
            {"index": 0, "score": 8, "reason": "First"},
            {"index": 0, "score": 5, "reason": "Duplicate - should skip"}
        ]
        '''

        result = reranker._parse_response(response, items, original_scores)

        # Should have both items
        assert len(result) == 2

        # item0 should only have first score
        item0_result = next(r for r in result if r.item == "item0")
        assert item0_result.rerank_score == 0.8

    def test_parse_response_invalid_json(self):
        """Invalid JSON should return passthrough."""
        reranker = LLMReranker()

        items = ["item0", "item1"]
        original_scores = [0.5, 0.6]

        result = reranker._parse_response("not valid json", items, original_scores)

        assert len(result) == 2
        assert result[0].item == "item0"
        assert "passthrough" in result[0].relevance_reason.lower()

    def test_passthrough(self):
        """Test passthrough method."""
        reranker = LLMReranker()

        items = ["a", "b", "c"]
        scores = [0.9, 0.8, 0.7]

        result = reranker._passthrough(items, scores)

        assert len(result) == 3
        assert result[0].item == "a"
        assert result[0].original_score == 0.9
        assert result[0].rerank_score == 0.9  # Same as original
        assert "passthrough" in result[0].relevance_reason.lower()

    def test_passthrough_no_scores(self):
        """Passthrough without scores should use 0.0."""
        reranker = LLMReranker()

        items = ["a", "b"]

        result = reranker._passthrough(items, None)

        assert len(result) == 2
        assert result[0].original_score == 0.0
        assert result[0].rerank_score == 0.0

    @pytest.mark.asyncio
    async def test_respects_top_k(self):
        """Should return at most top_k items."""
        reranker = LLMReranker()
        original = settings.RERANKER_ENABLED

        try:
            settings.RERANKER_ENABLED = False  # Use passthrough

            items = ["a", "b", "c", "d", "e"]
            scores = [0.9, 0.8, 0.7, 0.6, 0.5]

            result = await reranker.rerank(
                query="test",
                items=items,
                item_formatter=str,
                original_scores=scores,
                top_k=3
            )

            assert len(result) == 3
        finally:
            settings.RERANKER_ENABLED = original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
