"""
Tests for Parallel Retrieval (Fix 2)
"""

import pytest
import asyncio
import time
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestParallelRetrieval:
    """Test cases for parallel retrieval (Fix 2)."""

    @pytest.mark.asyncio
    async def test_parallel_retrieval_timing(self):
        """
        Verify that entity and episode searches run in parallel,
        not sequentially.

        If parallel: ~0.5s total
        If sequential: ~1.0s total
        """
        from core.retrieval import HybridRetriever
        from core.models import Entity, Episode

        # Create mock retriever
        mock_db_pool = MagicMock()
        mock_redis = MagicMock()
        mock_embedder = MagicMock()

        retriever = HybridRetriever(mock_db_pool, mock_redis, mock_embedder)

        # Mock slow database calls (0.5s each)
        async def slow_entity_search(*args, **kwargs):
            await asyncio.sleep(0.5)
            return []

        async def slow_episode_search(*args, **kwargs):
            await asyncio.sleep(0.5)
            return []

        async def fast_graph_search(*args, **kwargs):
            return []

        async def fast_stm_turns(*args, **kwargs):
            return []

        # Patch the internal methods
        retriever._search_entities = slow_entity_search
        retriever._search_episodes = slow_episode_search
        retriever._search_graph = fast_graph_search
        retriever._get_stm_turns = fast_stm_turns

        # Time the retrieval
        start = time.time()

        await retriever.build_context(
            query="test query",
            query_vec=[0.1] * 384,
            session_id=uuid4()
        )

        elapsed = time.time() - start

        # If parallel, should be ~0.5s (both run simultaneously)
        # If sequential, would be ~1.0s (0.5 + 0.5)
        # Allow some margin for overhead
        assert elapsed < 0.8, f"Retrieval took {elapsed}s - not running in parallel!"

    @pytest.mark.asyncio
    async def test_graph_search_waits_for_entities(self):
        """
        Verify that graph search runs AFTER entity search completes.
        (Sequential dependency)
        """
        from core.retrieval import HybridRetriever
        from core.models import Entity
        from uuid import uuid4
        from datetime import datetime

        mock_db_pool = MagicMock()
        mock_redis = MagicMock()
        mock_embedder = MagicMock()

        retriever = HybridRetriever(mock_db_pool, mock_redis, mock_embedder)

        # Track call order
        call_order = []

        async def entity_search(*args, **kwargs):
            call_order.append("entities")
            await asyncio.sleep(0.1)
            # Return mock entities
            return [
                (Entity(
                    id=uuid4(),
                    session_id=uuid4(),
                    name="Test",
                    normalized_name="test",
                    entity_type="concept",
                    embedding=[0.1] * 384,
                    created_at=datetime.utcnow()
                ), 0.9)
            ]

        async def episode_search(*args, **kwargs):
            call_order.append("episodes")
            return []

        async def graph_search(entity_ids, *args, **kwargs):
            call_order.append("graph")
            # Verify we received entity IDs
            assert len(entity_ids) > 0, "Graph search called before entities returned"
            return []

        async def stm_turns(*args, **kwargs):
            call_order.append("stm")
            return []

        retriever._search_entities = entity_search
        retriever._search_episodes = episode_search
        retriever._search_graph = graph_search
        retriever._get_stm_turns = stm_turns

        await retriever.build_context(
            query="test",
            query_vec=[0.1] * 384,
            session_id=uuid4()
        )

        # Entities and episodes should run first (parallel)
        # Graph should run after entities complete
        assert "graph" in call_order
        entities_idx = call_order.index("entities")
        graph_idx = call_order.index("graph")
        assert graph_idx > entities_idx, "Graph search ran before entity search completed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
