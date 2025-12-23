"""
Tests for OpenAI Embedder (GPT-5.1 Upgrade 4)
"""

import pytest

from janus_core.embedder import Embedder
from janus_core.config import settings


class TestEmbedder:
    """Test Embedder class."""

    def test_singleton_pattern(self):
        """Embedder should be a singleton."""
        embedder1 = Embedder()
        embedder2 = Embedder()

        assert embedder1 is embedder2

    def test_dimension_property(self):
        """Dimension property should return configured value."""
        embedder = Embedder()

        assert embedder.dimension == settings.EMBEDDING_DIM
        assert embedder.dimension == 1536  # GPT-5.1 upgrade value

    def test_has_api_property(self):
        """has_api should indicate if client is available."""
        embedder = Embedder()

        # This depends on whether OPENAI_API_KEY is set
        # Just verify the property exists and returns a bool
        assert isinstance(embedder.has_api, bool)

    def test_mock_embedding_dimension(self):
        """Mock embedding should have correct dimension."""
        embedder = Embedder()

        mock = embedder._mock_embedding()

        assert len(mock) == settings.EMBEDDING_DIM
        assert all(isinstance(v, float) for v in mock)

    def test_mock_embedding_value_range(self):
        """Mock embedding values should be in reasonable range."""
        embedder = Embedder()

        mock = embedder._mock_embedding()

        # Values should be between -0.1 and 0.1 (as per implementation)
        assert all(-0.1 <= v <= 0.1 for v in mock)

    @pytest.mark.asyncio
    async def test_encode_returns_correct_dimension(self):
        """Encode should return correct dimension vector."""
        embedder = Embedder()

        # This will use mock if no API key is available
        result = await embedder.encode("Hello, world!")

        assert len(result) == settings.EMBEDDING_DIM
        assert all(isinstance(v, float) for v in result)

    @pytest.mark.asyncio
    async def test_encode_batch_empty(self):
        """Empty batch should return empty list."""
        embedder = Embedder()

        result = await embedder.encode_batch([])

        assert result == []

    @pytest.mark.asyncio
    async def test_encode_batch_returns_correct_count(self):
        """Batch encode should return same number of vectors as inputs."""
        embedder = Embedder()

        texts = ["Hello", "World", "Test"]
        result = await embedder.encode_batch(texts)

        assert len(result) == 3
        assert all(len(vec) == settings.EMBEDDING_DIM for vec in result)

    def test_cosine_distance_identical_vectors(self):
        """Identical vectors should have distance 0."""
        vec = [0.1, 0.2, 0.3, 0.4]

        distance = Embedder.cosine_distance(vec, vec)

        assert distance == pytest.approx(0.0, abs=1e-6)

    def test_cosine_distance_opposite_vectors(self):
        """Opposite vectors should have distance close to 2."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [-1.0, 0.0, 0.0]

        distance = Embedder.cosine_distance(vec_a, vec_b)

        assert distance == pytest.approx(2.0, abs=1e-6)

    def test_cosine_distance_orthogonal_vectors(self):
        """Orthogonal vectors should have distance 1."""
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]

        distance = Embedder.cosine_distance(vec_a, vec_b)

        assert distance == pytest.approx(1.0, abs=1e-6)

    def test_cosine_distance_zero_vector(self):
        """Zero vector should return distance 1."""
        vec_a = [0.0, 0.0, 0.0]
        vec_b = [1.0, 2.0, 3.0]

        distance = Embedder.cosine_distance(vec_a, vec_b)

        assert distance == 1.0

    def test_cosine_similarity_identical(self):
        """Identical vectors should have similarity 1."""
        vec = [0.1, 0.2, 0.3]

        similarity = Embedder.cosine_similarity(vec, vec)

        assert similarity == pytest.approx(1.0, abs=1e-6)

    def test_cosine_similarity_opposite(self):
        """Opposite vectors should have similarity -1."""
        vec_a = [1.0, 0.0]
        vec_b = [-1.0, 0.0]

        similarity = Embedder.cosine_similarity(vec_a, vec_b)

        assert similarity == pytest.approx(-1.0, abs=1e-6)

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors should have similarity 0."""
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]

        similarity = Embedder.cosine_similarity(vec_a, vec_b)

        assert similarity == pytest.approx(0.0, abs=1e-6)

    def test_similarity_distance_relationship(self):
        """Similarity and distance should satisfy: distance = 1 - similarity."""
        vec_a = [0.5, 0.3, 0.8]
        vec_b = [0.2, 0.6, 0.4]

        distance = Embedder.cosine_distance(vec_a, vec_b)
        similarity = Embedder.cosine_similarity(vec_a, vec_b)

        assert distance == pytest.approx(1.0 - similarity, abs=1e-6)


class TestEmbedderConfiguration:
    """Test embedder configuration."""

    def test_embedding_model_config(self):
        """Verify embedding model is configured correctly."""
        assert settings.EMBEDDING_MODEL == "text-embedding-3-large"

    def test_embedding_dimension_config(self):
        """Verify embedding dimension is 1536 (GPT-5.1 upgrade)."""
        assert settings.EMBEDDING_DIM == 1536


class TestEmbedderMockBehavior:
    """Test mock behavior when API is unavailable."""

    @pytest.mark.asyncio
    async def test_encode_uses_mock_without_api(self):
        """When API unavailable, should use mock embedding."""
        embedder = Embedder()

        if not embedder.has_api:
            result = await embedder.encode("Test text")

            # Should still return valid embedding
            assert len(result) == settings.EMBEDDING_DIM
            # Values should be in mock range
            assert all(-0.1 <= v <= 0.1 for v in result)

    @pytest.mark.asyncio
    async def test_batch_encode_uses_mock_without_api(self):
        """When API unavailable, batch should use mock embeddings."""
        embedder = Embedder()

        if not embedder.has_api:
            texts = ["Text 1", "Text 2"]
            result = await embedder.encode_batch(texts)

            assert len(result) == 2
            for vec in result:
                assert len(vec) == settings.EMBEDDING_DIM


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
