"""
Embedding Service

Async OpenAI embedding service using text-embedding-3-large.
"""

import random
from typing import List, Optional

import numpy as np
from openai import AsyncOpenAI

from janus_core.config import settings
from janus_core.patterns import singleton


@singleton
class Embedder:
    """Async OpenAI embedding service with configurable dimensions."""

    _client: Optional[AsyncOpenAI] = None

    def __init__(self):
        if Embedder._client is None and settings.OPENAI_API_KEY:
            Embedder._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    @property
    def client(self) -> Optional[AsyncOpenAI]:
        """Get the OpenAI client."""
        return Embedder._client

    @property
    def has_api(self) -> bool:
        """Check if API is available."""
        return Embedder._client is not None

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return settings.EMBEDDING_DIM

    async def encode(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        if not self.has_api:
            return self._mock_embedding()

        response = await self.client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=text,
            dimensions=settings.EMBEDDING_DIM
        )
        return response.data[0].embedding

    async def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in a single API call.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors (same order as input).
        """
        if not texts:
            return []

        if not self.has_api:
            return [self._mock_embedding() for _ in texts]

        response = await self.client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=texts,
            dimensions=settings.EMBEDDING_DIM
        )

        # Sort by index to maintain input order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    def _mock_embedding(self) -> List[float]:
        """Generate mock embedding for testing when API unavailable."""
        # Generate deterministic-ish mock for consistency
        return [random.uniform(-0.1, 0.1) for _ in range(settings.EMBEDDING_DIM)]

    @staticmethod
    def cosine_distance(vec_a: List[float], vec_b: List[float]) -> float:
        """
        Calculate cosine distance between two vectors.

        Returns:
            Float between 0 (identical) and 2 (opposite).
        """
        a = np.array(vec_a)
        b = np.array(vec_b)

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 1.0  # Maximum distance for zero vectors

        similarity = dot_product / (norm_a * norm_b)
        return 1.0 - similarity  # Convert similarity to distance

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Returns:
            Float between -1 (opposite) and 1 (identical).
        """
        return 1.0 - Embedder.cosine_distance(vec_a, vec_b)
