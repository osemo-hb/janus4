"""
Embedding Service

Async wrapper around SentenceTransformer for thread-safe embedding generation.
Uses ThreadPoolExecutor to avoid blocking the async event loop.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
import numpy as np

from sentence_transformers import SentenceTransformer

from config import settings


class Embedder:
    """
    Thread-safe async embedding service.

    Uses singleton pattern to ensure model is loaded only once.
    ThreadPoolExecutor offloads CPU-bound embedding to separate threads.
    """

    _instance: Optional["Embedder"] = None
    _executor = ThreadPoolExecutor(max_workers=2)
    _model: Optional[SentenceTransformer] = None

    def __new__(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only load model once
        if Embedder._model is None:
            print(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            Embedder._model = SentenceTransformer(settings.EMBEDDING_MODEL)
            print("Embedding model loaded.")

    @property
    def model(self) -> SentenceTransformer:
        """Get the SentenceTransformer model."""
        if Embedder._model is None:
            raise RuntimeError("Model not loaded")
        return Embedder._model

    def _encode_sync(self, text: str) -> List[float]:
        """Synchronous encoding (runs in thread pool)."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _encode_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous batch encoding (runs in thread pool)."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    async def encode(self, text: str) -> List[float]:
        """
        Async embedding generation for a single text.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._encode_sync,
            text
        )

    async def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Async batch embedding generation.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._encode_batch_sync,
            texts
        )

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
