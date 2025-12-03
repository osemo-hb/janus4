"""
Entity Linker using GLiNER

Extracts named entities from text using GLiNER model.
Note: Entity aliases skipped per user decision - will rely on pgvector fallback.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional
from dataclasses import dataclass

from config import settings


@dataclass
class ExtractedEntity:
    """Entity extracted from text."""
    text: str
    label: str  # Entity type (person, location, etc.)
    score: float
    start: int  # Character offset
    end: int


class EntityLinker:
    """
    Entity extraction using GLiNER.

    Uses singleton pattern for model loading.
    Runs extraction in thread pool to avoid blocking.
    """

    _instance: Optional["EntityLinker"] = None
    _executor = ThreadPoolExecutor(max_workers=1)
    _model = None
    _initialized = False

    def __new__(cls) -> "EntityLinker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Lazy initialization - model loaded on first use
        pass

    def _ensure_model(self):
        """Load model if not already loaded."""
        if not EntityLinker._initialized:
            print(f"Loading GLiNER model: {settings.GLINER_MODEL}")
            try:
                from gliner import GLiNER
                EntityLinker._model = GLiNER.from_pretrained(settings.GLINER_MODEL)
                EntityLinker._initialized = True
                print("GLiNER model loaded.")
            except Exception as e:
                print(f"Warning: Failed to load GLiNER model: {e}")
                print("Entity extraction will be disabled.")
                EntityLinker._initialized = True  # Mark as initialized to avoid retries

    @property
    def model(self):
        """Get the GLiNER model."""
        self._ensure_model()
        return EntityLinker._model

    def _extract_sync(self, text: str) -> List[ExtractedEntity]:
        """Synchronous entity extraction (runs in thread pool)."""
        self._ensure_model()

        if EntityLinker._model is None:
            return []

        try:
            entities = self.model.predict_entities(
                text,
                settings.GLINER_LABELS
            )

            return [
                ExtractedEntity(
                    text=e["text"],
                    label=e["label"],
                    score=e["score"],
                    start=e["start"],
                    end=e["end"]
                )
                for e in entities
            ]
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return []

    async def extract(self, text: str) -> List[ExtractedEntity]:
        """
        Async entity extraction.

        Args:
            text: Text to extract entities from.

        Returns:
            List of ExtractedEntity objects.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._extract_sync,
            text
        )

    async def extract_batch(self, texts: List[str]) -> List[List[ExtractedEntity]]:
        """
        Async batch entity extraction.

        Args:
            texts: List of texts to extract entities from.

        Returns:
            List of entity lists, one per input text.
        """
        tasks = [self.extract(text) for text in texts]
        return await asyncio.gather(*tasks)

    @staticmethod
    def normalize(entity_text: str) -> str:
        """
        Normalize entity text for deduplication.

        Simple normalization: lowercase and strip whitespace.
        Production would use lemmatization.

        Args:
            entity_text: Raw entity text.

        Returns:
            Normalized entity text.
        """
        return entity_text.lower().strip()

    @staticmethod
    def get_entity_types() -> List[str]:
        """Get the list of entity types we extract."""
        return list(settings.GLINER_LABELS)
