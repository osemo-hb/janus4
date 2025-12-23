"""
Embedding Model Registry

Manages embedding model metadata and compatibility checking.
Ensures vectors are only compared when produced by compatible models.

This prevents "model soup" issues where vectors from different embedding
models are incorrectly compared, leading to poor retrieval quality.

Usage:
    from janus3.core.model_registry import ModelRegistry, get_current_model

    # Get current model
    model = get_current_model()

    # Check compatibility before comparison
    if ModelRegistry.can_compare(model_a_id, model_b_id):
        similarity = cosine_similarity(vec_a, vec_b)
    else:
        raise IncompatibleModelsError(...)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from janus_core.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingModel:
    """
    Embedding model specification.

    Immutable to prevent accidental modification.
    """

    model_id: str  # Unique identifier (e.g., "all-MiniLM-L6-v2")
    dimension: int  # Output vector dimension
    dtype: str  # Data type ("float32", "float16")
    version: str  # Model version for tracking
    description: str = ""  # Human-readable description

    # List of model IDs that produce compatible embeddings
    # Empty means only self-compatible
    compatible_with: tuple = field(default_factory=tuple)

    # Normalization (some models output normalized vectors)
    is_normalized: bool = True

    # Maximum input length (tokens or characters depending on model)
    max_input_length: int = 512

    def __hash__(self) -> int:
        return hash((self.model_id, self.version))


class IncompatibleModelsError(Exception):
    """Raised when attempting to compare vectors from incompatible models."""

    def __init__(self, model_a: str, model_b: str):
        self.model_a = model_a
        self.model_b = model_b
        super().__init__(
            f"Cannot compare vectors from incompatible models: "
            f"{model_a} and {model_b}"
        )


class ModelRegistry:
    """
    Registry of known embedding models and their compatibility.

    Singleton pattern ensures consistent model information across the system.
    """

    # Known embedding models
    _models: Dict[str, EmbeddingModel] = {
        "all-MiniLM-L6-v2": EmbeddingModel(
            model_id="all-MiniLM-L6-v2",
            dimension=384,
            dtype="float32",
            version="1.0.0",
            description="Sentence Transformers MiniLM (fast, good quality)",
            compatible_with=(),
            is_normalized=True,
            max_input_length=512,
        ),
        "all-mpnet-base-v2": EmbeddingModel(
            model_id="all-mpnet-base-v2",
            dimension=768,
            dtype="float32",
            version="1.0.0",
            description="Sentence Transformers MPNet (better quality, slower)",
            compatible_with=(),
            is_normalized=True,
            max_input_length=512,
        ),
        "text-embedding-ada-002": EmbeddingModel(
            model_id="text-embedding-ada-002",
            dimension=1536,
            dtype="float32",
            version="2.0.0",
            description="OpenAI Ada embedding model",
            compatible_with=(),
            is_normalized=True,
            max_input_length=8191,
        ),
        "text-embedding-3-small": EmbeddingModel(
            model_id="text-embedding-3-small",
            dimension=1536,
            dtype="float32",
            version="3.0.0",
            description="OpenAI v3 small embedding model",
            compatible_with=(),
            is_normalized=True,
            max_input_length=8191,
        ),
        "text-embedding-3-large": EmbeddingModel(
            model_id="text-embedding-3-large",
            dimension=3072,
            dtype="float32",
            version="3.0.0",
            description="OpenAI v3 large embedding model",
            compatible_with=(),
            is_normalized=True,
            max_input_length=8191,
        ),
    }

    @classmethod
    def get_model(cls, model_id: str) -> Optional[EmbeddingModel]:
        """
        Get model specification by ID.

        Args:
            model_id: The model identifier

        Returns:
            EmbeddingModel if found, None otherwise
        """
        return cls._models.get(model_id)

    @classmethod
    def get_model_or_raise(cls, model_id: str) -> EmbeddingModel:
        """
        Get model specification by ID, raising if not found.

        Args:
            model_id: The model identifier

        Returns:
            EmbeddingModel

        Raises:
            ValueError: If model not found
        """
        model = cls._models.get(model_id)
        if model is None:
            raise ValueError(f"Unknown embedding model: {model_id}")
        return model

    @classmethod
    def register_model(cls, model: EmbeddingModel) -> None:
        """
        Register a new embedding model.

        Args:
            model: The model specification to register
        """
        if model.model_id in cls._models:
            logger.warning(f"Overwriting existing model: {model.model_id}")

        cls._models[model.model_id] = model
        logger.info(f"Registered embedding model: {model.model_id}")

    @classmethod
    def can_compare(cls, model_a: str, model_b: str) -> bool:
        """
        Check if two models produce comparable embeddings.

        Args:
            model_a: First model ID
            model_b: Second model ID

        Returns:
            True if vectors can be compared, False otherwise
        """
        # Same model is always comparable
        if model_a == model_b:
            return True

        # Check explicit compatibility
        model_a_spec = cls._models.get(model_a)
        model_b_spec = cls._models.get(model_b)

        if model_a_spec and model_b in model_a_spec.compatible_with:
            return True

        if model_b_spec and model_a in model_b_spec.compatible_with:
            return True

        return False

    @classmethod
    def assert_compatible(cls, model_a: str, model_b: str) -> None:
        """
        Assert that two models are compatible, raising if not.

        Args:
            model_a: First model ID
            model_b: Second model ID

        Raises:
            IncompatibleModelsError: If models are not compatible
        """
        if not cls.can_compare(model_a, model_b):
            raise IncompatibleModelsError(model_a, model_b)

    @classmethod
    def list_models(cls) -> List[str]:
        """Get list of all registered model IDs."""
        return list(cls._models.keys())

    @classmethod
    def get_dimension(cls, model_id: str) -> int:
        """
        Get the output dimension for a model.

        Args:
            model_id: The model identifier

        Returns:
            Output vector dimension

        Raises:
            ValueError: If model not found
        """
        model = cls.get_model_or_raise(model_id)
        return model.dimension


def get_current_model() -> EmbeddingModel:
    """
    Get the currently configured embedding model.

    Returns:
        The current EmbeddingModel specification
    """
    model_id = settings.EMBEDDING_MODEL_ID
    model = ModelRegistry.get_model(model_id)

    if model is None:
        # Create a default spec if not in registry
        logger.warning(f"Model {model_id} not in registry, using defaults")
        return EmbeddingModel(
            model_id=model_id,
            dimension=settings.EMBEDDING_DIM,
            dtype="float32",
            version=settings.EMBEDDING_MODEL_VERSION,
            description="Custom model",
        )

    return model


def get_model_id() -> str:
    """Get the current embedding model ID."""
    return settings.EMBEDDING_MODEL_ID


def get_model_version() -> str:
    """Get the current embedding model version."""
    return settings.EMBEDDING_MODEL_VERSION


def get_model_dimension() -> int:
    """Get the current embedding model output dimension."""
    model = get_current_model()
    return model.dimension


@dataclass
class VectorMetadata:
    """
    Metadata to store alongside vectors.

    Include this with every stored vector to enable model tracking.
    """

    model_id: str
    model_version: str
    dimension: int

    @classmethod
    def from_current(cls) -> "VectorMetadata":
        """Create metadata from current model settings."""
        model = get_current_model()
        return cls(
            model_id=model.model_id,
            model_version=model.version,
            dimension=model.dimension,
        )

    def is_compatible_with(self, other: "VectorMetadata") -> bool:
        """Check if this vector is compatible with another."""
        return ModelRegistry.can_compare(self.model_id, other.model_id)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "dimension": self.dimension,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VectorMetadata":
        """Create from dictionary."""
        return cls(
            model_id=data["model_id"],
            model_version=data["model_version"],
            dimension=data["dimension"],
        )
