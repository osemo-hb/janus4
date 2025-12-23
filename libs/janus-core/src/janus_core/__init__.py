"""
Janus Core - Conversational memory system library.

This package provides the core functionality for building conversational memory
systems with LLM applications, including:
- Embedding generation (OpenAI)
- Entity and fact extraction
- Topic boundary detection
- Memory consolidation and retrieval
- Database repositories for PostgreSQL + Redis
"""

from janus_core.config import settings
from janus_core.models import (
    Episode,
    Entity,
    Fact,
    ConsolidationResult,
)
from janus_core.distillation import DistilledMemory, MemoryDistiller
from janus_core.embedder import Embedder
from janus_core.unified_extractor import UnifiedExtractor
from janus_core.topic_detector import TopicDetector
from janus_core.retrieval import HybridRetriever
from janus_core.reranker import LLMReranker
from janus_core.consolidation import ConsolidationService

__version__ = "4.0.0"

__all__ = [
    # Config
    "settings",
    # Models
    "Episode",
    "Entity",
    "Fact",
    "ConsolidationResult",
    "DistilledMemory",
    # Services
    "Embedder",
    "UnifiedExtractor",
    "TopicDetector",
    "HybridRetriever",
    "LLMReranker",
    "MemoryDistiller",
    "ConsolidationService",
]
