"""
Pydantic Models for Janus3

Type-safe data structures for the memory system.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import UUID
from datetime import datetime
from enum import Enum


# ============================================
# Enums
# ============================================

class VectorType(str, Enum):
    """Types of vectors stored in episode_vectors table."""
    SUMMARY = "summary"
    CENTROID = "centroid"
    USER_TURN = "user_turn"


class FactStatus(str, Enum):
    """Status of facts in the knowledge graph."""
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CONFLICTED = "conflicted"
    RETRACTED = "retracted"


# ============================================
# Core Domain Models
# ============================================

class Turn(BaseModel):
    """A single interaction in the Short Term Memory (Redis Stream)."""
    id: str  # Redis stream entry ID (e.g., "1234567890123-0")
    session_id: UUID
    role: str  # 'user' or 'assistant'
    content: str
    vector: List[float]  # 384-dim embedding
    timestamp: datetime

    class Config:
        frozen = True


class Entity(BaseModel):
    """Knowledge graph entity with embedding."""
    id: UUID
    session_id: UUID
    name: str
    normalized_name: str
    entity_type: str
    embedding: List[float]
    created_at: datetime

    class Config:
        frozen = True


class EpisodeVector(BaseModel):
    """Individual vector associated with an episode."""
    id: UUID
    episode_id: UUID
    vector_type: VectorType
    turn_index: Optional[int] = None
    embedding: List[float]


class Episode(BaseModel):
    """Consolidated LTM episode (compressed conversation segment)."""
    id: UUID
    session_id: UUID
    summary: str
    turn_start: int
    turn_end: int
    created_at: datetime
    vectors: List[EpisodeVector] = Field(default_factory=list)

    class Config:
        frozen = True


class Fact(BaseModel):
    """Versioned fact in the knowledge graph."""
    id: UUID
    session_id: UUID
    subject_entity_id: UUID
    predicate: str
    object: str
    version: int
    confidence: float = 0.9
    is_current: bool = True
    source_episode_id: Optional[UUID] = None
    created_at: datetime

    class Config:
        frozen = True


class Conflict(BaseModel):
    """Conflict between two facts."""
    id: UUID
    session_id: UUID
    fact_id_1: UUID
    fact_id_2: UUID
    resolution_strategy: str = "highest_confidence"
    resolved: bool = False
    resolved_value: Optional[str] = None
    created_at: datetime


# ============================================
# API Request/Response Models
# ============================================

class CreateSessionResponse(BaseModel):
    """Response for session creation."""
    session_id: UUID
    created_at: datetime


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    content: str = Field(..., min_length=1, max_length=10000)


class RetrievalStats(BaseModel):
    """Statistics about context retrieval."""
    entities_found: int
    episodes_retrieved: int
    facts_retrieved: int
    stm_turns: int
    retrieval_time_ms: float


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""
    user_message: str
    assistant_message: str
    session_id: UUID
    turn_number: int
    retrieval_stats: RetrievalStats


class SessionInfo(BaseModel):
    """Session information."""
    id: UUID
    created_at: datetime
    last_activity: datetime
    metadata: dict = Field(default_factory=dict)
    episode_count: int = 0
    fact_count: int = 0


# ============================================
# Internal Models (for worker/consumer)
# ============================================

class ConsolidationResult(BaseModel):
    """Result of memory consolidation."""
    episode_id: UUID
    summary: str
    facts_extracted: int
    entities_created: int
    turns_processed: int


class ExtractedFact(BaseModel):
    """Fact extracted by LLM during consolidation."""
    subject: str
    predicate: str
    object: str
    confidence: float = 0.9
