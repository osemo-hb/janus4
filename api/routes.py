"""
FastAPI Routes

Endpoints for the Janus3 Memory System API.
"""

from datetime import datetime
from uuid import UUID, uuid4
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
import asyncpg

from core.models import (
    ChatRequest, ChatResponse, CreateSessionResponse,
    SessionInfo, RetrievalStats
)
from core.embedder import Embedder
from core.retrieval import HybridRetriever, ContextBuilder
from core.threshold import AdaptiveThreshold
from db.stm import STMManager
from db.connection import get_db_pool
from db.episodes import EpisodeRepository
from db.facts import FactRepository
from services.llm_service import LLMService
from api.dependencies import (
    get_embedder, get_retriever, get_stm_manager,
    get_llm_service, get_redis, get_threshold, clear_threshold
)

router = APIRouter()


# ============================================
# Session Endpoints
# ============================================

@router.post("/sessions", response_model=CreateSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    user_id: Optional[UUID] = None
):
    """
    Create a new chat session.

    Args:
        user_id: Optional user identifier.

    Returns:
        Session ID and creation timestamp.
    """
    session_id = uuid4()
    created_at = datetime.utcnow()

    # Create session in database
    db_pool = await get_db_pool()
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO sessions (id, user_id, created_at, last_activity)
            VALUES ($1, $2, $3, $3)
        """, session_id, user_id, created_at)

    # Initialize STM consumer group
    stm = await get_stm_manager()
    await stm.ensure_consumer_group(session_id)

    return CreateSessionResponse(
        session_id=session_id,
        created_at=created_at
    )


@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: UUID):
    """
    Get session information.

    Args:
        session_id: Session UUID.

    Returns:
        Session info including episode and fact counts.
    """
    db_pool = await get_db_pool()

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT id, created_at, last_activity, metadata
            FROM sessions WHERE id = $1
        """, session_id)

        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )

        # Get counts
        episode_count = await conn.fetchval("""
            SELECT COUNT(*) FROM episodes WHERE session_id = $1
        """, session_id)

        fact_count = await conn.fetchval("""
            SELECT COUNT(*) FROM facts WHERE session_id = $1 AND is_current = TRUE
        """, session_id)

    return SessionInfo(
        id=row["id"],
        created_at=row["created_at"],
        last_activity=row["last_activity"],
        metadata=row["metadata"] or {},
        episode_count=episode_count,
        fact_count=fact_count
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: UUID):
    """
    Delete a session and all associated data.

    Args:
        session_id: Session UUID.
    """
    db_pool = await get_db_pool()

    # Delete from database (cascades to episodes, facts, entities)
    async with db_pool.acquire() as conn:
        result = await conn.execute("""
            DELETE FROM sessions WHERE id = $1
        """, session_id)

        if "DELETE 0" in result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )

    # Clean up Redis STM stream
    stm = await get_stm_manager()
    await stm.delete_stream(session_id)

    # Clear threshold cache
    clear_threshold(session_id)


# ============================================
# Chat Endpoint
# ============================================

@router.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(
    session_id: UUID,
    request: ChatRequest
):
    """
    Process a chat message with memory retrieval.

    Implements the full chat flow:
    1. Embed user query
    2. Store in STM
    3. Check for topic boundary
    4. Retrieve context (parallel)
    5. Generate response
    6. Store response in STM

    Args:
        session_id: Session UUID.
        request: Chat request with user message.

    Returns:
        Chat response with assistant message and stats.
    """
    # Verify session exists
    db_pool = await get_db_pool()
    async with db_pool.acquire() as conn:
        exists = await conn.fetchval("""
            SELECT EXISTS(SELECT 1 FROM sessions WHERE id = $1)
        """, session_id)

        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )

        # Update last activity
        await conn.execute("""
            UPDATE sessions SET last_activity = NOW() WHERE id = $1
        """, session_id)

    # Get services
    embedder = get_embedder()
    retriever = await get_retriever()
    stm = await get_stm_manager()
    llm = get_llm_service()
    threshold = get_threshold(session_id)

    # 1. Embed user query
    query_vec = await embedder.encode(request.content)

    # 2. Get current turn count
    stm_length = await stm.get_stream_length(session_id)
    turn_number = stm_length + 1

    # 3. Store user message in STM
    await stm.add_turn(
        session_id=session_id,
        role="user",
        content=request.content,
        vector=query_vec,
        turn_index=turn_number
    )

    # 4. Check adaptive threshold for topic boundary
    velocity, is_boundary = threshold.update(query_vec)

    # If boundary detected, signal consolidation (handled by consumer)
    if is_boundary:
        redis_client = await get_redis()
        await redis_client.publish(
            f"consolidation:{session_id}",
            "boundary_detected"
        )

    # 5. Retrieve context (Fix 2: parallelized)
    context = await retriever.build_context(
        query=request.content,
        query_vec=query_vec,
        session_id=session_id
    )

    # 6. Build system prompt with context
    system_prompt = ContextBuilder.build_system_prompt(context)

    # 7. Generate response
    response_text = await llm.chat(
        system_prompt=system_prompt,
        user_message=request.content
    )

    # 8. Embed and store response in STM
    response_vec = await embedder.encode(response_text)
    await stm.add_turn(
        session_id=session_id,
        role="assistant",
        content=response_text,
        vector=response_vec,
        turn_index=turn_number + 1
    )

    return ChatResponse(
        user_message=request.content,
        assistant_message=response_text,
        session_id=session_id,
        turn_number=turn_number,
        retrieval_stats=context.stats
    )


# ============================================
# Debug/Admin Endpoints
# ============================================

@router.get("/sessions/{session_id}/stm")
async def get_stm(session_id: UUID, limit: int = 20):
    """
    Get recent STM turns for debugging.

    Args:
        session_id: Session UUID.
        limit: Maximum turns to return.

    Returns:
        List of recent turns.
    """
    stm = await get_stm_manager()
    turns = await stm.get_recent_turns(session_id, count=limit)

    return {
        "session_id": session_id,
        "turn_count": len(turns),
        "turns": [
            {
                "id": t.id,
                "role": t.role,
                "content": t.content[:200] + "..." if len(t.content) > 200 else t.content,
                "timestamp": t.timestamp.isoformat()
            }
            for t in turns
        ]
    }


@router.get("/sessions/{session_id}/episodes")
async def get_episodes(session_id: UUID, limit: int = 20):
    """
    Get episodes for a session.

    Args:
        session_id: Session UUID.
        limit: Maximum episodes to return.

    Returns:
        List of episodes.
    """
    db_pool = await get_db_pool()
    episode_repo = EpisodeRepository(db_pool)

    episodes = await episode_repo.get_by_session(session_id, limit=limit)

    return {
        "session_id": session_id,
        "episode_count": len(episodes),
        "episodes": [
            {
                "id": str(e.id),
                "summary": e.summary,
                "turn_range": f"{e.turn_start}-{e.turn_end}",
                "created_at": e.created_at.isoformat()
            }
            for e in episodes
        ]
    }


@router.get("/sessions/{session_id}/facts")
async def get_facts(session_id: UUID, limit: int = 50):
    """
    Get current facts for a session.

    Args:
        session_id: Session UUID.
        limit: Maximum facts to return.

    Returns:
        List of current facts.
    """
    db_pool = await get_db_pool()
    fact_repo = FactRepository(db_pool)

    facts = await fact_repo.get_all_current_facts(session_id, limit=limit)

    return {
        "session_id": session_id,
        "fact_count": len(facts),
        "facts": [
            {
                "id": str(f.id),
                "subject_entity_id": str(f.subject_entity_id),
                "predicate": f.predicate,
                "object": f.object,
                "confidence": f.confidence,
                "version": f.version
            }
            for f in facts
        ]
    }


@router.get("/sessions/{session_id}/threshold")
async def get_threshold_stats(session_id: UUID):
    """
    Get adaptive threshold statistics for debugging.

    Args:
        session_id: Session UUID.

    Returns:
        Threshold statistics.
    """
    threshold = get_threshold(session_id)
    return {
        "session_id": session_id,
        **threshold.get_stats()
    }
