"""
FastAPI Routes - Janus 3.5

Endpoints for the Janus3 Memory System API.

Simplified for Janus 3.5:
- Redis Streams for STM
- PostgreSQL for LTM (episodes, entities, facts)
- OpenTelemetry tracing on all endpoints
"""

import logging
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, status

from janus_core.config import settings
from janus_core.models import (
    ChatRequest,
    ChatResponse,
    CreateSessionResponse,
    RetrievalStats,
    SessionInfo,
)
from janus_core.retrieval import ContextBuilder
from janus_core.db.connection import get_db_pool
from janus_core.db.episodes import EpisodeRepository
from janus_core.db.facts import FactRepository
from janus_core.tracing import tracer
from janus_core.topic_detector import Turn as TopicTurn

from janus_app.config import app_settings
from janus_app.infra.metrics import metrics

from .dependencies import (
    clear_threshold,
    get_agentic_handler,
    get_embedder,
    get_llm_service,
    get_redis,
    get_retriever,
    get_stm_manager,
    get_threshold,
    get_threshold_store,
    get_topic_detector,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/sessions", response_model=CreateSessionResponse, status_code=status.HTTP_201_CREATED
)
async def create_session(user_id: Optional[UUID] = None):
    """
    Create a new chat session.

    Args:
        user_id: Optional user identifier.

    Returns:
        Session ID and creation timestamp.
    """
    with tracer.start_as_current_span("create_session") as span:
        session_id = uuid4()
        created_at = datetime.utcnow()

        span.set_attribute("session_id", str(session_id))
        span.set_attribute("user_id", str(user_id) if user_id else "anonymous")

        # Create session in database
        db_pool = await get_db_pool()
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sessions (id, user_id, created_at, last_activity)
                VALUES ($1, $2, $3, $3)
            """,
                session_id,
                user_id,
                created_at,
            )

        # Initialize STM consumer group
        stm = await get_stm_manager()
        await stm.ensure_consumer_group(session_id)

        metrics.sessions_created.inc()

        return CreateSessionResponse(session_id=session_id, created_at=created_at)


@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: UUID):
    """
    Get session information.

    Args:
        session_id: Session UUID.

    Returns:
        Session info including episode and fact counts.
    """
    with tracer.start_as_current_span("get_session") as span:
        span.set_attribute("session_id", str(session_id))

        db_pool = await get_db_pool()

        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, created_at, last_activity, metadata
                FROM sessions WHERE id = $1
            """,
                session_id,
            )

            if not row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
                )

            # Get counts
            episode_count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM episodes WHERE session_id = $1
            """,
                session_id,
            )

            fact_count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM facts WHERE session_id = $1
            """,
                session_id,
            )

        span.set_attribute("episode_count", episode_count)
        span.set_attribute("fact_count", fact_count)

        return SessionInfo(
            id=row["id"],
            created_at=row["created_at"],
            last_activity=row["last_activity"],
            metadata=row["metadata"] or {},
            episode_count=episode_count,
            fact_count=fact_count,
        )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: UUID):
    """
    Delete a session and all associated data.

    GDPR-compliant: removes data from all stores.

    Args:
        session_id: Session UUID.
    """
    with tracer.start_as_current_span("delete_session") as span:
        span.set_attribute("session_id", str(session_id))

        db_pool = await get_db_pool()

        # Delete from database (cascades to episodes, facts, entities)
        async with db_pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM sessions WHERE id = $1
            """,
                session_id,
            )

            if "DELETE 0" in result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
                )

        # Clean up Redis STM stream
        stm = await get_stm_manager()
        await stm.delete_stream(session_id)

        # Clean up Redis threshold state
        threshold_store = await get_threshold_store()
        await threshold_store.delete(session_id)

        # Clear local threshold cache
        clear_threshold(session_id)

        # Audit log
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO audit_log (event_type, session_id, details)
                VALUES ('GDPR_ERASURE', $1, $2)
            """,
                session_id,
                {"source": "api", "timestamp": datetime.utcnow().isoformat()},
            )

        metrics.sessions_deleted.inc()


@router.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(session_id: UUID, request: ChatRequest):
    """
    Process a chat message with memory retrieval.

    Implements the full chat flow:
    1. Embed user query
    2. Store in STM (Redis Streams)
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
    with tracer.start_as_current_span("chat") as span:
        span.set_attribute("session_id", str(session_id))
        span.set_attribute("content_length", len(request.content))

        # Start timing
        start_time = datetime.utcnow()

        # Verify session exists
        db_pool = await get_db_pool()
        async with db_pool.acquire() as conn:
            exists = await conn.fetchval(
                """
                SELECT EXISTS(SELECT 1 FROM sessions WHERE id = $1)
            """,
                session_id,
            )

            if not exists:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
                )

            # Update last activity
            await conn.execute(
                """
                UPDATE sessions SET last_activity = NOW() WHERE id = $1
            """,
                session_id,
            )

        # Get services
        embedder = get_embedder()
        retriever = await get_retriever()
        llm = get_llm_service()
        stm = await get_stm_manager()
        topic_detector = get_topic_detector()

        # 1. Embed user query
        with tracer.start_as_current_span("embed_query"):
            query_vec = await embedder.encode(request.content)

        # 2. Get current turn count and recent turns for topic detection
        stm_length = await stm.get_stream_length(session_id)
        turn_number = stm_length + 1

        # Get recent turns for topic detection
        recent_turns = await stm.get_recent_turns(session_id, count=settings.TOPIC_DETECTION_CONTEXT_TURNS)

        # 3. Store user message in STM
        with tracer.start_as_current_span("store_user_turn"):
            await stm.add_turn(
                session_id=session_id,
                role="user",
                content=request.content,
                vector=query_vec,
                turn_index=turn_number,
            )

        # 4. Detect topic boundary (LLM or fallback to semantic velocity)
        with tracer.start_as_current_span("check_topic_boundary"):
            if settings.TOPIC_DETECTION_ENABLED:
                # Convert STM turns to TopicTurn format
                topic_turns = [
                    TopicTurn(role=t.role, content=t.content)
                    for t in recent_turns
                ]
                boundary_result = await topic_detector.detect_boundary(
                    topic_turns, request.content
                )
                is_boundary = boundary_result.is_boundary
                span.set_attribute("topic_label", boundary_result.topic_label)
                span.set_attribute("topic_confidence", boundary_result.confidence)
            else:
                # Fallback to semantic velocity
                threshold = get_threshold(session_id)
                velocity, is_boundary = threshold.update(query_vec)
                span.set_attribute("velocity", velocity)
                metrics.threshold_velocities.observe(velocity)

            span.set_attribute("is_boundary", is_boundary)

        # If boundary detected, signal consolidation via pub/sub
        if is_boundary:
            metrics.topic_boundaries_detected.inc()
            redis_client = await get_redis()
            await redis_client.publish(
                f"consolidation:{session_id}", "boundary_detected"
            )

        # 5. Retrieve context (parallelized)
        with tracer.start_as_current_span("retrieve_context"):
            context = await retriever.build_context(
                query=request.content, query_vec=query_vec, session_id=session_id
            )
            span.set_attribute("entities_found", context.stats.entities_found)
            span.set_attribute("episodes_retrieved", context.stats.episodes_retrieved)

        # 6. Build system prompt with context
        system_prompt = ContextBuilder.build_system_prompt(context)

        # 7. Generate response (with optional web search via tool calling)
        with tracer.start_as_current_span("generate_response"):
            tool_calls_made = 0

            if app_settings.TAVILY_ENABLED and app_settings.TAVILY_API_KEY:
                # Use agentic handler with web search tools
                agentic_handler = get_agentic_handler()
                agentic_response = await agentic_handler.chat(
                    system_prompt=system_prompt,
                    user_message=request.content
                )
                response_text = agentic_response.content
                tool_calls_made = agentic_response.tool_calls_made

                if tool_calls_made > 0:
                    span.set_attribute("tools_used", ",".join(agentic_response.tools_used))
            else:
                # Fallback to simple chat (no tools)
                response_text = await llm.chat(
                    system_prompt=system_prompt,
                    user_message=request.content
                )

            span.set_attribute("response_length", len(response_text))
            span.set_attribute("tool_calls_made", tool_calls_made)

        # 8. Store response in STM
        with tracer.start_as_current_span("store_assistant_turn"):
            response_vec = await embedder.encode(response_text)
            await stm.add_turn(
                session_id=session_id,
                role="assistant",
                content=response_text,
                vector=response_vec,
                turn_index=turn_number + 1,
            )

        # Record metrics
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        metrics.chat_request_duration.observe(duration_ms / 1000)
        metrics.chat_requests_total.labels(status="success").inc()

        return ChatResponse(
            user_message=request.content,
            assistant_message=response_text,
            session_id=session_id,
            turn_number=turn_number,
            retrieval_stats=context.stats,
        )


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
    with tracer.start_as_current_span("get_stm") as span:
        span.set_attribute("session_id", str(session_id))

        stm = await get_stm_manager()
        turns = await stm.get_recent_turns(session_id, count=limit)

        return {
            "session_id": session_id,
            "turn_count": len(turns),
            "turns": [
                {
                    "id": t.id,
                    "role": t.role,
                    "content": t.content[:200] + "..."
                    if len(t.content) > 200
                    else t.content,
                    "timestamp": t.timestamp.isoformat(),
                }
                for t in turns
            ],
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
    with tracer.start_as_current_span("get_episodes") as span:
        span.set_attribute("session_id", str(session_id))

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
                    "created_at": e.created_at.isoformat(),
                }
                for e in episodes
            ],
        }


@router.get("/sessions/{session_id}/facts")
async def get_facts(session_id: UUID, limit: int = 50):
    """
    Get facts for a session.

    Args:
        session_id: Session UUID.
        limit: Maximum facts to return.

    Returns:
        List of facts.
    """
    with tracer.start_as_current_span("get_facts") as span:
        span.set_attribute("session_id", str(session_id))

        db_pool = await get_db_pool()
        fact_repo = FactRepository(db_pool)
        facts = await fact_repo.get_all_facts(session_id, limit=limit)

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
                }
                for f in facts
            ],
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
    with tracer.start_as_current_span("get_threshold_stats") as span:
        span.set_attribute("session_id", str(session_id))

        # Try Redis-backed store first
        threshold_store = await get_threshold_store()
        redis_state = await threshold_store.get(session_id)

        # Get local cache stats
        threshold = get_threshold(session_id)
        local_stats = threshold.get_stats()

        return {
            "session_id": session_id,
            "local": local_stats,
            "redis": redis_state,
        }


@router.get("/health")
async def health_check():
    """
    Health check endpoint for load balancers.
    """
    checks = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "3.5",
        "checks": {},
    }

    # Check database
    try:
        db_pool = await get_db_pool()
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        checks["checks"]["database"] = "ok"
    except Exception as e:
        checks["checks"]["database"] = f"error: {str(e)}"
        checks["status"] = "degraded"

    # Check Redis
    try:
        redis_client = await get_redis()
        await redis_client.ping()
        checks["checks"]["redis"] = "ok"
    except Exception as e:
        checks["checks"]["redis"] = f"error: {str(e)}"
        checks["status"] = "degraded"

    return checks


@router.get("/metrics")
async def get_metrics():
    """
    Prometheus metrics endpoint.
    """
    return metrics.generate_metrics()


@router.get("/config")
async def get_config():
    """
    Get current configuration (non-sensitive values).
    """
    return {
        "version": "4.0",
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_dim": settings.EMBEDDING_DIM,
        "extraction_model": settings.EXTRACTION_MODEL,
        "min_turns_for_consolidation": settings.MIN_TURNS_FOR_CONSOLIDATION,
        "max_buffer_size": settings.MAX_BUFFER_SIZE,
    }
