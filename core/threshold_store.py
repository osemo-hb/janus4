"""
Atomic Redis Threshold Store

Provides atomic updates to adaptive threshold state using Redis Lua scripts.
This enables horizontal scaling of API instances while maintaining consistent
threshold state per session.

The threshold state consists of:
- velocities: Rolling window of semantic velocity values
- last_vector: The last user embedding for computing new velocities

Usage:
    store = ThresholdStore(redis_client)

    # Atomic update
    velocities = await store.update_atomic(
        session_id=session_id,
        velocity=0.45,
        last_vector=embedding
    )

    # Load state
    velocities, last_vector = await store.load_state(session_id)
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import UUID

import redis.asyncio as redis

from janus3.config import settings
from janus3.infra.metrics import metrics

logger = logging.getLogger(__name__)

# Load Lua script from file
_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "threshold_update.lua"


class ThresholdStore:
    """
    Redis-backed store for adaptive threshold state.

    Uses Lua scripting for atomic updates to ensure consistency
    across multiple API instances.
    """

    KEY_PREFIX = "threshold"

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self._update_script: Optional[redis.client.Script] = None
        self._script_loaded = False

    async def _ensure_script_loaded(self) -> None:
        """Load and register the Lua script if not already done."""
        if self._script_loaded:
            return

        try:
            # Read Lua script from file
            if _SCRIPT_PATH.exists():
                lua_code = _SCRIPT_PATH.read_text()
            else:
                # Fallback inline script if file not found
                lua_code = self._get_inline_script()
                logger.warning("Using inline Lua script (file not found)")

            # Register script with Redis
            self._update_script = self.redis.register_script(lua_code)
            self._script_loaded = True
            logger.debug("Threshold update Lua script registered")

        except Exception as e:
            logger.error(f"Failed to load Lua script: {e}")
            raise

    def _get_inline_script(self) -> str:
        """Fallback inline Lua script."""
        return """
        local base_key = KEYS[1]
        local velocity = tonumber(ARGV[1])
        local last_vector = ARGV[2]
        local window_size = tonumber(ARGV[3])
        local ttl = tonumber(ARGV[4])

        local velocities_key = base_key .. ':velocities'
        local last_vector_key = base_key .. ':last_vector'

        redis.call('RPUSH', velocities_key, velocity)
        redis.call('LTRIM', velocities_key, -window_size, -1)
        redis.call('SET', last_vector_key, last_vector)
        redis.call('EXPIRE', velocities_key, ttl)
        redis.call('EXPIRE', last_vector_key, ttl)

        return redis.call('LRANGE', velocities_key, 0, -1)
        """

    def _make_key(self, session_id: UUID) -> str:
        """Generate the base Redis key for a session's threshold state."""
        return f"{self.KEY_PREFIX}:{session_id}"

    async def update_atomic(
        self,
        session_id: UUID,
        velocity: float,
        last_vector: List[float],
    ) -> List[float]:
        """
        Atomically update threshold state and return current velocities.

        This operation is atomic via Lua scripting, ensuring consistency
        even with concurrent updates from multiple API instances.

        Args:
            session_id: The session UUID
            velocity: The new semantic velocity value
            last_vector: The new last embedding vector

        Returns:
            List of current velocities after the update
        """
        await self._ensure_script_loaded()

        base_key = self._make_key(session_id)

        try:
            # Execute atomic Lua script
            result = await self._update_script(
                keys=[base_key],
                args=[
                    str(velocity),
                    json.dumps(last_vector),
                    str(settings.ADAPTIVE_THRESHOLD_WINDOW),
                    str(settings.THRESHOLD_REDIS_TTL),
                ],
            )

            # Convert result to float list
            velocities = [float(v) for v in result]

            # Update metrics
            metrics.threshold_updates.labels(result="updated").inc()
            metrics.threshold_velocity.observe(velocity)

            logger.debug(
                f"Threshold updated for session {session_id}: "
                f"velocity={velocity:.3f}, buffer_size={len(velocities)}"
            )

            return velocities

        except redis.RedisError as e:
            logger.error(f"Redis error updating threshold: {e}")
            metrics.threshold_updates.labels(result="error").inc()
            raise

    async def load_state(
        self,
        session_id: UUID,
    ) -> Tuple[List[float], Optional[List[float]]]:
        """
        Load threshold state from Redis.

        Returns cold-start values if no state exists for this session.

        Args:
            session_id: The session UUID

        Returns:
            Tuple of (velocities, last_vector)
        """
        base_key = self._make_key(session_id)
        velocities_key = f"{base_key}:velocities"
        last_vector_key = f"{base_key}:last_vector"

        try:
            # Use pipeline for efficiency
            async with self.redis.pipeline() as pipe:
                pipe.lrange(velocities_key, 0, -1)
                pipe.get(last_vector_key)
                results = await pipe.execute()

            velocities_raw, last_vector_raw = results

            # Parse velocities
            if velocities_raw:
                velocities = [float(v) for v in velocities_raw]
            else:
                # Cold start: use pre-calculated global averages
                velocities = list(settings.COLD_START_VELOCITIES)
                logger.debug(f"Using cold start velocities for session {session_id}")

            # Parse last vector
            last_vector = None
            if last_vector_raw:
                last_vector = json.loads(last_vector_raw)

            return velocities, last_vector

        except redis.RedisError as e:
            logger.error(f"Redis error loading threshold state: {e}")
            # Return cold start values on error
            return list(settings.COLD_START_VELOCITIES), None

    async def delete_state(self, session_id: UUID) -> None:
        """
        Delete threshold state for a session.

        Called during session cleanup or GDPR erasure.

        Args:
            session_id: The session UUID
        """
        base_key = self._make_key(session_id)

        try:
            # Delete all keys for this session
            await self.redis.delete(
                f"{base_key}:velocities",
                f"{base_key}:last_vector",
            )
            logger.debug(f"Threshold state deleted for session {session_id}")

        except redis.RedisError as e:
            logger.error(f"Redis error deleting threshold state: {e}")
            raise

    async def get_velocity_stats(
        self,
        session_id: UUID,
    ) -> dict:
        """
        Get threshold statistics for debugging/monitoring.

        Args:
            session_id: The session UUID

        Returns:
            Dictionary with velocity statistics
        """
        velocities, last_vector = await self.load_state(session_id)

        if not velocities:
            return {
                "session_id": str(session_id),
                "velocities_count": 0,
                "has_last_vector": False,
                "is_cold_start": True,
            }

        import statistics

        return {
            "session_id": str(session_id),
            "velocities_count": len(velocities),
            "mean_velocity": statistics.mean(velocities),
            "std_velocity": statistics.stdev(velocities) if len(velocities) > 1 else 0.0,
            "min_velocity": min(velocities),
            "max_velocity": max(velocities),
            "has_last_vector": last_vector is not None,
            "is_cold_start": velocities == list(settings.COLD_START_VELOCITIES),
        }


# Global store instance (initialized with Redis client on first use)
_threshold_store: Optional[ThresholdStore] = None


async def get_threshold_store(redis_client: redis.Redis) -> ThresholdStore:
    """
    Get or create the global threshold store instance.

    Args:
        redis_client: Redis client instance

    Returns:
        ThresholdStore instance
    """
    global _threshold_store

    if _threshold_store is None:
        _threshold_store = ThresholdStore(redis_client)

    return _threshold_store
