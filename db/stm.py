"""
Short-Term Memory Manager using Redis Streams

Implements Fix 1: Redis Streams for concurrency-safe STM with crash recovery.
Uses XADD/XREADGROUP/XACK/XAUTOCLAIM pattern.
"""

import json
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from uuid import UUID

import redis.asyncio as redis

from config import settings
from core.models import Turn


class STMManager:
    """
    Redis Streams-based Short Term Memory.

    Fix 1: Uses XADD/XREADGROUP for crash recovery.
    - API: XADD to append turns
    - Consumer: XREADGROUP to read, XACK after processing
    - Crash Recovery: XAUTOCLAIM for abandoned messages
    """

    def __init__(self, redis_client: redis.Redis):
        """
        Initialize STM Manager.

        Args:
            redis_client: Async Redis client instance.
        """
        self.redis = redis_client
        self.consumer_group = settings.CORTEX_CONSUMER_GROUP
        self.max_len = settings.STM_MAX_LEN

    def _stream_key(self, session_id: UUID) -> str:
        """Get Redis stream key for a session."""
        return f"{settings.STM_STREAM_PREFIX}:{session_id}"

    async def ensure_consumer_group(self, session_id: UUID) -> bool:
        """
        Create consumer group if it doesn't exist.

        Args:
            session_id: Session UUID.

        Returns:
            True if group was created, False if it already existed.
        """
        stream_key = self._stream_key(session_id)
        try:
            await self.redis.xgroup_create(
                stream_key,
                self.consumer_group,
                id="0",  # Start from beginning
                mkstream=True  # Create stream if not exists
            )
            return True
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                # Group already exists - this is fine
                return False
            raise

    async def add_turn(
        self,
        session_id: UUID,
        role: str,
        content: str,
        vector: List[float],
        turn_index: Optional[int] = None
    ) -> str:
        """
        Add a turn to the STM stream.

        Args:
            session_id: Session UUID.
            role: 'user' or 'assistant'.
            content: Message content.
            vector: Embedding vector.
            turn_index: Optional turn number.

        Returns:
            Redis stream entry ID (e.g., "1234567890123-0").
        """
        stream_key = self._stream_key(session_id)

        entry_id = await self.redis.xadd(
            stream_key,
            {
                "role": role,
                "content": content,
                "vector": json.dumps(vector),
                "timestamp": datetime.utcnow().isoformat(),
                "turn_index": str(turn_index) if turn_index is not None else "",
            },
            maxlen=self.max_len,
            approximate=True  # Use ~ for efficiency
        )

        return entry_id

    async def get_recent_turns(
        self,
        session_id: UUID,
        count: int = 20
    ) -> List[Turn]:
        """
        Get most recent turns from stream (non-destructive read).

        Args:
            session_id: Session UUID.
            count: Maximum number of turns to retrieve.

        Returns:
            List of Turn objects in chronological order.
        """
        stream_key = self._stream_key(session_id)

        # XREVRANGE gets most recent entries
        entries = await self.redis.xrevrange(stream_key, count=count)

        turns = []
        for entry_id, fields in reversed(entries):  # Reverse to chronological
            turns.append(self._parse_turn(session_id, entry_id, fields))

        return turns

    async def get_all_turns(self, session_id: UUID) -> List[Turn]:
        """
        Get all turns from stream.

        Args:
            session_id: Session UUID.

        Returns:
            List of all Turn objects in chronological order.
        """
        stream_key = self._stream_key(session_id)

        entries = await self.redis.xrange(stream_key)

        return [
            self._parse_turn(session_id, entry_id, fields)
            for entry_id, fields in entries
        ]

    async def read_new_entries(
        self,
        session_id: UUID,
        consumer_name: str,
        count: int = 10,
        block_ms: int = 2000
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Consumer method: Read new entries using XREADGROUP.

        Args:
            session_id: Session UUID.
            consumer_name: Unique consumer identifier.
            count: Maximum entries to read.
            block_ms: Block timeout in milliseconds.

        Returns:
            List of (entry_id, fields) tuples.
        """
        stream_key = self._stream_key(session_id)

        result = await self.redis.xreadgroup(
            self.consumer_group,
            consumer_name,
            {stream_key: ">"},  # Only new messages
            count=count,
            block=block_ms
        )

        if not result:
            return []

        # Result format: [[stream_key, [(id, fields), ...]]]
        stream_data = result[0] if result else None
        if not stream_data or len(stream_data) < 2:
            return []

        return stream_data[1]

    async def acknowledge(
        self,
        session_id: UUID,
        entry_ids: List[str]
    ) -> int:
        """
        Acknowledge processed entries with XACK.

        Args:
            session_id: Session UUID.
            entry_ids: List of entry IDs to acknowledge.

        Returns:
            Number of entries acknowledged.
        """
        if not entry_ids:
            return 0

        stream_key = self._stream_key(session_id)
        return await self.redis.xack(
            stream_key,
            self.consumer_group,
            *entry_ids
        )

    async def get_pending_entries(
        self,
        session_id: UUID,
        count: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get entries in Pending Entry List (PEL).

        Args:
            session_id: Session UUID.
            count: Maximum entries to retrieve.

        Returns:
            List of pending entry info dicts.
        """
        stream_key = self._stream_key(session_id)

        try:
            pending = await self.redis.xpending_range(
                stream_key,
                self.consumer_group,
                min="-",
                max="+",
                count=count
            )
            return pending
        except redis.ResponseError:
            return []

    async def claim_abandoned(
        self,
        session_id: UUID,
        consumer_name: str,
        min_idle_ms: int = None
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Claim entries abandoned by crashed consumers using XAUTOCLAIM.

        Args:
            session_id: Session UUID.
            consumer_name: Consumer claiming the entries.
            min_idle_ms: Minimum idle time to claim (default from settings).

        Returns:
            List of (entry_id, fields) tuples for claimed entries.
        """
        if min_idle_ms is None:
            min_idle_ms = settings.CONSOLIDATION_IDLE_TIMEOUT_MS

        stream_key = self._stream_key(session_id)

        try:
            # XAUTOCLAIM atomically claims idle messages
            result = await self.redis.xautoclaim(
                stream_key,
                self.consumer_group,
                consumer_name,
                min_idle_time=min_idle_ms,
                start_id="0-0",
                count=10
            )

            # Result format: [next_start_id, [(id, fields), ...], [deleted_ids]]
            if result and len(result) >= 2:
                return result[1]
            return []

        except redis.ResponseError as e:
            # Group might not exist yet
            if "NOGROUP" in str(e):
                return []
            raise

    async def get_stream_length(self, session_id: UUID) -> int:
        """
        Get the current length of the stream.

        Args:
            session_id: Session UUID.

        Returns:
            Number of entries in the stream.
        """
        stream_key = self._stream_key(session_id)
        return await self.redis.xlen(stream_key)

    async def delete_stream(self, session_id: UUID) -> bool:
        """
        Delete the entire stream (cleanup on session delete).

        Args:
            session_id: Session UUID.

        Returns:
            True if stream was deleted.
        """
        stream_key = self._stream_key(session_id)
        result = await self.redis.delete(stream_key)
        return result > 0

    def _parse_turn(
        self,
        session_id: UUID,
        entry_id: str,
        fields: Dict[str, str]
    ) -> Turn:
        """Parse Redis stream entry into Turn object."""
        return Turn(
            id=entry_id,
            session_id=session_id,
            role=fields.get("role", "unknown"),
            content=fields.get("content", ""),
            vector=json.loads(fields.get("vector", "[]")),
            timestamp=datetime.fromisoformat(
                fields.get("timestamp", datetime.utcnow().isoformat())
            )
        )
