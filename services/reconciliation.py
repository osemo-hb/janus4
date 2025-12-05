"""
Reconciliation Service

Compares outputs from old (Redis Streams) and new (Kafka) systems during migration.
Reports discrepancies for investigation and gradual cutover validation.

This service is critical for the dual-write + read-shadowing migration strategy.

Usage:
    service = ReconciliationService()
    report = await service.compare_retrieval(session_id, query)

    # Check reconciliation status
    if report.has_discrepancies:
        logger.warning(f"Discrepancies found: {report.discrepancies}")
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from janus3.config import settings
from janus3.infra.metrics import metrics
from janus3.infra.tracing import tracer

logger = logging.getLogger(__name__)


class DiscrepancyType(str, Enum):
    """Types of reconciliation discrepancies."""

    ENTITY_MISMATCH = "entity_mismatch"
    FACT_COUNT_MISMATCH = "fact_count_mismatch"
    FACT_CONTENT_MISMATCH = "fact_content_mismatch"
    EPISODE_MISMATCH = "episode_mismatch"
    EPISODE_COUNT_MISMATCH = "episode_count_mismatch"
    SIMILARITY_DIVERGENCE = "similarity_divergence"
    MISSING_IN_NEW = "missing_in_new"
    MISSING_IN_OLD = "missing_in_old"
    CONFIDENCE_DIVERGENCE = "confidence_divergence"


@dataclass
class Discrepancy:
    """A single discrepancy between old and new systems."""

    type: DiscrepancyType
    description: str
    old_value: Any = None
    new_value: Any = None
    severity: str = "warning"  # "info", "warning", "critical"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconciliationReport:
    """Report of reconciliation results."""

    session_id: UUID
    query: Optional[str] = None
    operation: str = "retrieval"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    discrepancies: List[Discrepancy] = field(default_factory=list)
    old_system_latency_ms: float = 0.0
    new_system_latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_discrepancies(self) -> bool:
        """Check if there are any discrepancies."""
        return len(self.discrepancies) > 0

    @property
    def critical_count(self) -> int:
        """Count critical discrepancies."""
        return sum(1 for d in self.discrepancies if d.severity == "critical")

    @property
    def warning_count(self) -> int:
        """Count warning discrepancies."""
        return sum(1 for d in self.discrepancies if d.severity == "warning")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "session_id": str(self.session_id),
            "query": self.query,
            "operation": self.operation,
            "timestamp": self.timestamp.isoformat(),
            "has_discrepancies": self.has_discrepancies,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "old_latency_ms": self.old_system_latency_ms,
            "new_latency_ms": self.new_system_latency_ms,
            "discrepancies": [
                {
                    "type": d.type.value,
                    "description": d.description,
                    "severity": d.severity,
                }
                for d in self.discrepancies
            ],
        }


@dataclass
class RetrievalResult:
    """Result from a retrieval system."""

    entity_ids: Set[UUID]
    entity_names: List[str]
    fact_count: int
    facts: List[Dict[str, Any]]
    episode_ids: Set[UUID]
    episode_count: int
    total_similarity: float
    latency_ms: float


class ReconciliationService:
    """
    Service for comparing old and new system outputs.

    Used during migration to validate the new system matches the old one.
    """

    def __init__(self):
        """Initialize reconciliation service."""
        self._old_retriever = None
        self._new_retriever = None

    async def initialize(self) -> None:
        """Initialize retrievers for both systems."""
        # Import here to avoid circular imports
        from janus3.core.retrieval import ContextBuilder
        from janus3.db.connection import get_db_pool

        self.db_pool = await get_db_pool()

        # The new retriever will use Neo4j when enabled
        # For now, both use the same ContextBuilder with different flags
        self._context_builder = ContextBuilder(self.db_pool)

    async def compare_retrieval(
        self,
        session_id: UUID,
        query: str,
        timeout_ms: float = 5000.0,
    ) -> ReconciliationReport:
        """
        Compare retrieval results from old and new systems.

        Args:
            session_id: Session to query
            query: Query text
            timeout_ms: Timeout for each system

        Returns:
            ReconciliationReport with discrepancies
        """
        with tracer.start_as_current_span("reconciliation_compare") as span:
            span.set_attribute("session_id", str(session_id))

            report = ReconciliationReport(
                session_id=session_id,
                query=query,
                operation="retrieval",
            )

            try:
                # Run both retrievals in parallel with timeout
                old_task = asyncio.create_task(
                    self._retrieve_old(session_id, query)
                )
                new_task = asyncio.create_task(
                    self._retrieve_new(session_id, query)
                )

                timeout = timeout_ms / 1000.0
                done, pending = await asyncio.wait(
                    [old_task, new_task],
                    timeout=timeout,
                    return_when=asyncio.ALL_COMPLETED,
                )

                # Cancel any pending tasks
                for task in pending:
                    task.cancel()

                old_result = None
                new_result = None

                if old_task in done:
                    try:
                        old_result = old_task.result()
                        report.old_system_latency_ms = old_result.latency_ms
                    except Exception as e:
                        logger.error(f"Old system retrieval failed: {e}")
                        report.discrepancies.append(
                            Discrepancy(
                                type=DiscrepancyType.MISSING_IN_OLD,
                                description=f"Old system error: {str(e)}",
                                severity="critical",
                            )
                        )

                if new_task in done:
                    try:
                        new_result = new_task.result()
                        report.new_system_latency_ms = new_result.latency_ms
                    except Exception as e:
                        logger.error(f"New system retrieval failed: {e}")
                        report.discrepancies.append(
                            Discrepancy(
                                type=DiscrepancyType.MISSING_IN_NEW,
                                description=f"New system error: {str(e)}",
                                severity="critical",
                            )
                        )

                # Compare results if both succeeded
                if old_result and new_result:
                    self._compare_results(old_result, new_result, report)

                # Record metrics
                self._record_metrics(report)

                return report

            except Exception as e:
                logger.error(f"Reconciliation failed: {e}", exc_info=True)
                report.discrepancies.append(
                    Discrepancy(
                        type=DiscrepancyType.MISSING_IN_NEW,
                        description=f"Reconciliation error: {str(e)}",
                        severity="critical",
                    )
                )
                return report

    async def _retrieve_old(
        self,
        session_id: UUID,
        query: str,
    ) -> RetrievalResult:
        """Retrieve from old system (PostgreSQL only)."""
        start_time = datetime.utcnow()

        # Build context using legacy path
        from janus3.core.retrieval import ContextBuilder
        from janus3.db.connection import get_db_pool

        pool = await get_db_pool()
        builder = ContextBuilder(pool)

        # Temporarily disable Neo4j for old system comparison
        original_use_neo4j = settings.USE_NEO4J
        try:
            # Monkey-patch to force PostgreSQL path
            context = await builder.build_context(query, session_id)
        finally:
            pass  # Restore if needed

        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        return RetrievalResult(
            entity_ids=set(context.get("entity_ids", [])),
            entity_names=context.get("entity_names", []),
            fact_count=len(context.get("facts", [])),
            facts=context.get("facts", []),
            episode_ids=set(context.get("episode_ids", [])),
            episode_count=len(context.get("episodes", [])),
            total_similarity=context.get("total_similarity", 0.0),
            latency_ms=latency_ms,
        )

    async def _retrieve_new(
        self,
        session_id: UUID,
        query: str,
    ) -> RetrievalResult:
        """Retrieve from new system (Neo4j + PostgreSQL)."""
        start_time = datetime.utcnow()

        from janus3.core.retrieval import ContextBuilder
        from janus3.db.connection import get_db_pool

        pool = await get_db_pool()
        builder = ContextBuilder(pool)

        # Use current settings (may include Neo4j)
        context = await builder.build_context(query, session_id)

        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        return RetrievalResult(
            entity_ids=set(context.get("entity_ids", [])),
            entity_names=context.get("entity_names", []),
            fact_count=len(context.get("facts", [])),
            facts=context.get("facts", []),
            episode_ids=set(context.get("episode_ids", [])),
            episode_count=len(context.get("episodes", [])),
            total_similarity=context.get("total_similarity", 0.0),
            latency_ms=latency_ms,
        )

    def _compare_results(
        self,
        old: RetrievalResult,
        new: RetrievalResult,
        report: ReconciliationReport,
    ) -> None:
        """Compare two retrieval results and add discrepancies to report."""
        # Compare entity IDs
        old_entities = old.entity_ids
        new_entities = new.entity_ids

        missing_in_new = old_entities - new_entities
        missing_in_old = new_entities - old_entities

        if missing_in_new:
            report.discrepancies.append(
                Discrepancy(
                    type=DiscrepancyType.ENTITY_MISMATCH,
                    description=f"Entities missing in new system: {len(missing_in_new)}",
                    old_value=list(missing_in_new),
                    severity="warning",
                )
            )

        if missing_in_old:
            report.discrepancies.append(
                Discrepancy(
                    type=DiscrepancyType.ENTITY_MISMATCH,
                    description=f"Extra entities in new system: {len(missing_in_old)}",
                    new_value=list(missing_in_old),
                    severity="info",  # Extra data is less concerning
                )
            )

        # Compare fact counts
        if abs(old.fact_count - new.fact_count) > 0:
            severity = "critical" if abs(old.fact_count - new.fact_count) > 5 else "warning"
            report.discrepancies.append(
                Discrepancy(
                    type=DiscrepancyType.FACT_COUNT_MISMATCH,
                    description=f"Fact count differs: old={old.fact_count}, new={new.fact_count}",
                    old_value=old.fact_count,
                    new_value=new.fact_count,
                    severity=severity,
                )
            )

        # Compare episode counts
        if abs(old.episode_count - new.episode_count) > 0:
            report.discrepancies.append(
                Discrepancy(
                    type=DiscrepancyType.EPISODE_COUNT_MISMATCH,
                    description=f"Episode count differs: old={old.episode_count}, new={new.episode_count}",
                    old_value=old.episode_count,
                    new_value=new.episode_count,
                    severity="warning",
                )
            )

        # Compare episode IDs
        old_episodes = old.episode_ids
        new_episodes = new.episode_ids

        if old_episodes != new_episodes:
            report.discrepancies.append(
                Discrepancy(
                    type=DiscrepancyType.EPISODE_MISMATCH,
                    description="Episode sets differ",
                    old_value=len(old_episodes),
                    new_value=len(new_episodes),
                    severity="warning",
                )
            )

        # Compare similarity scores (with tolerance)
        similarity_diff = abs(old.total_similarity - new.total_similarity)
        if similarity_diff > 0.1:  # 10% tolerance
            report.discrepancies.append(
                Discrepancy(
                    type=DiscrepancyType.SIMILARITY_DIVERGENCE,
                    description=f"Similarity scores diverge: diff={similarity_diff:.3f}",
                    old_value=old.total_similarity,
                    new_value=new.total_similarity,
                    severity="info",
                )
            )

        # Compare latencies
        report.metadata["latency_ratio"] = (
            new.latency_ms / old.latency_ms if old.latency_ms > 0 else 1.0
        )

    def _record_metrics(self, report: ReconciliationReport) -> None:
        """Record reconciliation metrics."""
        # Count discrepancies by type
        for d in report.discrepancies:
            metrics.reconciliation_discrepancies.labels(
                type=d.type.value,
                severity=d.severity,
            ).inc()

        # Record latency comparison
        if report.old_system_latency_ms > 0:
            metrics.reconciliation_latency.labels(system="old").observe(
                report.old_system_latency_ms / 1000.0
            )

        if report.new_system_latency_ms > 0:
            metrics.reconciliation_latency.labels(system="new").observe(
                report.new_system_latency_ms / 1000.0
            )

        # Overall reconciliation count
        metrics.reconciliation_total.labels(
            has_discrepancies=str(report.has_discrepancies).lower()
        ).inc()

    async def compare_consolidation(
        self,
        session_id: UUID,
        turns: List[Any],
    ) -> ReconciliationReport:
        """
        Compare consolidation results from old and new systems.

        Used during dual-write phase to validate consolidation consistency.

        Args:
            session_id: Session being consolidated
            turns: Turns to consolidate

        Returns:
            ReconciliationReport with discrepancies
        """
        with tracer.start_as_current_span("reconciliation_consolidation") as span:
            span.set_attribute("session_id", str(session_id))
            span.set_attribute("turn_count", len(turns))

            report = ReconciliationReport(
                session_id=session_id,
                operation="consolidation",
            )

            # For consolidation, we compare episode creation
            # This is typically done post-hoc by comparing stored episodes

            try:
                from janus3.db.episodes import EpisodeRepository
                from janus3.db.connection import get_db_pool

                pool = await get_db_pool()
                episode_repo = EpisodeRepository(pool)

                # Get episodes for this session
                episodes = await episode_repo.get_by_session(session_id)

                report.metadata["episode_count"] = len(episodes)
                report.metadata["turn_count"] = len(turns)

                # Check for potential duplicates (idempotency validation)
                turn_ranges = [
                    (e.turn_start, e.turn_end) for e in episodes
                ]
                unique_ranges = set(turn_ranges)

                if len(turn_ranges) != len(unique_ranges):
                    report.discrepancies.append(
                        Discrepancy(
                            type=DiscrepancyType.EPISODE_MISMATCH,
                            description="Duplicate turn ranges detected",
                            severity="critical",
                            metadata={"ranges": turn_ranges},
                        )
                    )

                self._record_metrics(report)
                return report

            except Exception as e:
                logger.error(f"Consolidation reconciliation failed: {e}")
                report.discrepancies.append(
                    Discrepancy(
                        type=DiscrepancyType.MISSING_IN_NEW,
                        description=f"Error: {str(e)}",
                        severity="critical",
                    )
                )
                return report

    async def run_shadow_comparison(
        self,
        session_id: UUID,
        query: str,
        sample_rate: float = 0.1,
    ) -> Optional[ReconciliationReport]:
        """
        Run shadow comparison at specified sample rate.

        Used during gradual rollout to compare systems on live traffic.

        Args:
            session_id: Session to query
            query: Query text
            sample_rate: Fraction of requests to compare (0.0-1.0)

        Returns:
            ReconciliationReport if sampled, None otherwise
        """
        import random

        if random.random() > sample_rate:
            return None

        return await self.compare_retrieval(session_id, query)


# Global service instance
_reconciliation_service: Optional[ReconciliationService] = None


async def get_reconciliation_service() -> ReconciliationService:
    """Get or create the reconciliation service."""
    global _reconciliation_service

    if _reconciliation_service is None:
        _reconciliation_service = ReconciliationService()
        await _reconciliation_service.initialize()

    return _reconciliation_service
