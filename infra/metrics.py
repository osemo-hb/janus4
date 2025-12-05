"""
Prometheus Metrics Infrastructure - Janus 3.5

Provides metrics collection and exposure for the Janus3 system.
Covers consolidation, retrieval, LLM, embedding, and buffer metrics.

Usage:
    from janus3.infra.metrics import metrics

    # Increment counter
    metrics.consolidation_failures.labels(reason="timeout").inc()

    # Observe histogram
    metrics.consolidation_duration.observe(1.5)

The metrics are exposed via the /metrics endpoint in the API.
"""

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, Optional, TypeVar

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
    multiprocess,
)

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])

# Default histogram buckets for latency metrics
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)

# Larger buckets for consolidation (can take longer)
CONSOLIDATION_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)


class Janus3Metrics:
    """
    Centralized metrics registry for Janus3.

    All metrics are defined here to ensure consistent naming and labeling.
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()

        # ============================================================
        # Consolidation Metrics
        # ============================================================

        self.consolidation_duration = Histogram(
            "janus3_consolidation_duration_seconds",
            "Time to consolidate a session",
            ["trigger_reason"],
            buckets=CONSOLIDATION_BUCKETS,
            registry=self.registry,
        )

        self.consolidation_total = Counter(
            "janus3_consolidation_total",
            "Total consolidation attempts",
            ["status", "trigger_reason"],
            registry=self.registry,
        )

        self.consolidation_failures = Counter(
            "janus3_consolidation_failures_total",
            "Total failed consolidations",
            ["reason"],
            registry=self.registry,
        )

        self.consolidation_queue_size = Gauge(
            "janus3_consolidation_queue_size",
            "Number of pending consolidation jobs",
            registry=self.registry,
        )

        self.consolidation_turns_processed = Counter(
            "janus3_consolidation_turns_processed_total",
            "Total turns processed during consolidation",
            registry=self.registry,
        )

        self.consolidation_facts_extracted = Counter(
            "janus3_consolidation_facts_extracted_total",
            "Total facts extracted during consolidation",
            ["extractor"],
            registry=self.registry,
        )

        self.consolidation_entities_created = Counter(
            "janus3_consolidation_entities_created_total",
            "Total entities created during consolidation",
            registry=self.registry,
        )

        # ============================================================
        # Retrieval Metrics
        # ============================================================

        self.retrieval_duration = Histogram(
            "janus3_retrieval_duration_seconds",
            "Time for retrieval operations",
            ["retrieval_type"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )

        self.retrieval_results_count = Histogram(
            "janus3_retrieval_results_count",
            "Number of results returned from retrieval",
            ["retrieval_type"],
            buckets=(0, 1, 2, 5, 10, 20, 50, 100),
            registry=self.registry,
        )

        self.retrieval_cache_hits = Counter(
            "janus3_retrieval_cache_hits_total",
            "Total retrieval cache hits",
            ["cache_type"],
            registry=self.registry,
        )

        self.retrieval_cache_misses = Counter(
            "janus3_retrieval_cache_misses_total",
            "Total retrieval cache misses",
            ["cache_type"],
            registry=self.registry,
        )

        # ============================================================
        # Buffer Metrics
        # ============================================================

        self.session_buffer_size = Gauge(
            "janus3_session_buffer_size",
            "Number of turns buffered per session",
            ["session_id"],
            registry=self.registry,
        )

        self.session_buffer_total_size = Gauge(
            "janus3_session_buffer_total_size",
            "Total turns buffered across all sessions",
            registry=self.registry,
        )

        self.buffer_evictions = Counter(
            "janus3_buffer_evictions_total",
            "Total buffer evictions due to overflow",
            registry=self.registry,
        )

        # ============================================================
        # LLM Metrics
        # ============================================================

        self.llm_requests_total = Counter(
            "janus3_llm_requests_total",
            "Total LLM API requests",
            ["model", "operation", "status"],
            registry=self.registry,
        )

        self.llm_request_duration = Histogram(
            "janus3_llm_request_seconds",
            "Time for LLM API requests",
            ["model", "operation"],
            buckets=CONSOLIDATION_BUCKETS,  # LLM calls can be slow
            registry=self.registry,
        )

        self.llm_tokens_used = Counter(
            "janus3_llm_tokens_used_total",
            "Total tokens used in LLM calls",
            ["model", "token_type"],
            registry=self.registry,
        )

        # ============================================================
        # Embedding Metrics
        # ============================================================

        self.embedding_requests_total = Counter(
            "janus3_embedding_requests_total",
            "Total embedding requests",
            ["model", "status"],
            registry=self.registry,
        )

        self.embedding_duration = Histogram(
            "janus3_embedding_seconds",
            "Time for embedding generation",
            ["model", "batch_size"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )

        # ============================================================
        # STM Metrics (Redis Streams)
        # ============================================================

        self.stm_turns_added = Counter(
            "janus3_stm_turns_added_total",
            "Total turns added to STM",
            ["session_id"],
            registry=self.registry,
        )

        # ============================================================
        # Session Metrics
        # ============================================================

        self.sessions_created = Counter(
            "janus3_sessions_created_total",
            "Total sessions created",
            registry=self.registry,
        )

        self.sessions_deleted = Counter(
            "janus3_sessions_deleted_total",
            "Total sessions deleted",
            registry=self.registry,
        )

        # ============================================================
        # Threshold Metrics (Topic Boundary Detection)
        # ============================================================

        self.threshold_velocities = Histogram(
            "janus3_threshold_velocities",
            "Distribution of semantic velocities",
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=self.registry,
        )

        self.topic_boundaries_detected = Counter(
            "janus3_topic_boundaries_detected_total",
            "Total topic boundaries detected",
            registry=self.registry,
        )

        # ============================================================
        # Chat Metrics
        # ============================================================

        self.chat_requests_total = Counter(
            "janus3_chat_requests_total",
            "Total chat requests",
            ["status"],
            registry=self.registry,
        )

        self.chat_request_duration = Histogram(
            "janus3_chat_request_duration_seconds",
            "Time for chat requests",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )

        self.reconciliation_checks = Counter(
            "janus3_reconciliation_checks_total",
            "Total reconciliation checks performed",
            ["status"],  # "match" or "mismatch"
            registry=self.registry,
        )

        self.reconciliation_mismatches = Counter(
            "janus3_reconciliation_mismatches_total",
            "Total reconciliation mismatches found",
            ["mismatch_type"],
            registry=self.registry,
        )

        # ============================================================
        # API Metrics
        # ============================================================

        self.api_requests_total = Counter(
            "janus3_api_requests_total",
            "Total API requests",
            ["method", "endpoint", "status"],
            registry=self.registry,
        )

        self.api_request_duration = Histogram(
            "janus3_api_request_seconds",
            "Time for API requests",
            ["method", "endpoint"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )

        self.api_active_connections = Gauge(
            "janus3_api_active_connections",
            "Number of active API connections",
            registry=self.registry,
        )

        # ============================================================
        # System Info
        # ============================================================

        self.build_info = Info(
            "janus3_build",
            "Build information",
            registry=self.registry,
        )

    def generate_metrics(self) -> bytes:
        """Generate metrics output for Prometheus scraping."""
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """Get the content type for metrics response."""
        return CONTENT_TYPE_LATEST

    @contextmanager
    def timer(
        self,
        histogram: Histogram,
        labels: Optional[dict] = None,
    ) -> Generator[None, None, None]:
        """
        Context manager for timing operations.

        Usage:
            with metrics.timer(metrics.consolidation_duration, {"trigger_reason": "boundary"}):
                await consolidate(session_id)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            if labels:
                histogram.labels(**labels).observe(duration)
            else:
                histogram.observe(duration)

    def timed(
        self,
        histogram: Histogram,
        labels_func: Optional[Callable[..., dict]] = None,
    ) -> Callable[[F], F]:
        """
        Decorator for timing functions.

        Usage:
            @metrics.timed(metrics.retrieval_duration, lambda: {"retrieval_type": "entity"})
            async def search_entities():
                ...
        """

        def decorator(func: F) -> F:
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                labels = labels_func() if labels_func else {}
                with self.timer(histogram, labels):
                    return await func(*args, **kwargs)

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                labels = labels_func() if labels_func else {}
                with self.timer(histogram, labels):
                    return func(*args, **kwargs)

            import asyncio

            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            return sync_wrapper  # type: ignore

        return decorator


# Global metrics instance
metrics = Janus3Metrics()


def setup_multiprocess_mode(prometheus_multiproc_dir: str) -> None:
    """
    Configure Prometheus for multiprocess mode (e.g., with Gunicorn workers).

    Call this before creating the metrics instance if using multiprocess mode.

    Args:
        prometheus_multiproc_dir: Directory for multiprocess metrics files
    """
    import os

    os.environ["PROMETHEUS_MULTIPROC_DIR"] = prometheus_multiproc_dir

    # Re-create registry with multiprocess collector
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)

    global metrics
    metrics = Janus3Metrics(registry=registry)
