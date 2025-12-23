"""
OpenTelemetry Tracing - API Only

This module provides tracing utilities using only the OpenTelemetry API.
If no SDK is configured (by the application layer), all tracing operations
become no-ops, making the library usable without any tracing infrastructure.

The application (e.g., janus-app) is responsible for:
- Configuring the TracerProvider
- Setting up exporters (OTLP, Jaeger, etc.)
- Auto-instrumenting libraries (asyncpg, redis, etc.)

Usage:
    from janus_core.tracing import tracer, trace_async

    @trace_async("my_operation")
    async def my_function():
        ...

    # Or manually:
    with tracer.start_as_current_span("operation_name") as span:
        span.set_attribute("key", "value")
        ...
"""

import functools
import logging
from typing import Any, Callable, Dict, Optional, TypeVar
from uuid import UUID

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Span
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

logger = logging.getLogger(__name__)

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])

# Global tracer instance - returns no-op tracer if no SDK configured
tracer = trace.get_tracer("janus_core", "4.0.0")

# Propagator for distributed tracing across services
propagator = TraceContextTextMapPropagator()


def trace_async(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """
    Decorator for tracing async functions.

    Args:
        name: Span name (defaults to function name)
        attributes: Static attributes to add to the span

    Usage:
        @trace_async("consolidate_session")
        async def consolidate(session_id: UUID):
            ...

        @trace_async(attributes={"component": "retrieval"})
        async def search_entities():
            ...
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(span_name) as span:
                # Add static attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                # Try to extract common attributes from kwargs
                _add_common_attributes(span, kwargs)

                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper  # type: ignore

    return decorator


def trace_sync(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """
    Decorator for tracing sync functions.

    Args:
        name: Span name (defaults to function name)
        attributes: Static attributes to add to the span
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                _add_common_attributes(span, kwargs)

                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper  # type: ignore

    return decorator


def _add_common_attributes(span: Span, kwargs: Dict[str, Any]) -> None:
    """Extract and add common attributes from function kwargs."""
    # Session ID
    if "session_id" in kwargs:
        session_id = kwargs["session_id"]
        if isinstance(session_id, UUID):
            span.set_attribute("session_id", str(session_id))
        else:
            span.set_attribute("session_id", session_id)

    # Query text (truncated)
    if "query" in kwargs:
        query = kwargs["query"]
        if isinstance(query, str):
            span.set_attribute("query", query[:100])

    # Turn count
    if "turns" in kwargs:
        turns = kwargs["turns"]
        if isinstance(turns, (list, tuple)):
            span.set_attribute("turn_count", len(turns))


def get_current_trace_id() -> Optional[str]:
    """Get the current trace ID as a hex string, or None if no active trace."""
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().trace_id, "032x")
    return None


def get_current_span_id() -> Optional[str]:
    """Get the current span ID as a hex string, or None if no active span."""
    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().span_id, "016x")
    return None


def inject_trace_context(carrier: Dict[str, str]) -> None:
    """
    Inject trace context into a carrier dict for propagation.

    Use this when sending messages or making HTTP calls.

    Args:
        carrier: Dictionary to inject trace headers into
    """
    propagator.inject(carrier)


def extract_trace_context(carrier: Dict[str, str]) -> trace.Context:
    """
    Extract trace context from a carrier dict.

    Use this when receiving messages or HTTP requests.

    Args:
        carrier: Dictionary containing trace headers

    Returns:
        Extracted trace context
    """
    return propagator.extract(carrier)


class SpanContext:
    """
    Context manager for creating child spans with automatic attribute handling.

    Usage:
        async with SpanContext("db_query", session_id=session_id) as span:
            span.set_attribute("query_type", "entity_search")
            result = await db.fetch(...)
    """

    def __init__(
        self,
        name: str,
        session_id: Optional[UUID] = None,
        **attributes: Any,
    ):
        self.name = name
        self.session_id = session_id
        self.attributes = attributes
        self._span: Optional[Span] = None

    def __enter__(self) -> Span:
        self._span = tracer.start_span(self.name)
        self._span.__enter__()

        if self.session_id:
            self._span.set_attribute("session_id", str(self.session_id))

        for key, value in self.attributes.items():
            if value is not None:
                self._span.set_attribute(key, value)

        return self._span

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._span:
            if exc_val:
                self._span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self._span.record_exception(exc_val)
            else:
                self._span.set_status(Status(StatusCode.OK))
            self._span.__exit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self) -> Span:
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)
