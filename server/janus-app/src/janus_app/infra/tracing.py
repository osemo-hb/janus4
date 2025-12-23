"""
OpenTelemetry Tracing Infrastructure - Janus App

This module configures the OpenTelemetry SDK for the Janus application.
It sets up exporters, processors, and auto-instrumentation.

The core library (janus-core) uses the OpenTelemetry API only.
This module configures the SDK so that all tracing in core becomes active.

Usage:
    from janus_app.infra.tracing import configure_tracing

    # In startup:
    configure_tracing(
        service_name="janus-app",
        otlp_endpoint="localhost:4317"
    )

    # Tracing utilities are re-exported from janus_core:
    from janus_core.tracing import tracer, trace_async, trace_sync
"""

import logging
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from janus_core.config import settings as core_settings
from janus_app.config import app_settings

# Re-export tracing utilities from core
from janus_core.tracing import (
    tracer,
    trace_async,
    trace_sync,
    get_current_trace_id,
    get_current_span_id,
    inject_trace_context,
    extract_trace_context,
    SpanContext,
)

logger = logging.getLogger(__name__)

__all__ = [
    # SDK configuration
    "configure_tracing",
    # Re-exports from janus_core.tracing
    "tracer",
    "trace_async",
    "trace_sync",
    "get_current_trace_id",
    "get_current_span_id",
    "inject_trace_context",
    "extract_trace_context",
    "SpanContext",
]


def configure_tracing(
    service_name: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
    console_export: Optional[bool] = None,
) -> Optional[TracerProvider]:
    """
    Configure OpenTelemetry SDK for the application.

    This sets up the global TracerProvider, which activates tracing
    for both janus-app and janus-core (which uses API only).

    Args:
        service_name: Name of the service for trace identification.
                     Defaults to app_settings.OTEL_SERVICE_NAME.
        otlp_endpoint: OTLP collector endpoint (e.g., "localhost:4317").
                      Defaults to app_settings.OTEL_EXPORTER_OTLP_ENDPOINT.
        console_export: If True, also export spans to console (for debugging).
                       Defaults to app_settings.OTEL_CONSOLE_EXPORT.

    Returns:
        Configured TracerProvider, or None if tracing is disabled.
    """
    if not app_settings.OTEL_ENABLED:
        logger.info("OpenTelemetry tracing is disabled")
        return None

    # Use defaults from settings if not provided
    service_name = service_name or app_settings.OTEL_SERVICE_NAME
    otlp_endpoint = otlp_endpoint or app_settings.OTEL_EXPORTER_OTLP_ENDPOINT
    console_export = console_export if console_export is not None else app_settings.OTEL_CONSOLE_EXPORT

    # Create resource with service information
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "4.0.0",
            "deployment.environment": core_settings.ENVIRONMENT,
        }
    )

    # Create and configure provider
    provider = TracerProvider(resource=resource)

    # Add OTLP exporter if endpoint provided
    if otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        logger.info(f"Tracing configured with OTLP endpoint: {otlp_endpoint}")

    # Add console exporter for debugging
    if console_export:
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(console_exporter))
        logger.info("Tracing configured with console export")

    # Set as global provider - this activates tracing for janus_core too!
    trace.set_tracer_provider(provider)

    # Auto-instrument common libraries
    _instrument_libraries()

    logger.info(f"OpenTelemetry tracing configured for '{service_name}'")

    return provider


def _instrument_libraries() -> None:
    """Auto-instrument common async libraries."""
    try:
        AsyncPGInstrumentor().instrument()
        logger.debug("AsyncPG instrumented for tracing")
    except Exception as e:
        logger.warning(f"Failed to instrument AsyncPG: {e}")

    try:
        RedisInstrumentor().instrument()
        logger.debug("Redis instrumented for tracing")
    except Exception as e:
        logger.warning(f"Failed to instrument Redis: {e}")
