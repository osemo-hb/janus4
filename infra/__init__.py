"""
Janus3 Infrastructure Module

Core infrastructure components for observability, messaging, and external services.
"""

from janus3.infra.tracing import (
    tracer,
    configure_tracing,
    trace_async,
    trace_sync,
    get_current_trace_id,
    inject_trace_context,
    extract_trace_context,
    SpanContext,
)
from janus3.infra.metrics import metrics
from janus3.infra.kafka_schemas import (
    MessageType,
    FactEventType,
    EntityEventType,
    MessageSource,
    TurnMessage,
    ConsolidateSignal,
    FactEvent,
    EntityEvent,
    DLQMessage,
    partition_key,
)
from janus3.infra.kafka_producer import (
    KafkaProducer,
    get_kafka_producer,
    close_kafka_producer,
)

__all__ = [
    # Tracing
    "tracer",
    "configure_tracing",
    "trace_async",
    "trace_sync",
    "get_current_trace_id",
    "inject_trace_context",
    "extract_trace_context",
    "SpanContext",
    # Metrics
    "metrics",
    # Kafka Schemas
    "MessageType",
    "FactEventType",
    "EntityEventType",
    "MessageSource",
    "TurnMessage",
    "ConsolidateSignal",
    "FactEvent",
    "EntityEvent",
    "DLQMessage",
    "partition_key",
    # Kafka Producer
    "KafkaProducer",
    "get_kafka_producer",
    "close_kafka_producer",
]
