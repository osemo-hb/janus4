# Janus 3.5 Architecture Documentation

## Executive Summary

**Janus 3.5** is a streamlined, dual-process conversational memory system that combines fast chat responses with asynchronous memory consolidation. It implements a focused memory model with:

- **Nervous System (FastAPI)**: Handles real-time chat with sub-100ms TTFT
- **Cortex (Async Consumer)**: Consolidates memories asynchronously
- **Unified Extraction**: Single LLM call for entities, facts, and summaries
- **Simplified Storage**: PostgreSQL + pgvector + Redis Streams (no Neo4j/Kafka)

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Core Components](#2-core-components)
3. [Data Flow](#3-data-flow)
4. [State Management](#4-state-management)
5. [External Integrations](#5-external-integrations)
6. [Configuration](#6-configuration)
7. [Entry Points](#7-entry-points)
8. [Design Patterns](#8-design-patterns)
9. [Database Schema](#9-database-schema)
10. [Production Features](#10-production-features)
11. [Testing Strategy](#11-testing-strategy)
12. [Quick Start Reference](#12-quick-start-reference)

---

## 1. Project Structure

```
janus3/
├── api/                      # FastAPI Nervous System (chat + retrieval)
│   ├── main.py              # FastAPI app setup with lifespan management
│   ├── routes.py            # API endpoints (sessions, chat, debug/admin)
│   └── dependencies.py      # Dependency injection and singleton management
│
├── consumer/                 # Native asyncio Cortex (consolidation)
│   ├── consolidation.py     # ConsolidationService - main consolidation logic
│   └── run_consumer.py      # Entry point with graceful shutdown
│
├── core/                     # Business logic and ML services
│   ├── embedder.py          # Thread-safe async embedding service
│   ├── unified_extractor.py # Single LLM extraction (entities + facts + summary)
│   ├── model_registry.py    # Embedding model versioning and compatibility
│   ├── retrieval.py         # Hybrid retrieval (vector search)
│   ├── threshold.py         # Adaptive semantic boundary detection
│   ├── threshold_store.py   # Redis-backed threshold persistence
│   └── models.py            # Pydantic data structures
│
├── db/                       # Database layer (repositories + connection)
│   ├── connection.py        # asyncpg pool management with pgvector
│   ├── stm.py               # Redis Streams-based Short-Term Memory
│   ├── episodes.py          # Episode (LTM) CRUD with multi-vector storage
│   ├── facts.py             # Simple facts and entities (no versioning)
│   └── migrations/          # PostgreSQL schema
│       ├── 001_initial.sql
│       ├── 002_architecture_refactor.sql
│       └── 003_simplify_facts.sql  # Janus 3.5 simplification
│
├── infra/                    # Infrastructure and observability
│   ├── tracing.py           # OpenTelemetry distributed tracing
│   └── metrics.py           # Prometheus metrics collection
│
├── services/                 # External service integrations
│   ├── llm_service.py       # Async OpenAI wrapper
│   └── reconciliation.py    # Data reconciliation utilities
│
├── scripts/                  # Utility scripts
│   └── threshold_update.lua # Lua script for atomic Redis operations
│
├── tests/                    # Test suite
│   ├── test_threshold.py
│   └── test_parallel_retrieval.py
│
├── config.py                 # Centralized configuration
├── janus3.py                 # Legacy demo (V2 architecture reference)
├── docker-compose.yml        # Infrastructure definition
└── requirements.txt          # Python dependencies
```

### Directory Responsibilities

| Directory | Purpose | Key Responsibilities |
|-----------|---------|---------------------|
| `api/` | FastAPI Layer | HTTP requests, <100ms TTFT, parallel retrieval |
| `consumer/` | Async Consumer | Background STM→LTM consolidation |
| `core/` | Business Logic | Embeddings, unified extraction, retrieval, thresholds |
| `db/` | Database Layer | Repository pattern, asyncpg, Redis Streams |
| `infra/` | Observability | Tracing (OpenTelemetry), Metrics (Prometheus) |
| `services/` | External APIs | OpenAI integration |
| `scripts/` | Utilities | Lua scripts for Redis atomic operations |

---

## 2. Core Components

### Component Interaction Diagram

```
User Message
    │
    ▼
┌─────────────────────────────────────────┐
│  API: chat() endpoint                   │
│  1. Embed query → QueryEmbedder         │
│  2. Store in STM → Redis Streams (XADD) │
│  3. Check boundary → AdaptiveThreshold  │
│  4. Retrieve context → HybridRetriever  │
│  5. Generate response → LLMService      │
│  6. Store response in STM               │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Redis Streams (STM)                    │
│  Queues consolidation work              │
│  (handles crash recovery via XAUTOCLAIM)│
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  Consumer: Cortex                       │
│  1. Read STM entries → XREADGROUP       │
│  2. Extract facts → LLMService          │
│  3. Create entities → EntityLinker      │
│  4. Store episode → PostgreSQL          │
│  5. ACK completion → XACK               │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  PostgreSQL (LTM)                       │
│  Episodes + Vectors + Facts + Entities  │
│  (indexed with pgvector HNSW)           │
└─────────────────────────────────────────┘
```

### 2.1 Embedder (`core/embedder.py`)

**Responsibility**: Generate vector embeddings for text

| Property | Value |
|----------|-------|
| Model | SentenceTransformer ("all-MiniLM-L6-v2") |
| Dimensions | 384 |
| Pattern | Singleton with ThreadPoolExecutor |

**API**:
```python
encode(text: str) → List[float]
encode_batch(texts: List[str]) → List[List[float]]
cosine_distance(vec_a: List[float], vec_b: List[float]) → float
```

**Key Implementation**:
- Uses ThreadPoolExecutor to avoid blocking the async event loop
- Singleton pattern ensures single model load per process
- Thread-safe for concurrent requests

### 2.2 Unified Extractor (`core/unified_extractor.py`)

**Responsibility**: Extract entities, facts, and summary in a single LLM call

| Property | Value |
|----------|-------|
| Model | gpt-4o-mini (configurable) |
| Entity Types | person, organization, location, concept, product, event |
| Pattern | Singleton with LLMService |

**Replaces** (Janus 3.5 simplification):
- ~~GLiNER EntityLinker~~
- ~~spaCy SpaCyExtractor~~
- ~~LLMFactEnricher~~

**API**:
```python
result = await extractor.extract(text)
# Returns: ExtractionResult(entities, facts, summary)
```

**Behavior**:
- Single LLM call extracts all information
- Truncates text to ~4000 chars to avoid token limits
- Returns empty result on failure (graceful fallback)
- Low temperature (0.1) for consistent extraction

### 2.3 Adaptive Threshold (`core/threshold.py`)

**Responsibility**: Detect topic boundaries through semantic velocity

**Algorithm**:
1. Track cosine distance between consecutive user inputs
2. Maintain rolling statistics (mean + standard deviation)
3. Detect boundary when: `velocity > mean + (std_multiplier × std)`
4. Threshold clamped to [0.25, 0.65] to prevent extremes

**Cold Start Solution**:
```python
# Pre-initialized with calculated velocities to avoid cold start problem
COLD_START_VELOCITIES = [0.32, 0.28, 0.35, ...]  # 20 values
```

**Output**: `(velocity: float, is_boundary: bool)`

### 2.4 Hybrid Retriever (`core/retrieval.py`)

**Responsibility**: Build comprehensive context for LLM responses

**Parallelization Strategy**:
```python
# PARALLEL execution (no dependencies)
entities, episodes = await asyncio.gather(
    self._search_entities(query_vec, session_id),
    self._search_episodes(query_vec, session_id)
)

# SEQUENTIAL execution (depends on entities)
facts = await self._search_graph(entity_ids, session_id)
```

**Retrieval Sources**:
1. **Vector search on entities** - HNSW index on entities table
2. **Vector search on episodes** - HNSW index on episode_vectors (summary type)
3. **Graph traversal** - Facts related to discovered entities
4. **STM context** - Recent turns for immediate context

### 2.5 Model Registry (`core/model_registry.py`)

**Responsibility**: Embedding model versioning and compatibility checking

**Purpose**: Prevents "model soup" issues where vectors from different embedding models are incorrectly compared, leading to poor retrieval quality.

**API**:
```python
# Check compatibility before comparison
if ModelRegistry.can_compare(model_a_id, model_b_id):
    similarity = cosine_similarity(vec_a, vec_b)
else:
    raise IncompatibleModelsError(model_a, model_b)
```

**Registered Models**:
| Model ID | Dimensions | Description |
|----------|------------|-------------|
| `all-MiniLM-L6-v2` | 384 | Default (fast, good quality) |
| `all-mpnet-base-v2` | 768 | Higher quality, slower |
| `text-embedding-ada-002` | 1536 | OpenAI Ada |
| `text-embedding-3-small` | 1536 | OpenAI v3 small |
| `text-embedding-3-large` | 3072 | OpenAI v3 large |

**VectorMetadata**: Stored alongside vectors for tracking:
- `model_id`: Embedding model identifier
- `model_version`: Model version string
- `dimension`: Vector dimension

### 2.7 STM Manager (`db/stm.py`)

**Responsibility**: Manage Short-Term Memory in Redis Streams

**Redis Streams Pattern**:
| Operation | Redis Command | Purpose |
|-----------|---------------|---------|
| Add turn | `XADD` | Append to stream |
| Read new | `XREADGROUP` | Consumer group read |
| Acknowledge | `XACK` | Mark as processed |
| Recover | `XAUTOCLAIM` | Claim abandoned messages |
| Trim | `XTRIM MAXLEN ~1000` | Limit stream size |

**Stream Key Format**: `stm:{session_id}`

**Entry Fields**:
- `role`: "user" or "assistant"
- `content`: Message text
- `vector`: Serialized embedding (384-dim)
- `timestamp`: ISO timestamp
- `turn_index`: Sequential turn number

### 2.8 Episode Repository (`db/episodes.py`)

**Responsibility**: CRUD operations for Long-Term Memory episodes

**Multi-Vector Storage Strategy**:
| Vector Type | Purpose | Index |
|-------------|---------|-------|
| `summary` | Primary episode embedding | Partial HNSW |
| `centroid` | Mean of user vectors | None |
| `user_turn` | First 3 user inputs | Partial HNSW |

**Partial Index Design**:
```sql
-- Separate HNSW indexes for different vector types
CREATE INDEX idx_vec_summary ON episode_vectors
USING hnsw (embedding vector_cosine_ops)
WHERE vector_type = 'summary';

CREATE INDEX idx_vec_turns ON episode_vectors
USING hnsw (embedding vector_cosine_ops)
WHERE vector_type = 'user_turn';
```

### 2.8 Fact Repository (`db/facts.py`)

**Responsibility**: Simple fact storage with upsert (Janus 3.5)

**Upsert Pattern** (replaces versioning):
```sql
-- Simple upsert using unique constraint
INSERT INTO facts (session_id, subject_entity_id, predicate, object, ...)
VALUES ($1, $2, $3, $4, ...)
ON CONFLICT (session_id, subject_entity_id, predicate)
DO UPDATE SET object = $4, updated_at = NOW();
```

**Entity Management** (nested EntityRepository):
- Get-or-create with normalization (lowercase, trimmed)
- Vector similarity search for entity linking
- UNIQUE constraint on (session_id, name)

### 2.9 LLM Service (`services/llm_service.py`)

**Responsibility**: OpenAI integration for text generation

| Method | Purpose | Response Format |
|--------|---------|-----------------|
| `chat()` | Single-turn completion | String |
| `chat_with_history()` | Multi-turn conversation | String |
| `generate_json()` | Structured extraction | JSON string |
| `extract_facts()` | Conversation → summary + facts | Dict |

**Configuration**:
- Model: `gpt-4o-mini` (configurable)
- Client: `AsyncOpenAI`
- Fallback: Mock mode when API key unavailable

### 2.10 Consolidation Service (`consumer/consolidation.py`)

**Responsibility**: Memory consolidation pipeline (Janus 3.5)

**Singleton Pattern**: `await ConsolidationService.get_instance()`

**Simplified Consolidation Flow**:
```
Turns Input
    ↓
1. Format turns → text block
2. Unified extraction → UnifiedExtractor (single LLM call)
   - Extracts entities, facts, and summary together
3. Generate embeddings:
   - Summary vector
   - Centroid vector (mean of user turn vectors)
   - User turn vectors (first 3 for anchoring)
4. Create episode → EpisodeRepository (PostgreSQL)
5. Persist entities/facts → PostgreSQL (simple upsert)
```

**Key Features**:
- Single LLM call for extraction (replaces GLiNER + spaCy + LLM enrichment)
- LLM calls rate-limited via `asyncio.Semaphore(MAX_CONCURRENT_LLM_CALLS)`
- OpenTelemetry tracing for each phase
- Prometheus metrics for consolidation duration and counts

**Configuration**:
```python
MIN_TURNS_FOR_CONSOLIDATION = 5
MAX_CONCURRENT_LLM_CALLS = 5
MAX_CONCURRENT_CONSOLIDATIONS = 10
CONSOLIDATION_WORKERS = 4
```

---

## 3. Data Flow

### 3.1 Chat Request Flow

```
POST /api/v1/sessions/{id}/chat
│
├─ 1. EMBEDDING PHASE
│   └─ Embedder.encode(user_message) → query_vector (384-dim)
│       └─ ThreadPoolExecutor (non-blocking)
│
├─ 2. STM STORAGE
│   └─ STMManager.add_turn(session_id, "user", message, query_vector)
│       └─ Redis XADD: stm:{session_id}
│
├─ 3. BOUNDARY DETECTION
│   └─ AdaptiveThreshold.update(query_vector)
│       ├─ Calculate semantic velocity
│       ├─ Compare against adaptive threshold
│       └─ If boundary: signal consolidation
│
├─ 4. PARALLEL RETRIEVAL
│   ├─ Task 1 (parallel): _search_entities()
│   │   └─ PostgreSQL HNSW: entities table
│   │
│   ├─ Task 2 (parallel): _search_episodes()
│   │   └─ PostgreSQL HNSW: episode_vectors (summary)
│   │
│   ├─ Task 3 (sequential): _search_graph()
│   │   └─ Requires entity_ids from Task 1
│   │
│   └─ Task 4 (parallel): _get_stm_turns()
│       └─ Redis XREVRANGE: recent 10 turns
│
├─ 5. CONTEXT BUILDING
│   ├─ Format knowledge graph facts
│   ├─ Include episode summaries
│   └─ Add STM turns
│
├─ 6. LLM GENERATION
│   └─ LLMService.chat(system_prompt, user_message)
│
└─ 7. RESPONSE STORAGE
    └─ STMManager.add_turn(session_id, "assistant", response, vector)
```

### 3.2 Memory Consolidation Flow

```
Consumer Loop (every 1 second)
│
├─ 1. SESSION DISCOVERY
│   └─ SELECT sessions WHERE last_activity > NOW() - 24h
│
├─ 2. STM CHECK
│   └─ For each session: check stream length
│
├─ 3. CONSOLIDATION DECISION
│   ├─ Manual trigger (boundary signal)
│   ├─ Time-based (periodic)
│   └─ Volume-based (STM > threshold)
│
├─ 4. FACT EXTRACTION (LLM)
│   └─ Returns: {summary: str, facts: [{subject, predicate, object, confidence}]}
│
├─ 5. VECTOR GENERATION
│   ├─ Summary vector: encode(summary)
│   ├─ Centroid vector: mean(user_vectors)
│   └─ User turn vectors: first 3 inputs
│
├─ 6. EPISODE CREATION
│   └─ PostgreSQL transaction: episode + episode_vectors
│
├─ 7. ENTITY & FACT CREATION
│   ├─ Get-or-create entities
│   └─ Upsert facts with versioning
│
├─ 8. STREAM ACKNOWLEDGMENT
│   └─ Redis XACK: mark entries processed
│
└─ 9. CRASH RECOVERY
    └─ Redis XAUTOCLAIM: recover abandoned messages
```

### 3.3 Topic Boundary Example

```
Turn 1: "I'm planning a dinner party. My wife is vegetarian."
Turn 2: "What are some high-protein vegetarian dishes?"
Turn 3: "Okay enough food. I have a Python bug."  ← BOUNDARY DETECTED
```

**What Happens**:
1. Turn 1-2: Semantic distance < threshold → no action
2. Turn 3: Semantic distance > threshold → boundary detected
3. Consumer processes STM entries
4. LLM extracts: `{summary: "User discussed vegetarian dinner", facts: [{subject: "User's wife", predicate: "dietary_preference", object: "vegetarian"}]}`
5. Episode created, facts stored
6. Turn 3 starts fresh STM context

---

## 4. State Management

### 4.1 State Storage Layers

| Layer | Storage | Persistence | Consistency | Purpose |
|-------|---------|-------------|-------------|---------|
| **STM** | Redis Streams | Append-only | Ordered, ACK-based | Raw turns, immediate context |
| **LTM** | PostgreSQL | ACID transactions | Strong | Consolidated episodes, facts |
| **Session** | PostgreSQL | On delete cascade | FK constraints | Session metadata |
| **Entity** | PostgreSQL + pgvector | ACID + indexed | UNIQUE constraints | Named entities |
| **Threshold** | In-memory | Lost on restart | Session-scoped | Velocity history |

### 4.2 Session Lifecycle

```
CREATE_SESSION (POST /sessions)
├─ Create sessions record (PostgreSQL)
├─ Create STM consumer group (Redis XGROUP CREATE)
├─ Initialize AdaptiveThreshold in-memory
└─ Return session_id

CHAT (POST /sessions/{id}/chat)
├─ Update last_activity timestamp
├─ Add turn to STM stream
├─ Update threshold with new vector
└─ Return response

CONSOLIDATION (Consumer loop)
├─ Read STM entries (XREADGROUP)
├─ Process and ACK
├─ Create episode in PostgreSQL
└─ Update last_consolidation timestamp

DELETE_SESSION (DELETE /sessions/{id})
├─ Delete from sessions (cascades to episodes, facts, entities)
├─ Delete Redis STM stream
└─ Clear threshold cache
```

### 4.3 Threshold State (Per-Session)

```python
AdaptiveThreshold:
├─ velocities: List[float]    # 20 cold-start + accumulated
├─ last_vector: List[float]   # 384-dim, updated each turn
├─ window_size: int           # 50 (rolling window)
└─ std_multiplier: float      # 1.5 (sensitivity)

# Calculation:
threshold = mean(velocities) + std_multiplier * std(velocities)
threshold = clamp(threshold, 0.25, 0.65)
is_boundary = velocity > threshold
```

---

## 5. External Integrations

### 5.1 OpenAI Integration

**Service**: `LLMService` (Singleton)

| Property | Value |
|----------|-------|
| Model | gpt-4o-mini |
| Client | AsyncOpenAI |
| Fallback | Mock mode (testing) |

**Error Handling**:
```python
if not self.has_api:
    return mock_response()  # Graceful degradation
```

### 5.2 PostgreSQL with pgvector

| Property | Value |
|----------|-------|
| Driver | asyncpg (async, no ORM) |
| Extensions | vector, uuid-ossp |
| Pool Size | 5-20 connections |
| Timeout | 60 seconds |

**Key Vector Operations**:
```sql
-- Cosine similarity search
SELECT *, 1 - (embedding <=> $1::vector) as similarity
FROM entities
WHERE session_id = $2
ORDER BY embedding <=> $1::vector
LIMIT $3;
```

### 5.3 Redis Integration

| Property | Value |
|----------|-------|
| Purpose | STM storage, crash recovery, threshold persistence |
| Pattern | Redis Streams |
| Consumer Group | cortex_workers |
| Max Stream Length | 1000 entries |

**Stream Key**: `stm:{session_id}`

### 5.4 OpenTelemetry Tracing (`infra/tracing.py`)

| Property | Value |
|----------|-------|
| Provider | OpenTelemetry SDK |
| Exporter | OTLP (gRPC) |
| Feature Flag | `OTEL_ENABLED=true` |

**Auto-instrumented Libraries**:
- asyncpg (PostgreSQL)
- redis.asyncio

**Decorators**:
```python
@trace_async("operation_name")
async def my_function():
    ...

# Or manual spans:
with tracer.start_as_current_span("operation_name") as span:
    span.set_attribute("key", "value")
```

### 5.5 Prometheus Metrics (`infra/metrics.py`)

| Property | Value |
|----------|-------|
| Library | prometheus_client |
| Endpoint | `/metrics` |
| Feature Flag | `PROMETHEUS_ENABLED=true` |

**Metric Categories**:

| Category | Metrics |
|----------|---------|
| **Consolidation** | duration, total, failures, turns_processed, facts_extracted |
| **Retrieval** | duration, results_count |
| **LLM** | requests_total, request_duration |
| **Embedding** | requests_total, duration |
| **Threshold** | updates, velocity |
| **API** | requests_total, request_duration |

**Usage**:
```python
from janus3.infra.metrics import metrics

# Counter
metrics.consolidation_total.labels(status="success", trigger_reason="boundary").inc()

# Histogram
metrics.consolidation_duration.labels(trigger_reason="boundary").observe(1.5)

# Timer context manager
with metrics.timer(metrics.retrieval_duration, {"retrieval_type": "entity"}):
    results = await search_entities(...)
```

---

## 6. Configuration

**File**: `config.py`
**Pattern**: Pydantic BaseSettings (environment variable support)

### 6.1 Database Configuration

```python
DATABASE_URL: str = "postgresql://janus:janus_dev@localhost:5432/janus3_db"
```

### 6.2 Redis Configuration

```python
REDIS_URL: str = "redis://localhost:6379/0"
STM_STREAM_PREFIX: str = "stm"
CORTEX_CONSUMER_GROUP: str = "cortex_workers"
STM_MAX_LEN: int = 1000
```

### 6.3 Embedding Configuration

```python
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384
```

### 6.4 Threshold Configuration

```python
COLD_START_VELOCITIES: List[float] = [0.32, 0.28, ...]  # 20 values
ADAPTIVE_THRESHOLD_WINDOW: int = 50
ADAPTIVE_THRESHOLD_MULTIPLIER: float = 1.5
```

### 6.5 Retrieval Configuration

```python
RETRIEVAL_TOP_K_ENTITIES: int = 10
RETRIEVAL_TOP_K_EPISODES: int = 5
STM_CONTEXT_TURNS: int = 10
```

### 6.6 Consolidation Configuration

```python
MIN_TURNS_FOR_CONSOLIDATION: int = 5
CONSOLIDATION_IDLE_TIMEOUT_MS: int = 60000
MAX_BUFFER_SIZE: int = 100
MAX_CONCURRENT_CONSOLIDATIONS: int = 10
MAX_CONCURRENT_LLM_CALLS: int = 5
CONSOLIDATION_WORKERS: int = 4
```

### 6.7 LLM Configuration

```python
OPENAI_API_KEY: str = ""
GENERATION_MODEL: str = "gpt-4o-mini"
```

### 6.8 Extraction Configuration (Janus 3.5)

```python
EXTRACTION_MODEL: str = "gpt-4o-mini"  # For unified extractor
```

*Note: Janus 3.5 uses a unified LLM extractor. GLiNER and spaCy configurations have been removed.*

### 6.9 Observability Configuration

```python
# OpenTelemetry
OTEL_ENABLED: bool = True
OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = None  # e.g., "localhost:4317"
OTEL_SERVICE_NAME: str = "janus3"
OTEL_CONSOLE_EXPORT: bool = False

# Prometheus
PROMETHEUS_ENABLED: bool = True
PROMETHEUS_MULTIPROC_DIR: Optional[str] = None
```

### 6.10 Feature Flags

*Note: Janus 3.5 has removed migration-related feature flags (`USE_KAFKA`, `USE_NEO4J`, `USE_LEGACY_STM`). The architecture now uses PostgreSQL + Redis Streams exclusively.*

---

## 7. Entry Points

### 7.1 API Entry Point

**File**: `api/main.py`
**Command**: `uvicorn api.main:app --reload`

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db_pool()
    await get_redis()
    get_embedder()
    yield
    # Shutdown
    await close_db_pool()
    await close_redis()
```

### 7.2 API Routes

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v1/sessions` | Create session |
| GET | `/api/v1/sessions/{id}` | Session info |
| DELETE | `/api/v1/sessions/{id}` | Delete session |
| POST | `/api/v1/sessions/{id}/chat` | Chat (main endpoint) |
| GET | `/api/v1/sessions/{id}/stm` | Debug STM |
| GET | `/api/v1/sessions/{id}/episodes` | Debug episodes |
| GET | `/api/v1/sessions/{id}/facts` | Debug facts |
| GET | `/api/v1/sessions/{id}/threshold` | Debug threshold |

### 7.3 Consumer Entry Point

**File**: `consumer/run_consumer.py`
**Command**: `python -m consumer.run_consumer`

```python
def main():
    consumer = create_consumer()
    loop = asyncio.new_event_loop()

    # Graceful shutdown handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    loop.run_until_complete(consumer.run())
```

---

## 8. Design Patterns

### 8.1 Singleton with Lazy Loading

**Applied To**: Embedder, EntityLinker, LLMService

```python
class Embedder:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if Embedder._model is None:
            Embedder._model = SentenceTransformer(...)
```

**Benefit**: Single model load per process, shared across all requests

### 8.2 Dependency Injection with lru_cache

**Applied To**: API dependencies

```python
@lru_cache()
def get_embedder() -> Embedder:
    return Embedder()
```

**Benefit**: Singletons with FastAPI integration

### 8.3 Repository Pattern

**Applied To**: Database access

```python
class EpisodeRepository:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def create_episode(...) → Episode
    async def search_by_summary(...) → List[Tuple[Episode, float]]
```

**Benefit**: Abstraction layer, testability, consistent CRUD interface

### 8.4 Redis Streams for Crash Recovery

**Applied To**: STM management

```python
# Add work
await redis.xadd(stream_key, {...})

# Read and process
entries = await redis.xreadgroup(group, consumer, {stream_key: ">"})
await redis.xack(stream_key, group, *entry_ids)

# Crash recovery
await redis.xautoclaim(stream_key, group, consumer, min_idle_ms=60000)
```

**Benefit**: Built-in crash recovery without separate job queue

### 8.5 Parallel Retrieval with asyncio.gather

**Applied To**: HybridRetriever

```python
# Independent tasks run in parallel
entities, episodes = await asyncio.gather(
    self._search_entities(query_vec, session_id),
    self._search_episodes(query_vec, session_id)
)

# Dependent task runs sequentially
facts = await self._search_graph(entity_ids, session_id)
```

**Benefit**: Reduced retrieval latency

### 8.6 Thread Pool Executor for Blocking Operations

**Applied To**: Embedder, EntityLinker

```python
class Embedder:
    _executor = ThreadPoolExecutor(max_workers=2)

    async def encode(self, text: str) → List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._encode_sync,
            text
        )
```

**Benefit**: Non-blocking async loop, CPU-bound work offloaded

### 8.7 Simple Upsert Facts (Janus 3.5)

**Applied To**: Facts table

```sql
-- Simple upsert using unique constraint
INSERT INTO facts (session_id, subject_entity_id, predicate, object, ...)
ON CONFLICT (session_id, subject_entity_id, predicate)
DO UPDATE SET object = EXCLUDED.object, updated_at = NOW();
```

**Benefit**: Simpler logic, reduced storage, faster queries

### 8.8 Context Manager Lifespan

**Applied To**: FastAPI app

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()
    yield
    await shutdown()
```

**Benefit**: Guaranteed cleanup, no orphaned connections

### 8.9 OpenTelemetry Trace Decorator

**Applied To**: All async operations

```python
@trace_async("operation_name")
async def my_function(session_id: UUID):
    # Automatically creates span with session_id attribute
    pass
```

**Benefit**: Distributed tracing across services, automatic error recording

### 8.10 Prometheus Metrics Timer

**Applied To**: Performance-critical operations

```python
with metrics.timer(metrics.consolidation_duration, {"trigger_reason": "boundary"}):
    await consolidate(session_id)
```

**Benefit**: Easy latency measurement, Prometheus histograms

### 8.11 Model Registry for Vector Compatibility

**Applied To**: Embedding operations

```python
if ModelRegistry.can_compare(model_a_id, model_b_id):
    similarity = cosine_similarity(vec_a, vec_b)
else:
    raise IncompatibleModelsError(model_a, model_b)
```

**Benefit**: Prevents "model soup" from mixing incompatible embeddings

### 8.12 Unified LLM Extraction (Janus 3.5)

**Applied To**: Consolidation pipeline

```python
# Single LLM call extracts everything
extraction = await unified_extractor.extract(text)
# Returns: ExtractionResult(entities, facts, summary)
```

**Benefit**: Simpler architecture, single LLM call, consistent extraction

---

## 9. Database Schema

### 9.1 Sessions Table

```sql
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMP DEFAULT NOW(),
    last_activity TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);
```

### 9.2 Episodes Table

```sql
CREATE TABLE episodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    turn_start INT NOT NULL,
    turn_end INT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_episodes_session ON episodes(session_id);
```

### 9.3 Episode Vectors Table

```sql
CREATE TABLE episode_vectors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    episode_id UUID REFERENCES episodes(id) ON DELETE CASCADE,
    vector_type VARCHAR(20) NOT NULL,  -- 'summary', 'centroid', 'user_turn'
    embedding VECTOR(384) NOT NULL,
    turn_index INT  -- For user_turn type
);

-- Partial HNSW indexes for efficient search
CREATE INDEX idx_vec_summary ON episode_vectors
USING hnsw (embedding vector_cosine_ops)
WHERE vector_type = 'summary';

CREATE INDEX idx_vec_turns ON episode_vectors
USING hnsw (embedding vector_cosine_ops)
WHERE vector_type = 'user_turn';
```

### 9.4 Entities Table

```sql
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    entity_type VARCHAR(50),
    embedding VECTOR(384),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(session_id, name)
);

CREATE INDEX idx_entities_session ON entities(session_id);
CREATE INDEX idx_entities_embedding ON entities
USING hnsw (embedding vector_cosine_ops);
```

### 9.5 Facts Table (Janus 3.5 - Simplified)

```sql
-- Simplified facts table - no versioning, simple upsert
CREATE TABLE facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    subject_entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    predicate VARCHAR(255) NOT NULL,
    object TEXT NOT NULL,
    confidence FLOAT DEFAULT 0.9 CHECK (confidence >= 0 AND confidence <= 1),
    source_episode_id UUID REFERENCES episodes(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ,

    -- Unique constraint for simple upsert (replaces versioning)
    CONSTRAINT facts_unique_subject_predicate
        UNIQUE (session_id, subject_entity_id, predicate)
);

CREATE INDEX idx_facts_session ON facts(session_id);
CREATE INDEX idx_facts_subject ON facts(subject_entity_id);
CREATE INDEX idx_facts_session_subject ON facts(session_id, subject_entity_id);
```

*Note: Janus 3.5 removes fact versioning. Facts are now upserted (updated in place) rather than creating new versions.*

### 9.6 Audit Log Table

```sql
-- Compliance tracking for GDPR and debugging
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    event_type VARCHAR(50) NOT NULL,  -- 'create', 'update', 'delete', 'consolidate'
    entity_type VARCHAR(50) NOT NULL,  -- 'session', 'episode', 'fact', 'entity'
    entity_id UUID,
    details JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_audit_log_session ON audit_log(session_id);
CREATE INDEX idx_audit_log_event_type ON audit_log(event_type);
CREATE INDEX idx_audit_log_created ON audit_log(created_at);
```

*Note: Janus 3.5 removes `entity_embeddings` (Neo4j bridge), `migration_checkpoints`, and `feature_flags` tables.*

---

## 10. Production Features

### 10.1 High Availability Patterns

| Feature | Implementation |
|---------|---------------|
| Crash Recovery | Redis XAUTOCLAIM for abandoned messages |
| Transactional Safety | PostgreSQL transactions for episode+vectors |
| Graceful Shutdown | SIGINT/SIGTERM handlers in consumer |
| Connection Pooling | asyncpg pool (5-20 connections) |
| Health Checks | `/health` endpoint checks DB and Redis |
| Idempotent Operations | PostgreSQL ON CONFLICT for upserts |

### 10.2 Performance Optimizations

| Optimization | Impact |
|--------------|--------|
| Parallel Retrieval | Entity + episode searches run simultaneously |
| Partial HNSW Indexes | O(log N) search for different vector types |
| Stream Retention | XTRIM MAX~1000 prevents unbounded growth |
| Thread Pool Execution | Embeddings off event loop |
| LLM Rate Limiting | Semaphore-based concurrency control |
| Unified Extraction | Single LLM call replaces 3-phase pipeline |

### 10.3 Scalability Considerations

| Aspect | Design Decision |
|--------|-----------------|
| Session Isolation | Each session has own Redis stream + DB records |
| Stateless API | Multiple API instances behind load balancer |
| Async Consumer | Single consumer handles multiple sessions |
| Database Indexing | Indexes on session_id, subject_entity_id |
| Lightweight Thresholds | Redis-backed persistence with in-memory cache |

### 10.4 Observability Stack

| Component | Purpose |
|-----------|---------|
| OpenTelemetry | Distributed tracing across API/Consumer |
| Prometheus | Metrics collection and alerting |
| Auto-instrumentation | asyncpg, redis automatically traced |

---

## 11. Testing Strategy

### 11.1 Test Files

| File | Purpose |
|------|---------|
| `tests/test_threshold.py` | Adaptive threshold, cold start, boundaries |
| `tests/test_parallel_retrieval.py` | Parallelization verification |

### 11.2 Key Test Cases

**Threshold Tests**:
```python
test_cold_start_initialization()      # Verify pre-calculated velocities
test_first_input_no_boundary()        # No velocity on first input
test_dissimilar_vectors_trigger_boundary()  # Large jump detection
test_rolling_window()                 # Window trimming
```

**Retrieval Tests**:
```python
test_parallel_retrieval_timing()      # <0.8s for 0.5s+0.5s tasks
test_graph_search_waits_for_entities()  # Sequential dependency
```

---

## 12. Quick Start Reference

### 12.1 Start Infrastructure

```bash
docker-compose up -d
```

### 12.2 Install Dependencies

```bash
pip install -r requirements.txt
```

### 12.3 Configure Environment

```bash
cp .env.example .env
# Edit .env with OPENAI_API_KEY
```

### 12.4 Start API

```bash
uvicorn api.main:app --reload
# Runs at http://localhost:8000
```

### 12.5 Start Consumer

```bash
python -m consumer.run_consumer
```

### 12.6 Test Endpoints

```bash
# Create session
curl -X POST http://localhost:8000/api/v1/sessions

# Chat
curl -X POST http://localhost:8000/api/v1/sessions/{id}/chat \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello, I am testing the memory system"}'

# Debug endpoints
curl http://localhost:8000/api/v1/sessions/{id}/stm
curl http://localhost:8000/api/v1/sessions/{id}/episodes
curl http://localhost:8000/api/v1/sessions/{id}/facts
```

---

## Technical Decisions and Trade-offs

| Decision | Rationale | Trade-offs |
|----------|-----------|------------|
| Redis Streams (no RQ) | Native crash recovery | More low-level code |
| Async/await everywhere | Max throughput | More complex debugging |
| pgvector HNSW | Fast approximate search | Not exact, needs tuning |
| Partial indexes | Multiple vector types | More complex schema |
| Simple upsert facts | Reduced complexity | No history preservation |
| SentenceTransformer | Lightweight, fast | Lower quality than large models |
| Unified LLM extraction | Single call, consistent output | LLM cost per consolidation |
| Redis threshold store | Persistence across restarts | Redis dependency |
| PostgreSQL only | Simpler ops, no Neo4j/Kafka | No graph traversal, no event sourcing |
| Model registry | Prevent embedding mismatch | Version tracking overhead |
| OpenTelemetry tracing | Full observability | Performance overhead |

---

## File Cross-References

| Feature | Implementation | Configuration | Tests |
|---------|---------------|---------------|-------|
| Chat Flow | `api/routes.py:chat()` | config.py | Manual |
| Consolidation | `consumer/consolidation.py` | MIN_TURNS_FOR_CONSOLIDATION | N/A |
| Boundary Detection | `core/threshold.py` | COLD_START_VELOCITIES | test_threshold.py |
| Parallel Retrieval | `core/retrieval.py:build_context()` | RETRIEVAL_TOP_K_* | test_parallel_retrieval.py |
| Unified Extraction | `core/unified_extractor.py` | EXTRACTION_MODEL | N/A |
| Model Registry | `core/model_registry.py` | EMBEDDING_MODEL_ID | N/A |
| Vector Storage | `db/episodes.py`, `db/facts.py` | N/A | N/A |
| STM Management | `db/stm.py` | STM_STREAM_PREFIX | N/A |
| LLM Integration | `services/llm_service.py` | OPENAI_API_KEY | N/A |
| Tracing | `infra/tracing.py` | OTEL_* | N/A |
| Metrics | `infra/metrics.py` | PROMETHEUS_* | N/A |

---

## Summary

Janus 3.5 is a streamlined, production-ready memory system that elegantly separates concerns between a fast chat API and asynchronous memory consolidation. Key architectural highlights:

1. **Dual-Process Design**: Nervous System (API) + Cortex (Consumer) connected via Redis Streams
2. **Core Features**:
   - Crash recovery via XAUTOCLAIM
   - Parallelized retrieval with asyncio.gather
   - Partial HNSW indexes for O(log N) search
   - Cold start velocities for stable thresholds
   - Native asyncio consumer (no RQ)
3. **Unified Extraction** (Janus 3.5):
   - Single LLM call extracts entities, facts, and summary
   - Replaces GLiNER + spaCy + LLM enrichment pipeline
   - Simpler architecture, consistent output
4. **Simplified Storage** (Janus 3.5):
   - PostgreSQL + pgvector for all persistent data
   - Redis Streams for STM and threshold persistence
   - Simple upsert facts (no versioning)
   - Removed: Neo4j, Kafka, dual-write, feature flags
5. **Observability**:
   - OpenTelemetry distributed tracing
   - Prometheus metrics for key components
6. **Model Registry**: Prevents "model soup" from mixing incompatible embeddings
7. **Type Safety**: Comprehensive Pydantic models
8. **Scalability**: Stateless API, async consumer, session isolation
9. **Production Ready**: Graceful shutdown, health checks, audit logging

The architecture achieves <100ms TTFT for chat through parallel retrieval while maintaining persistent memory through asynchronous consolidation. Janus 3.5 prioritizes simplicity and operational ease over the flexibility of the previous Neo4j/Kafka-enabled architecture.
