# Janus 4.0 Architecture Documentation

## Executive Summary

**Janus 4.0** is a streamlined, dual-process conversational memory system refactored into a monorepo with a shared core library. It combines fast chat responses with asynchronous memory consolidation.

**Key Changes from 3.5:**
- **Monorepo Structure**: Reorganized into `libs/janus-core` (shared library) and `server/janus-app` (FastAPI reference implementation)
- **Package Management**: Uses `uv` workspace with `pyproject.toml` for each package
- **Clear Separation**: Core memory logic vs. application-specific concerns (CORS, Tavily, OTEL endpoints)

**Architecture Overview:**
- **Nervous System (janus-app)**: FastAPI server with <100ms TTFT
- **Cortex (janus-core consumer)**: Async memory consolidation worker
- **Unified Extraction**: Single LLM call for entities, facts, and summaries
- **Simplified Storage**: PostgreSQL + pgvector + Redis Streams
- **Agentic Chat**: Tool-calling loop with web search via Tavily
- **LLM-Based Features**: Topic detection, reranking, and memory distillation
- **OpenAI Embeddings**: text-embedding-3-large (1536 dimensions)

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Package Architecture](#2-package-architecture)
3. [Core Components](#3-core-components)
4. [Data Flow](#4-data-flow)
5. [State Management](#5-state-management)
6. [External Integrations](#6-external-integrations)
7. [Configuration](#7-configuration)
8. [Entry Points](#8-entry-points)
9. [Design Patterns](#9-design-patterns)
10. [Database Schema](#10-database-schema)
11. [Production Features](#11-production-features)
12. [Testing Strategy](#12-testing-strategy)
13. [Quick Start Reference](#13-quick-start-reference)

---

## 1. Project Structure

```
janus3/
├── pyproject.toml                    # Workspace root configuration
├── docker-compose.yml                # Infrastructure (PostgreSQL, Redis)
├── .env.example                      # Environment template
│
├── libs/                             # Shared libraries
│   └── janus-core/                   # Core memory system library
│       ├── pyproject.toml            # Package configuration
│       ├── src/janus_core/
│       │   ├── __init__.py           # Public API exports
│       │   ├── config.py             # Core settings (DB, Redis, LLM)
│       │   ├── models.py             # Pydantic data structures
│       │   ├── tracing.py            # OpenTelemetry tracer setup
│       │   │
│       │   ├── embedder.py           # OpenAI embedding service
│       │   ├── unified_extractor.py  # Entity/fact/summary extraction
│       │   ├── topic_detector.py     # LLM-based topic boundaries
│       │   ├── retrieval.py          # Hybrid retrieval + context builder
│       │   ├── reranker.py           # LLM-based memory reranking
│       │   ├── distillation.py       # Memory compression
│       │   ├── consolidation.py      # STM → LTM pipeline
│       │   ├── threshold.py          # Semantic velocity (fallback)
│       │   ├── threshold_store.py    # Redis-backed persistence
│       │   ├── model_registry.py     # Embedding model versioning
│       │   │
│       │   ├── llm/                   # LLM service layer
│       │   │   ├── __init__.py
│       │   │   └── service.py         # Async OpenAI wrapper
│       │   │
│       │   ├── db/                    # Database repositories
│       │   │   ├── __init__.py
│       │   │   ├── connection.py      # asyncpg pool management
│       │   │   ├── stm.py             # Redis Streams STM
│       │   │   ├── episodes.py        # Episode CRUD
│       │   │   ├── facts.py           # Facts + entities
│       │   │   └── distilled.py       # Distilled memories
│       │   │
│       │   ├── consumer/              # Async consolidation worker
│       │   │   ├── __init__.py
│       │   │   └── runner.py          # Entry point + graceful shutdown
│       │   │
│       │   ├── migrations/            # SQL schema migrations
│       │   │   ├── 001_initial.sql
│       │   │   ├── 002_architecture_refactor.sql
│       │   │   ├── 003_simplify_facts.sql
│       │   │   └── 004_gpt51_upgrade.sql
│       │   │
│       │   └── scripts/
│       │       └── threshold_update.lua  # Atomic Redis operations
│       │
│       └── tests/                     # Core library tests
│           ├── test_threshold.py
│           ├── test_parallel_retrieval.py
│           ├── test_topic_detector.py
│           ├── test_reranker.py
│           ├── test_distillation.py
│           ├── test_embedder_openai.py
│           └── test_unified_extractor.py
│
├── server/                           # Application servers
│   └── janus-app/                    # FastAPI reference implementation
│       ├── pyproject.toml            # Package config (depends on janus-core)
│       ├── src/janus_app/
│       │   ├── __init__.py
│       │   ├── config.py             # App settings (CORS, Tavily, OTEL)
│       │   ├── main.py               # FastAPI app + lifespan
│       │   ├── routes.py             # API endpoints
│       │   ├── dependencies.py       # Dependency injection
│       │   │
│       │   ├── services/             # App-specific services
│       │   │   ├── __init__.py
│       │   │   ├── agentic.py        # Tool-calling chat handler
│       │   │   ├── tool_executor.py  # Tavily tool execution
│       │   │   └── tavily.py         # Tavily API client
│       │   │
│       │   └── infra/                # Observability (SDK config)
│       │       ├── __init__.py
│       │       ├── tracing.py        # OTEL SDK + exporters
│       │       └── metrics.py        # Prometheus metrics
│       │
│       └── tests/
│           └── test_tavily_integration.py
│
└── docs/
    └── ARCHITECTURE.md               # This file
```

### Package Responsibilities

| Package | Purpose | Dependencies |
|---------|---------|--------------|
| `janus-core` | Core memory logic | openai, asyncpg, redis, pydantic |
| `janus-app` | FastAPI server | janus-core, fastapi, uvicorn, httpx |

### Layer Separation

| Layer | Package | Responsibilities |
|-------|---------|------------------|
| **Core Library** | `janus-core` | Embeddings, extraction, retrieval, consolidation, DB repos |
| **Application** | `janus-app` | HTTP routes, CORS, web search (Tavily), OTEL SDK config |

---

## 2. Package Architecture

### 2.1 janus-core (Library)

The core library is a reusable package that can be imported by any Python application:

```python
from janus_core import (
    settings,
    Embedder,
    UnifiedExtractor,
    TopicDetector,
    HybridRetriever,
    LLMReranker,
    MemoryDistiller,
    ConsolidationService,
)
```

**Public API**:
- `settings` - Core configuration
- `Episode`, `Entity`, `Fact`, `DistilledMemory` - Data models
- `Embedder` - OpenAI embedding service
- `UnifiedExtractor` - Entity/fact/summary extraction
- `TopicDetector` - LLM-based boundary detection
- `HybridRetriever` - Parallel context retrieval
- `LLMReranker` - Memory relevance scoring
- `MemoryDistiller` - Memory compression
- `ConsolidationService` - STM → LTM pipeline

**Entry Point**:
```bash
janus-consumer  # Runs the Cortex consolidation worker
```

### 2.2 janus-app (Server)

The FastAPI reference implementation demonstrates how to use janus-core:

```python
from janus_core import settings, Embedder, HybridRetriever
from janus_app.config import app_settings
from janus_app.services.agentic import AgenticChatHandler
```

**Application-Specific Features**:
- HTTP API with CORS
- Tavily web search integration
- OpenTelemetry SDK configuration
- Prometheus metrics collection

### 2.3 Workspace Configuration

Root `pyproject.toml`:
```toml
[project]
name = "janus"
version = "4.0.0"

[tool.uv.workspace]
members = ["libs/*", "server/*"]
```

Package dependency (janus-app):
```toml
[tool.uv.sources]
janus-core = { workspace = true }
```

---

## 3. Core Components

### Component Interaction Diagram

```
User Message
    │
    ▼
┌─────────────────────────────────────────┐
│  janus-app: chat() endpoint             │
│  1. Embed query → janus_core.Embedder   │
│  2. Store in STM → janus_core.db.stm    │
│  3. Check boundary → TopicDetector      │
│  4. Retrieve context → HybridRetriever  │
│  5. Generate response → AgenticHandler  │
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
│  janus-core: Cortex Consumer            │
│  1. Read STM entries → XREADGROUP       │
│  2. Extract facts → UnifiedExtractor    │
│  3. Create entities → FactRepository    │
│  4. Store episode → EpisodeRepository   │
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

### 3.1 Embedder (`janus_core.embedder`)

**Responsibility**: Generate vector embeddings for text

| Property | Value |
|----------|-------|
| Model | OpenAI text-embedding-3-large |
| Dimensions | 1536 (configurable: 256, 512, 1024, 1536, 3072) |
| Pattern | Singleton with AsyncOpenAI client |

**API**:
```python
from janus_core import Embedder

embedder = Embedder()
vec = await embedder.encode("Hello world")
vecs = await embedder.encode_batch(["Hello", "World"])
```

### 3.2 Unified Extractor (`janus_core.unified_extractor`)

**Responsibility**: Extract entities, facts, and summary in a single LLM call

| Property | Value |
|----------|-------|
| Model | gpt-4o-mini (configurable) |
| Entity Types | person, organization, location, concept, product, event |

**API**:
```python
from janus_core import UnifiedExtractor

extractor = UnifiedExtractor()
result = await extractor.extract(text)
# Returns: ExtractionResult(entities, facts, summary)
```

### 3.3 Topic Detector (`janus_core.topic_detector`)

**Responsibility**: LLM-based topic boundary detection

| Property | Value |
|----------|-------|
| Model | gpt-4o-mini (configurable) |
| Context | Last 4 turns (configurable) |

**API**:
```python
from janus_core import TopicDetector

detector = TopicDetector()
result = await detector.detect_boundary(recent_turns, current_message)
# Returns: TopicBoundaryResult(is_boundary, confidence, topic_label, reasoning)
```

### 3.4 LLM Reranker (`janus_core.reranker`)

**Responsibility**: Rerank retrieved memories by relevance

| Property | Value |
|----------|-------|
| Model | gpt-4o-mini (configurable) |
| Default Top-K | 5 |

**API**:
```python
from janus_core import LLMReranker

reranker = LLMReranker()
ranked = await reranker.rerank(query, items, item_formatter, top_k=5)
# Returns: List[RankedItem(item, original_score, rerank_score, relevance_reason)]
```

### 3.5 Memory Distiller (`janus_core.distillation`)

**Responsibility**: Compress multiple episodes/facts into composite representations

| Property | Value |
|----------|-------|
| Model | gpt-4o-mini (configurable) |
| Min Episodes | 5 (to trigger distillation) |

**API**:
```python
from janus_core import MemoryDistiller

distiller = MemoryDistiller()
distilled = await distiller.distill_episodes(episodes, session_id)
# Returns: DistilledMemory(compressed_content, topic_cluster, embedding, ...)
```

### 3.6 Hybrid Retriever (`janus_core.retrieval`)

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

### 3.7 Consolidation Service (`janus_core.consolidation`)

**Responsibility**: Memory consolidation pipeline

**Simplified Flow**:
```
Turns Input
    ↓
1. Format turns → text block
2. Unified extraction → UnifiedExtractor (single LLM call)
3. Generate embeddings (summary, centroid, user turns)
4. Create episode → EpisodeRepository
5. Persist entities/facts → PostgreSQL
```

### 3.8 Agentic Chat Handler (`janus_app.services.agentic`)

**Responsibility**: Tool-calling loop for chat with web search

| Property | Value |
|----------|-------|
| Model | gpt-4o-mini (configurable) |
| Max Iterations | 3 (configurable) |
| Tools | tavily_search, tavily_extract |

**Flow**:
```
1. Call LLM with tools enabled
2. If tool_calls present → execute tools
3. Append tool results to messages
4. Call LLM again
5. Repeat until no tool calls or max iterations
6. Return final response
```

### 3.9 Database Repositories (`janus_core.db`)

| Repository | Purpose |
|------------|---------|
| `STMManager` | Redis Streams short-term memory |
| `EpisodeRepository` | Episode CRUD with multi-vector storage |
| `FactRepository` | Facts + nested EntityRepository |
| `DistilledMemoryRepository` | Compressed memory composites |

---

## 4. Data Flow

### 4.1 Chat Request Flow

```
POST /api/v1/sessions/{id}/chat
│
├─ 1. EMBEDDING PHASE
│   └─ janus_core.Embedder.encode(user_message) → query_vector
│
├─ 2. STM STORAGE
│   └─ janus_core.db.STMManager.add_turn()
│       └─ Redis XADD: stm:{session_id}
│
├─ 3. TOPIC DETECTION (LLM-based)
│   └─ janus_core.TopicDetector.detect_boundary()
│       ├─ Returns: is_boundary, confidence, topic_label
│       └─ If boundary: publish to Redis consolidation channel
│
├─ 4. PARALLEL RETRIEVAL
│   └─ janus_core.HybridRetriever.build_context()
│       ├─ Task 1 (parallel): _search_entities()
│       ├─ Task 2 (parallel): _search_episodes()
│       ├─ Task 3 (sequential): _search_graph()
│       └─ Task 4 (parallel): _get_stm_turns()
│
├─ 5. CONTEXT BUILDING
│   └─ janus_core.retrieval.ContextBuilder.build_system_prompt()
│
├─ 6. AGENTIC LLM GENERATION
│   └─ janus_app.services.AgenticChatHandler.chat()
│       ├─ Call LLM with Tavily tool schemas
│       ├─ Execute tools if requested
│       └─ Return final response
│
└─ 7. RESPONSE STORAGE
    └─ janus_core.db.STMManager.add_turn()
```

### 4.2 Memory Consolidation Flow

```
janus-consumer (Cortex Consumer)
│
├─ 1. SESSION DISCOVERY
│   └─ Poll for sessions with pending turns
│
├─ 2. STM CHECK
│   └─ XREADGROUP for new entries
│
├─ 3. CONSOLIDATION
│   └─ janus_core.ConsolidationService.consolidate()
│       ├─ UnifiedExtractor.extract() → entities, facts, summary
│       ├─ Embedder.encode_batch() → vectors
│       ├─ EpisodeRepository.create_episode()
│       └─ FactRepository.upsert_facts()
│
└─ 4. ACKNOWLEDGMENT
    └─ Redis XACK
```

---

## 5. State Management

### 5.1 State Storage Layers

| Layer | Storage | Persistence | Purpose |
|-------|---------|-------------|---------|
| **STM** | Redis Streams | Append-only | Raw turns, immediate context |
| **LTM** | PostgreSQL | ACID | Episodes, facts, entities |
| **Threshold** | Redis + Memory | TTL-based | Velocity history |

### 5.2 Session Lifecycle

```
CREATE_SESSION (POST /sessions)
├─ Create sessions record (PostgreSQL)
├─ Create STM consumer group (Redis XGROUP CREATE)
└─ Return session_id

CHAT (POST /sessions/{id}/chat)
├─ Update last_activity timestamp
├─ Add turn to STM stream
├─ Detect topic boundary
└─ Return response

DELETE_SESSION (DELETE /sessions/{id})
├─ Delete from sessions (cascades)
├─ Delete Redis STM stream
├─ Delete Redis threshold state
└─ Log to audit_log (GDPR)
```

---

## 6. External Integrations

### 6.1 OpenAI Integration

| Service | Purpose | Package |
|---------|---------|---------|
| Embeddings | text-embedding-3-large | janus-core |
| LLM | gpt-4o-mini | janus-core |

### 6.2 PostgreSQL with pgvector

| Property | Value |
|----------|-------|
| Driver | asyncpg (async, no ORM) |
| Extensions | vector, uuid-ossp |
| Indexes | HNSW for vector search |

### 6.3 Redis

| Purpose | Pattern |
|---------|---------|
| STM storage | Redis Streams |
| Threshold persistence | Key-value with TTL |
| Consolidation signals | Pub/Sub |

### 6.4 Tavily (janus-app only)

| Property | Value |
|----------|-------|
| Client | httpx.AsyncClient |
| Tools | tavily_search, tavily_extract |

---

## 7. Configuration

### 7.1 Core Settings (`janus_core.config`)

```python
class CoreSettings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://..."

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    STM_STREAM_PREFIX: str = "stm"

    # Embeddings
    OPENAI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIM: int = 1536

    # LLM
    GENERATION_MODEL: str = "gpt-4o-mini"
    EXTRACTION_MODEL: str = "gpt-4o-mini"

    # Topic Detection
    TOPIC_DETECTION_ENABLED: bool = True
    TOPIC_DETECTION_MODEL: str = "gpt-4o-mini"

    # Reranker
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "gpt-4o-mini"

    # Distillation
    DISTILLATION_ENABLED: bool = True
    DISTILLATION_MODEL: str = "gpt-4o-mini"

    # Consolidation
    MIN_TURNS_FOR_CONSOLIDATION: int = 5
    MAX_CONCURRENT_LLM_CALLS: int = 5
```

### 7.2 App Settings (`janus_app.config`)

```python
class AppSettings(BaseSettings):
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]

    # Tavily
    TAVILY_API_KEY: str = ""
    TAVILY_ENABLED: bool = True
    TAVILY_MAX_TOOL_ITERATIONS: int = 3

    # OpenTelemetry
    OTEL_ENABLED: bool = True
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = None

    # Prometheus
    PROMETHEUS_ENABLED: bool = True
```

---

## 8. Entry Points

### 8.1 API Server (janus-app)

```bash
# Development
uvicorn janus_app.main:app --reload

# Or via Python
python -m janus_app.main
```

### 8.2 Cortex Consumer (janus-core)

```bash
# Via entry point
janus-consumer

# Or via Python
python -m janus_core.consumer.runner
```

### 8.3 API Routes

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
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |

---

## 9. Design Patterns

### 9.1 Singleton with Lazy Loading

Applied to: Embedder, LLMService, UnifiedExtractor

```python
class Embedder:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

### 9.2 Repository Pattern

Applied to: Database access

```python
class EpisodeRepository:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def create_episode(...) → Episode
    async def search_by_summary(...) → List[Tuple[Episode, float]]
```

### 9.3 Redis Streams for Crash Recovery

```python
# Add work
await redis.xadd(stream_key, {...})

# Read and process
entries = await redis.xreadgroup(group, consumer, {stream_key: ">"})
await redis.xack(stream_key, group, *entry_ids)

# Crash recovery
await redis.xautoclaim(stream_key, group, consumer, min_idle_ms=60000)
```

### 9.4 Parallel Retrieval with asyncio.gather

```python
entities, episodes = await asyncio.gather(
    self._search_entities(query_vec, session_id),
    self._search_episodes(query_vec, session_id)
)
```

### 9.5 Workspace Package Dependencies

```toml
# janus-app depends on janus-core
[tool.uv.sources]
janus-core = { workspace = true }
```

---

## 10. Database Schema

### 10.1 Sessions Table

```sql
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID,
    created_at TIMESTAMP DEFAULT NOW(),
    last_activity TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);
```

### 10.2 Episodes Table

```sql
CREATE TABLE episodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    compressed_state TEXT,
    topic_label VARCHAR(100),
    turn_start INT NOT NULL,
    turn_end INT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 10.3 Episode Vectors Table

```sql
CREATE TABLE episode_vectors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    episode_id UUID REFERENCES episodes(id) ON DELETE CASCADE,
    vector_type VARCHAR(20) NOT NULL,  -- 'summary', 'centroid', 'user_turn'
    embedding VECTOR(1536) NOT NULL,
    turn_index INT
);

CREATE INDEX idx_vec_summary ON episode_vectors
USING hnsw (embedding vector_cosine_ops)
WHERE vector_type = 'summary';
```

### 10.4 Entities Table

```sql
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    entity_type VARCHAR(50),
    embedding VECTOR(1536),
    UNIQUE(session_id, name)
);
```

### 10.5 Facts Table

```sql
CREATE TABLE facts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    subject_entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    predicate VARCHAR(255) NOT NULL,
    canonical_predicate VARCHAR(100),
    object TEXT NOT NULL,
    source_span TEXT,
    confidence FLOAT DEFAULT 0.9,
    source_episode_id UUID REFERENCES episodes(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ,

    CONSTRAINT facts_unique_subject_predicate
        UNIQUE (session_id, subject_entity_id, predicate)
);
```

### 10.6 Distilled Memories Table

```sql
CREATE TABLE distilled_memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    source_episode_ids UUID[] DEFAULT '{}',
    source_fact_ids UUID[] DEFAULT '{}',
    compressed_content TEXT NOT NULL,
    topic_cluster VARCHAR(100),
    time_range_start TIMESTAMPTZ,
    time_range_end TIMESTAMPTZ,
    embedding VECTOR(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 10.7 Audit Log Table

```sql
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
    event_type VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id UUID,
    details JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 11. Production Features

### 11.1 High Availability Patterns

| Feature | Implementation |
|---------|---------------|
| Crash Recovery | Redis XAUTOCLAIM |
| Transactional Safety | PostgreSQL transactions |
| Graceful Shutdown | SIGINT/SIGTERM handlers |
| Connection Pooling | asyncpg pool |
| Health Checks | `/health` endpoint |

### 11.2 Performance Optimizations

| Optimization | Impact |
|--------------|--------|
| Parallel Retrieval | Reduced latency |
| Partial HNSW Indexes | O(log N) search |
| Unified Extraction | Single LLM call |
| Async OpenAI | No thread pool |

### 11.3 Observability

| Component | Purpose |
|-----------|---------|
| OpenTelemetry | Distributed tracing |
| Prometheus | Metrics collection |
| Auto-instrumentation | asyncpg, redis |

---

## 12. Testing Strategy

### 12.1 Test Organization

| Package | Tests |
|---------|-------|
| janus-core | Core logic tests in `libs/janus-core/tests/` |
| janus-app | Integration tests in `server/janus-app/tests/` |

### 12.2 Running Tests

```bash
# Run all tests
pytest

# Run specific package tests
pytest libs/janus-core/tests/
pytest server/janus-app/tests/
```

---

## 13. Quick Start Reference

### 13.1 Start Infrastructure

```bash
docker-compose up -d
```

### 13.2 Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or pip
pip install -e libs/janus-core
pip install -e server/janus-app
```

### 13.3 Configure Environment

```bash
cp .env.example .env
# Edit .env with:
#   OPENAI_API_KEY=sk-...
#   TAVILY_API_KEY=tvly-...  (optional)
```

### 13.4 Start API

```bash
uvicorn janus_app.main:app --reload
# Runs at http://localhost:8000
```

### 13.5 Start Consumer

```bash
janus-consumer
# Or: python -m janus_core.consumer.runner
```

### 13.6 Test Endpoints

```bash
# Create session
curl -X POST http://localhost:8000/api/v1/sessions

# Chat
curl -X POST http://localhost:8000/api/v1/sessions/{id}/chat \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello, I am testing the memory system"}'
```

---

## Technical Decisions and Trade-offs

| Decision | Rationale | Trade-offs |
|----------|-----------|------------|
| Monorepo with uv | Shared code, single version | Build complexity |
| Core library | Reusable across apps | Abstraction overhead |
| App-specific Tavily | Not all apps need web search | Duplication if needed elsewhere |
| Split configuration | Clear separation of concerns | Two config files |
| PostgreSQL only | Simpler ops | No graph traversal |
| OpenAI embeddings | High quality, configurable | API cost |
| LLM features | Better accuracy | Additional latency/cost |

---

## Summary

Janus 4.0 restructures the codebase into a monorepo with clear separation between:

1. **janus-core**: Reusable library with all memory logic
2. **janus-app**: Reference FastAPI implementation

Key architectural highlights:
- **Dual-Process Design**: API + Consumer connected via Redis Streams
- **Parallel Retrieval**: asyncio.gather for entity + episode search
- **Unified Extraction**: Single LLM call for entities, facts, summary
- **LLM-Based Intelligence**: Topic detection, reranking, distillation
- **Clean Separation**: Core logic vs. application concerns
- **Workspace Management**: uv workspace with shared dependencies

The architecture achieves <100ms TTFT while maintaining persistent memory through async consolidation.
