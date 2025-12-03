# Janus3 Architecture Documentation

## Executive Summary

**Janus3** is a production-ready, dual-process conversational memory system that combines fast chat responses with asynchronous memory consolidation. It implements a sophisticated memory model with:

- **Nervous System (FastAPI)**: Handles real-time chat with sub-100ms TTFT
- **Cortex (Async Consumer)**: Consolidates memories asynchronously
- **Hybrid Memory**: Combines episodic memory, knowledge graphs, and vector search
- **Production Patterns**: Crash recovery, adaptive thresholds, parallel retrieval

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
│   ├── cortex.py            # Main consumer loop, topic boundaries, crash recovery
│   └── run_consumer.py      # Entry point with graceful shutdown
│
├── core/                     # Business logic and ML services
│   ├── embedder.py          # Thread-safe async embedding service
│   ├── entity_linker.py     # GLiNER-based named entity extraction
│   ├── retrieval.py         # Hybrid retrieval (vector + graph)
│   ├── threshold.py         # Adaptive semantic boundary detection
│   └── models.py            # Pydantic data structures
│
├── db/                       # Database layer (repositories + connection)
│   ├── connection.py        # asyncpg pool management with pgvector
│   ├── stm.py               # Redis Streams-based Short-Term Memory
│   ├── episodes.py          # Episode (LTM) CRUD with multi-vector storage
│   ├── facts.py             # Versioned knowledge graph facts and entities
│   └── migrations/          # PostgreSQL schema
│       └── 001_initial.sql
│
├── services/                 # External service integrations
│   └── llm_service.py       # Async OpenAI wrapper
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
| `core/` | Business Logic | Embeddings, entity extraction, retrieval, thresholds |
| `db/` | Database Layer | Repository pattern, asyncpg, Redis Streams |
| `services/` | External APIs | OpenAI integration |

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

### 2.2 Entity Linker (`core/entity_linker.py`)

**Responsibility**: Extract named entities from text

| Property | Value |
|----------|-------|
| Model | GLiNER (urchade/gliner_base) |
| Entity Types | person, organization, location, concept, product, event |
| Pattern | Singleton with lazy loading |

**Behavior**:
- Lazy loads model on first entity extraction request
- Returns empty list if model loading fails (graceful fallback)
- Uses thread pool execution for non-blocking async

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

### 2.5 STM Manager (`db/stm.py`)

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

### 2.6 Episode Repository (`db/episodes.py`)

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

### 2.7 Fact Repository (`db/facts.py`)

**Responsibility**: Versioned knowledge graph management

**Versioning Pattern**:
```sql
-- When updating a fact:
-- 1. Mark old version as not current
UPDATE facts SET is_current = FALSE
WHERE subject_entity_id = $1 AND predicate = $2 AND is_current = TRUE;

-- 2. Create new version
INSERT INTO facts (..., version = max_version + 1, is_current = TRUE);
```

**Entity Management** (nested EntityRepository):
- Get-or-create with normalization (lowercase, trimmed)
- Vector similarity search for entity linking
- UNIQUE constraint on (session_id, name)

### 2.8 LLM Service (`services/llm_service.py`)

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

### 2.9 Cortex Consumer (`consumer/cortex.py`)

**Responsibility**: Background consolidation of STM → LTM

**Processing Loop**:
1. Discover active sessions (24-hour activity window)
2. Check STM stream length for each session
3. Trigger consolidation if ≥ `MIN_TURNS_FOR_CONSOLIDATION` (default: 5)
4. Extract facts via LLM
5. Create episode with vectors
6. Acknowledge processed entries
7. Recover abandoned work via `XAUTOCLAIM`

**Entry Point**: `python -m consumer.run_consumer`

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
| Purpose | STM storage, crash recovery |
| Pattern | Redis Streams |
| Consumer Group | cortex_workers |
| Max Stream Length | 1000 entries |

**Stream Key**: `stm:{session_id}`

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
```

### 6.7 LLM Configuration

```python
OPENAI_API_KEY: str = ""
GENERATION_MODEL: str = "gpt-4o-mini"
```

### 6.8 Entity Extraction Configuration

```python
GLINER_MODEL: str = "urchade/gliner_base"
GLINER_LABELS: List[str] = ["person", "organization", "location", "concept", "product", "event"]
```

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

### 8.7 Versioned Facts

**Applied To**: Facts table

```sql
-- Update existing fact
UPDATE facts SET is_current = FALSE
WHERE subject_entity_id = $1 AND predicate = $2 AND is_current = TRUE;

-- Create new version
INSERT INTO facts (..., version = max_version + 1, is_current = TRUE);
```

**Benefit**: Fact history preserved, soft-delete semantics

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

### 9.5 Facts Table

```sql
CREATE TABLE facts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    subject_entity_id UUID REFERENCES entities(id),
    predicate VARCHAR(255) NOT NULL,
    object_value TEXT NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    source_episode_id UUID REFERENCES episodes(id),
    version INT DEFAULT 1,
    is_current BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_facts_session ON facts(session_id);
CREATE INDEX idx_facts_subject ON facts(subject_entity_id);
CREATE INDEX idx_facts_current ON facts(is_current) WHERE is_current = TRUE;
```

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

### 10.2 Performance Optimizations

| Optimization | Impact |
|--------------|--------|
| Parallel Retrieval | Entity + episode searches run simultaneously |
| Partial HNSW Indexes | O(log N) search for different vector types |
| Stream Retention | XTRIM MAX~1000 prevents unbounded growth |
| Lazy Model Loading | GLiNER loads only when needed |
| Thread Pool Execution | Embeddings off event loop |
| Batch Operations | Multiple embeddings in single call |

### 10.3 Scalability Considerations

| Aspect | Design Decision |
|--------|-----------------|
| Session Isolation | Each session has own Redis stream + DB records |
| Stateless API | Multiple API instances behind load balancer |
| Async Consumer | Single consumer handles multiple sessions |
| Database Indexing | Indexes on session_id, is_current |
| Lightweight Thresholds | Per-session in-memory cache |

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
| Versioned facts | Preserve history | More storage |
| SentenceTransformer | Lightweight, fast | Lower quality than large models |
| GLiNER entities | Custom labels | Slower than spaCy |
| In-memory threshold | Low latency | Lost on restart |

---

## File Cross-References

| Feature | Implementation | Configuration | Tests |
|---------|---------------|---------------|-------|
| Chat Flow | `api/routes.py:chat()` | config.py | Manual |
| Consolidation | `consumer/cortex.py:_consolidate()` | MIN_TURNS_FOR_CONSOLIDATION | N/A |
| Boundary Detection | `core/threshold.py` | COLD_START_VELOCITIES | test_threshold.py |
| Parallel Retrieval | `core/retrieval.py:build_context()` | RETRIEVAL_TOP_K_* | test_parallel_retrieval.py |
| Vector Storage | `db/episodes.py`, `db/facts.py` | N/A | N/A |
| STM Management | `db/stm.py` | STM_STREAM_PREFIX | N/A |
| Entity Extraction | `core/entity_linker.py` | GLINER_* | N/A |
| LLM Integration | `services/llm_service.py` | OPENAI_API_KEY | N/A |

---

## Summary

Janus3 is a sophisticated, production-ready memory system that elegantly separates concerns between a fast chat API and asynchronous memory consolidation. Key architectural highlights:

1. **Dual-Process Design**: Nervous System (API) + Cortex (Consumer) connected via Redis Streams
2. **Critical Features**:
   - Crash recovery via XAUTOCLAIM
   - Parallelized retrieval with asyncio.gather
   - Partial HNSW indexes for O(log N) search
   - Cold start velocities for stable thresholds
   - Native asyncio consumer (no RQ)
3. **Type Safety**: Comprehensive Pydantic models
4. **Scalability**: Stateless API, async consumer, session isolation
5. **Production Ready**: Graceful shutdown, health checks, error handling

The architecture achieves <100ms TTFT for chat through parallel retrieval while maintaining persistent, versioned memory through asynchronous consolidation.
