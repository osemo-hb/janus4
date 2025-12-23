# Janus 4.0

Dual-process conversational memory backend for stateful LLM applications.

Separates **latency-critical chat handling** from **asynchronous memory consolidation**:

- Sub-100ms time-to-first-token
- Structured long-term memory (episodes, entities, facts)
- Crash-safe background processing
- Memory growth without blocking requests

Ships as a **reusable library** (`janus-core`) and **reference FastAPI server** (`janus-app`).

## Architecture

```
Client
  ↓
FastAPI App ─── Redis Streams (STM) ─── Async Consumer ─── PostgreSQL + pgvector (LTM)
  │                                           │
  ├─ Chat API                                 ├─ Unified extraction
  ├─ Optional tool calling                    ├─ Embedding
  └─ Short-term memory                        ├─ Topic boundary detection
                                              └─ Memory consolidation
```

See [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) for details.

**Core invariant:** User-facing latency is never blocked by memory consolidation.

## Repository Structure

```
janus/
├── libs/janus-core/       # Memory system library (no HTTP dependencies)
├── server/janus-app/      # FastAPI reference implementation
├── docs/ARCHITECTURE.md   # System internals
├── docker-compose.yml     # PostgreSQL + Redis
└── pyproject.toml         # uv workspace
```

## Components

### janus-core

Pure Python library, embeddable in any application.

- OpenAI embeddings (configurable model + dimensionality)
- Unified LLM extraction (entities, facts, summaries in one call)
- LLM-based topic boundary detection
- Hybrid retrieval (episodes + entities + facts)
- Optional LLM reranking
- Memory distillation (episode compression)
- STM → LTM consolidation pipeline
- PostgreSQL + Redis repositories
- Standalone async consumer with crash recovery

### janus-app

Production-ready FastAPI server demonstrating `janus-core` usage.

- Session-based chat API
- Optional agentic tool-calling (Tavily web search)
- Dependency injection + lifecycle management
- OpenTelemetry tracing, Prometheus metrics

## Requirements

- Python 3.10+
- PostgreSQL 14+ with `pgvector`
- Redis 6+
- OpenAI API key
- (Optional) Tavily API key

## Quick Start

```bash
# 1. Start infrastructure
docker-compose up -d

# 2. Install dependencies
uv sync

# 3. Configure environment
cp .env.example .env
# Set OPENAI_API_KEY (required), TAVILY_API_KEY (optional)

# 4. Start API server
uvicorn janus_app.main:app --reload

# 5. Start background consumer
janus-consumer
# or: python -m janus_core.consumer.runner
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/sessions` | Create session |
| `GET /api/v1/sessions/{id}` | Get session info |
| `DELETE /api/v1/sessions/{id}` | Delete session (GDPR-compliant) |
| `POST /api/v1/sessions/{id}/chat` | Send message |
| `GET /api/v1/sessions/{id}/stm` | Debug: short-term memory |
| `GET /api/v1/sessions/{id}/episodes` | Debug: episodes |
| `GET /api/v1/sessions/{id}/facts` | Debug: facts |
| `GET /api/v1/health` | Health check |
| `GET /api/v1/metrics` | Prometheus metrics |

## Example Usage

```python
import httpx

# Create session
resp = httpx.post("http://localhost:8000/api/v1/sessions")
session_id = resp.json()["session_id"]

# Chat (memory persists across messages)
resp = httpx.post(
    f"http://localhost:8000/api/v1/sessions/{session_id}/chat",
    json={"content": "My wife is vegetarian and we're planning a dinner party."}
)
print(resp.json()["assistant_message"])
```

## Configuration

Key settings (see `.env.example`):

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `TAVILY_API_KEY` | No | Web search |
| `EMBEDDING_MODEL` | No | Embedding model |
| `EMBEDDING_DIM` | No | Embedding dimensions |
| `GENERATION_MODEL` | No | Generation model |
| `TOPIC_DETECTION_ENABLED` | No | LLM topic detection |
| `RERANKER_ENABLED` | No | LLM reranking |
| `DISTILLATION_ENABLED` | No | Memory distillation |

Core settings: `janus_core.config`  
App settings: `janus_app.config`

## Use Cases

- Stateful chat applications
- AI assistants with long-term memory
- Research systems exploring conversational memory
- Internal tools requiring persistent semantic context

## Status

Stable architectural baseline. Possible future work: user-level memory scopes, cost/latency budgeting, deterministic control-plane fallbacks.

## License

MIT License. See [`LICENSE`](LICENSE).