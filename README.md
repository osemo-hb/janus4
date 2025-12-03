# Janus3 Production-Ready Memory System

A dual-process conversational memory system with episodic memory, knowledge graph, and vector search.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  NERVOUS SYSTEM (FastAPI)                                       │
│  Handles: Chat requests with <100ms TTFT                        │
│  - Parallel entity/episode retrieval (Fix 2)                    │
│  - Redis Streams STM (Fix 1)                                    │
│  - Adaptive threshold detection (Fix 4)                         │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼ (Redis Streams)
┌─────────────────────────────────────────────────────────────────┐
│  CORTEX (Native asyncio Consumer)                               │
│  Handles: Memory consolidation                                  │
│  - LLM summarization                                            │
│  - Fact extraction                                              │
│  - XACK/XAUTOCLAIM crash recovery (Fix 1, Fix 5)                │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Start Infrastructure

```bash
cd janus3
docker-compose up -d
```

This starts:
- PostgreSQL with pgvector extension
- Redis for STM streams

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 4. Run Migrations

The migrations run automatically when PostgreSQL starts (via docker-entrypoint-initdb.d).

### 5. Start the API

```bash
uvicorn api.main:app --reload
```

API available at: http://localhost:8000

### 6. Start the Consumer (separate terminal)

```bash
python -m consumer.run_consumer
```

## API Endpoints

### Sessions

- `POST /api/v1/sessions` - Create new session
- `GET /api/v1/sessions/{id}` - Get session info
- `DELETE /api/v1/sessions/{id}` - Delete session

### Chat

- `POST /api/v1/sessions/{id}/chat` - Send message

### Debug

- `GET /api/v1/sessions/{id}/stm` - View STM
- `GET /api/v1/sessions/{id}/episodes` - View episodes
- `GET /api/v1/sessions/{id}/facts` - View facts
- `GET /api/v1/sessions/{id}/threshold` - View threshold stats

## Example Usage

```python
import httpx

# Create session
resp = httpx.post("http://localhost:8000/api/v1/sessions")
session_id = resp.json()["session_id"]

# Chat
resp = httpx.post(
    f"http://localhost:8000/api/v1/sessions/{session_id}/chat",
    json={"content": "My wife is vegetarian and we're planning a dinner party."}
)
print(resp.json()["assistant_message"])

# Chat again (memory should persist)
resp = httpx.post(
    f"http://localhost:8000/api/v1/sessions/{session_id}/chat",
    json={"content": "What dietary restrictions should I consider for dinner?"}
)
print(resp.json()["assistant_message"])  # Should mention vegetarian wife
```

## Key Fixes Implemented

| Fix | Issue | Solution |
|-----|-------|----------|
| 1 | Race condition in STM | Redis Streams with XREADGROUP/XACK |
| 2 | Sequential retrieval | asyncio.gather for parallel search |
| 3 | Full table scan | Two partial HNSW indexes |
| 4 | Cold start threshold | Pre-calculated velocities |
| 5 | RQ/Streams conflict | Native asyncio consumer |

## Running Tests

```bash
pytest tests/ -v
```

## File Structure

```
janus3/
├── api/              # FastAPI (Nervous System)
├── consumer/         # Native Stream Consumer (Cortex)
├── core/             # Business logic
├── db/               # Database layer
├── services/         # External services (LLM)
├── tests/            # Test suite
├── config.py         # Settings
├── docker-compose.yml
└── requirements.txt
```
