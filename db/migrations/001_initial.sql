-- Janus3 Initial Schema
-- PostgreSQL + pgvector

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- SESSIONS
-- Multi-session support from day one
-- ============================================
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID,  -- Optional, for future auth integration
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_activity TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_sessions_last_activity ON sessions(last_activity);

-- ============================================
-- ENTITIES (for entity linking)
-- Stores extracted entities with embeddings for vector search
-- ============================================
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,  -- Lowercase, trimmed for deduplication
    entity_type TEXT NOT NULL,      -- person, organization, location, etc.
    embedding vector(384) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(session_id, normalized_name)
);

-- HNSW index for fast vector similarity search on entities
CREATE INDEX idx_entities_embedding ON entities
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_entities_session ON entities(session_id);

-- ============================================
-- EPISODES (LTM - Long Term Memory)
-- Parent table for consolidated conversation segments
-- NOTE: Vectors stored in child table episode_vectors
-- ============================================
CREATE TABLE episodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    turn_start INT NOT NULL,
    turn_end INT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_episodes_session ON episodes(session_id);
CREATE INDEX idx_episodes_created ON episodes(created_at);

-- ============================================
-- EPISODE VECTORS (child table)
-- CRITICAL: pgvector cannot index vector[] arrays with HNSW
-- Each vector type gets its own partial index for O(log N) search
-- ============================================
CREATE TABLE episode_vectors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    episode_id UUID REFERENCES episodes(id) ON DELETE CASCADE,
    vector_type TEXT NOT NULL CHECK (vector_type IN ('summary', 'centroid', 'user_turn')),
    turn_index INT,  -- Position in episode (NULL for summary/centroid)
    embedding vector(384) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Fix 3: TWO partial HNSW indexes
-- Index 1: Fast Episode Lookup (Summaries)
CREATE INDEX idx_vec_summary ON episode_vectors
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
WHERE vector_type = 'summary';

-- Index 2: Granular Turn Lookup (User inputs for "that code block I mentioned")
CREATE INDEX idx_vec_turns ON episode_vectors
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
WHERE vector_type = 'user_turn';

CREATE INDEX idx_episode_vectors_episode ON episode_vectors(episode_id);

-- ============================================
-- FACTS (versioned knowledge graph)
-- Stores subject-predicate-object triples with versioning
-- ============================================
CREATE TABLE facts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    subject_entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    version INT DEFAULT 1,
    confidence FLOAT DEFAULT 0.9 CHECK (confidence >= 0 AND confidence <= 1),
    is_current BOOLEAN DEFAULT TRUE,  -- Versioning flag
    source_episode_id UUID REFERENCES episodes(id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for current facts lookup by entity
CREATE INDEX idx_facts_current ON facts(session_id, subject_entity_id)
WHERE is_current = TRUE;

-- Index for all facts by session
CREATE INDEX idx_facts_session ON facts(session_id, is_current);

-- ============================================
-- CONFLICTS (for future conflict resolution)
-- Tracks contradictory facts for manual or automatic resolution
-- ============================================
CREATE TABLE conflicts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    fact_id_1 UUID REFERENCES facts(id) ON DELETE CASCADE,
    fact_id_2 UUID REFERENCES facts(id) ON DELETE CASCADE,
    resolution_strategy TEXT DEFAULT 'highest_confidence',
    resolved BOOLEAN DEFAULT FALSE,
    resolved_value TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_conflicts_unresolved ON conflicts(session_id, resolved)
WHERE NOT resolved;

-- ============================================
-- NOTE: STM (Short-Term Memory) is stored in Redis Streams
-- Key pattern: stm:{session_id}
-- This enables crash recovery via XREADGROUP/XACK/XAUTOCLAIM
-- ============================================
