-- Migration: 002_architecture_refactor.sql
-- Description: Schema changes for Janus3 v4 architecture refactoring
--
-- Changes:
-- 1. Add embedding model metadata to vectors
-- 2. Create entity_embeddings table (bridge to Neo4j)
-- 3. Add unique constraint on episodes for idempotency
-- 4. Update episode_vectors with model metadata
-- 5. Prepare for facts/entities migration to Neo4j

-- ============================================================
-- 1. Entity Embeddings Table (Bridge to Neo4j)
-- ============================================================
-- Stores entity embeddings for pgvector similarity search
-- The entity data itself lives in Neo4j, this is just for vectors

CREATE TABLE IF NOT EXISTS entity_embeddings (
    entity_id UUID PRIMARY KEY,           -- References Neo4j entity (not FK)
    session_id UUID NOT NULL,
    embedding vector(384) NOT NULL,
    embedding_model_id VARCHAR(64) NOT NULL DEFAULT 'all-MiniLM-L6-v2',
    embedding_version VARCHAR(16) NOT NULL DEFAULT '1.0.0',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast similarity search
CREATE INDEX IF NOT EXISTS idx_entity_embeddings_hnsw ON entity_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Session index for filtering
CREATE INDEX IF NOT EXISTS idx_entity_embeddings_session ON entity_embeddings(session_id);

-- Partial index for current embedding model (faster queries)
CREATE INDEX IF NOT EXISTS idx_entity_embeddings_model ON entity_embeddings(session_id, embedding_model_id)
WHERE embedding_model_id = 'all-MiniLM-L6-v2';

COMMENT ON TABLE entity_embeddings IS
    'Entity embeddings for pgvector search. Entity data in Neo4j, vectors here.';

-- ============================================================
-- 2. Update episode_vectors with model metadata
-- ============================================================
-- Add embedding model tracking columns if they don't exist

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'episode_vectors'
        AND column_name = 'embedding_model_id'
    ) THEN
        ALTER TABLE episode_vectors
        ADD COLUMN embedding_model_id VARCHAR(64) NOT NULL DEFAULT 'all-MiniLM-L6-v2';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'episode_vectors'
        AND column_name = 'embedding_version'
    ) THEN
        ALTER TABLE episode_vectors
        ADD COLUMN embedding_version VARCHAR(16) NOT NULL DEFAULT '1.0.0';
    END IF;
END $$;

-- ============================================================
-- 3. Add idempotency constraint to episodes
-- ============================================================
-- Prevents duplicate episodes during consolidation

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'uniq_episode_turn_range'
    ) THEN
        ALTER TABLE episodes
        ADD CONSTRAINT uniq_episode_turn_range
        UNIQUE (session_id, turn_start, turn_end);
    END IF;
END $$;

COMMENT ON CONSTRAINT uniq_episode_turn_range ON episodes IS
    'Ensures consolidation idempotency - same turn range cannot be consolidated twice';

-- ============================================================
-- 4. Migration tracking table
-- ============================================================
-- Track what data has been migrated to Neo4j

CREATE TABLE IF NOT EXISTS migration_checkpoints (
    id SERIAL PRIMARY KEY,
    migration_name VARCHAR(255) NOT NULL UNIQUE,
    last_processed_id UUID,
    processed_count BIGINT DEFAULT 0,
    status VARCHAR(50) DEFAULT 'in_progress',
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    CONSTRAINT valid_status CHECK (status IN ('in_progress', 'completed', 'failed', 'paused'))
);

COMMENT ON TABLE migration_checkpoints IS
    'Tracks progress of data migrations (e.g., PostgreSQL facts to Neo4j)';

-- ============================================================
-- 5. Add soft-delete support for GDPR
-- ============================================================
-- Add deleted_at column to episodes for soft-delete

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'episodes'
        AND column_name = 'deleted_at'
    ) THEN
        ALTER TABLE episodes
        ADD COLUMN deleted_at TIMESTAMPTZ;
    END IF;
END $$;

-- Partial index excluding deleted episodes
CREATE INDEX IF NOT EXISTS idx_episodes_active ON episodes(session_id)
WHERE deleted_at IS NULL;

-- ============================================================
-- 6. Create audit log table
-- ============================================================
-- Track important events for compliance

CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    session_id UUID,
    entity_type VARCHAR(50),
    entity_id UUID,
    actor VARCHAR(255),
    details JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_log_session ON audit_log(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_event_type ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at);

COMMENT ON TABLE audit_log IS
    'Audit trail for compliance (GDPR erasure, fact changes, etc.)';

-- ============================================================
-- 7. Feature flags table (optional but useful)
-- ============================================================

CREATE TABLE IF NOT EXISTS feature_flags (
    name VARCHAR(100) PRIMARY KEY,
    enabled BOOLEAN DEFAULT false,
    rollout_percentage INTEGER DEFAULT 0,
    metadata JSONB,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert default flags for migration
INSERT INTO feature_flags (name, enabled, metadata) VALUES
    ('use_kafka', false, '{"description": "Use Kafka for message queue"}'),
    ('use_neo4j', false, '{"description": "Use Neo4j for facts"}'),
    ('use_legacy_stm', true, '{"description": "Use Redis Streams for STM"}'),
    ('dual_write_mode', false, '{"description": "Write to both old and new systems"}')
ON CONFLICT (name) DO NOTHING;

COMMENT ON TABLE feature_flags IS
    'Runtime feature flags for gradual migration rollout';

-- ============================================================
-- 8. Cleanup: Add comments for existing tables
-- ============================================================

COMMENT ON TABLE sessions IS 'Active conversation sessions';
COMMENT ON TABLE episodes IS 'Consolidated conversation segments (LTM)';
COMMENT ON TABLE episode_vectors IS 'Multi-vector storage for episodes (summary, centroid, turns)';

-- Mark entities and facts as deprecated (will be migrated to Neo4j)
COMMENT ON TABLE entities IS
    'DEPRECATED: Being migrated to Neo4j. Do not add new records.';
COMMENT ON TABLE facts IS
    'DEPRECATED: Being migrated to Neo4j. Do not add new records.';

-- ============================================================
-- Done
-- ============================================================
