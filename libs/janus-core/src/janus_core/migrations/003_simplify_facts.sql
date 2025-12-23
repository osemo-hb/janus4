-- Migration: 003_simplify_facts.sql
-- Description: Simplify facts table for Janus 3.5
--
-- Changes:
-- 1. Remove versioning (version, is_current columns)
-- 2. Add unique constraint for simple upsert
-- 3. Add updated_at column for tracking
-- 4. Drop entity_embeddings table (was Neo4j bridge)
-- 5. Clean up migration-related tables
-- 6. Update table comments

-- ============================================================
-- 1. Simplify facts table (remove versioning)
-- ============================================================

-- Create new simplified facts table
CREATE TABLE facts_new (
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

-- Migrate current facts only (drop historical versions)
INSERT INTO facts_new (id, session_id, subject_entity_id, predicate, object,
                       confidence, source_episode_id, created_at)
SELECT id, session_id, subject_entity_id, predicate, object,
       confidence, source_episode_id, created_at
FROM facts
WHERE is_current = TRUE;

-- Drop old table and rename
DROP TABLE facts;
ALTER TABLE facts_new RENAME TO facts;

-- Create indexes for the new facts table
CREATE INDEX idx_facts_session ON facts(session_id);
CREATE INDEX idx_facts_subject ON facts(subject_entity_id);
CREATE INDEX idx_facts_session_subject ON facts(session_id, subject_entity_id);

COMMENT ON TABLE facts IS
    'Simplified facts table for Janus 3.5 - no versioning, simple upsert';

-- ============================================================
-- 2. Drop Neo4j bridge table
-- ============================================================

DROP TABLE IF EXISTS entity_embeddings;

-- ============================================================
-- 3. Clean up migration tracking tables
-- ============================================================

-- Clear migration checkpoints (no longer migrating to Neo4j)
DELETE FROM migration_checkpoints WHERE migration_name LIKE '%neo4j%';

-- Update feature flags - remove migration-related flags
DELETE FROM feature_flags WHERE name IN ('use_kafka', 'use_neo4j', 'use_legacy_stm', 'dual_write_mode');

-- ============================================================
-- 4. Update table comments
-- ============================================================

COMMENT ON TABLE entities IS 'Named entities with embeddings for vector search';
COMMENT ON TABLE episodes IS 'Consolidated conversation segments (Long-Term Memory)';
COMMENT ON TABLE episode_vectors IS 'Multi-vector storage for episodes (summary, centroid, turns)';
COMMENT ON TABLE sessions IS 'Active conversation sessions';

-- ============================================================
-- 5. Drop conflicts table (not used in simplified architecture)
-- ============================================================

DROP TABLE IF EXISTS conflicts;

-- ============================================================
-- Done - Janus 3.5 schema ready
-- ============================================================
