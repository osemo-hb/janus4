-- Janus 3.5 GPT-5.1 Preparation Migration
-- IMPORTANT: This is a breaking change - requires fresh database
-- Changes:
--   1. Vector dimensions: 384 -> 1536 (OpenAI text-embedding-3-large)
--   2. Episodes: Add compressed_state, topic_label
--   3. Facts: Add canonical_predicate, source_span
--   4. New table: distilled_memories

-- ============================================
-- STEP 1: Update Entities (1536D vectors)
-- ============================================

-- Drop existing index
DROP INDEX IF EXISTS idx_entities_embedding;

-- Change vector dimension
ALTER TABLE entities
ALTER COLUMN embedding TYPE vector(1536);

-- Recreate HNSW index
CREATE INDEX idx_entities_embedding ON entities
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- ============================================
-- STEP 2: Update Episode Vectors (1536D)
-- ============================================

-- Drop existing indexes
DROP INDEX IF EXISTS idx_vec_summary;
DROP INDEX IF EXISTS idx_vec_turns;

-- Change vector dimension
ALTER TABLE episode_vectors
ALTER COLUMN embedding TYPE vector(1536);

-- Recreate partial HNSW indexes
CREATE INDEX idx_vec_summary ON episode_vectors
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
WHERE vector_type = 'summary';

CREATE INDEX idx_vec_turns ON episode_vectors
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64)
WHERE vector_type = 'user_turn';

-- ============================================
-- STEP 3: Add Episode Compression Fields
-- ============================================

-- Compressed state: 20-40 word abstraction
ALTER TABLE episodes
ADD COLUMN IF NOT EXISTS compressed_state TEXT;

-- Topic label: 2-3 word topic identifier
ALTER TABLE episodes
ADD COLUMN IF NOT EXISTS topic_label VARCHAR(100);

-- Index for clustering by topic
CREATE INDEX IF NOT EXISTS idx_episodes_topic
ON episodes(session_id, topic_label);

-- ============================================
-- STEP 4: Add Canonical Fact Fields
-- ============================================

-- Canonical predicate: normalized schema.org-style
ALTER TABLE facts
ADD COLUMN IF NOT EXISTS canonical_predicate VARCHAR(100);

-- Source span: original text for provenance
ALTER TABLE facts
ADD COLUMN IF NOT EXISTS source_span TEXT;

-- Index for predicate queries
CREATE INDEX IF NOT EXISTS idx_facts_canonical
ON facts(canonical_predicate);

-- Update existing facts with canonical predicates (if any exist)
UPDATE facts
SET canonical_predicate = REPLACE(LOWER(predicate), ' ', '_')
WHERE canonical_predicate IS NULL;

-- ============================================
-- STEP 5: Create Distilled Memories Table
-- ============================================

CREATE TABLE IF NOT EXISTS distilled_memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,

    -- Source tracking
    source_episode_ids UUID[] DEFAULT '{}',
    source_fact_ids UUID[] DEFAULT '{}',

    -- Compressed content
    compressed_content TEXT NOT NULL,
    topic_cluster VARCHAR(100),

    -- Time range covered
    time_range_start TIMESTAMPTZ,
    time_range_end TIMESTAMPTZ,

    -- Embedding for vector search
    embedding vector(1536),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for distilled memories
CREATE INDEX IF NOT EXISTS idx_distilled_session
ON distilled_memories(session_id);

CREATE INDEX IF NOT EXISTS idx_distilled_topic
ON distilled_memories(session_id, topic_cluster);

-- HNSW index for vector search
CREATE INDEX IF NOT EXISTS idx_distilled_embedding
ON distilled_memories
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- ============================================
-- STEP 6: Audit Log Entry
-- ============================================

-- Record migration in audit log
INSERT INTO audit_log (event_type, details)
VALUES (
    'MIGRATION_004',
    '{"description": "GPT-5.1 preparation", "changes": ["vectors 384->1536", "episode compression", "canonical facts", "distilled memories"]}'::jsonb
)
ON CONFLICT DO NOTHING;
