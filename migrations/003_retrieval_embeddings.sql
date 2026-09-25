-- Small retrieval corpus: keep vectors in PostgreSQL arrays until pgvector is available.
CREATE TABLE IF NOT EXISTS retrieval_embeddings (
    source_type text NOT NULL CHECK (source_type IN ('tool', 'artifact')),
    source_id text NOT NULL,
    session_id uuid REFERENCES sessions(id) ON DELETE CASCADE,
    content_sha256 text NOT NULL,
    model_name text NOT NULL,
    dimensions integer NOT NULL CHECK (dimensions > 0),
    embedding real[] NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (source_type, source_id),
    CHECK ((source_type = 'tool' AND session_id IS NULL)
        OR (source_type = 'artifact' AND session_id IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS idx_retrieval_embeddings_session
    ON retrieval_embeddings(session_id, source_type);

ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS retrieval_refs jsonb;
