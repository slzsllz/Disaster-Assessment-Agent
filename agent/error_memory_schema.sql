CREATE TABLE IF NOT EXISTS error_memory_vectors (
    entry_id TEXT NOT NULL,
    embedding_profile TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding REAL[] NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (entry_id, embedding_profile)
);
