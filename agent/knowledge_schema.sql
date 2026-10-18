CREATE TABLE IF NOT EXISTS knowledge_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    size_bytes BIGINT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued','parsing','embedding','ready','failed','needs_ocr')),
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    source TEXT NOT NULL DEFAULT '',
    version TEXT NOT NULL DEFAULT '',
    effective_date TEXT NOT NULL DEFAULT '',
    region TEXT NOT NULL DEFAULT '',
    disaster_type TEXT NOT NULL DEFAULT '',
    chunk_count INTEGER NOT NULL DEFAULT 0,
    embedding_profile TEXT,
    error TEXT NOT NULL DEFAULT '',
    lease_token UUID,
    lease_until TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    locator TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding REAL[] NOT NULL CHECK (cardinality(embedding) > 0),
    embedding_profile TEXT NOT NULL,
    UNIQUE (document_id, ordinal)
);
CREATE INDEX IF NOT EXISTS knowledge_chunks_document_idx ON knowledge_chunks(document_id);
CREATE INDEX IF NOT EXISTS knowledge_chunks_profile_idx ON knowledge_chunks(embedding_profile);
ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS retrieval_refs JSONB;
