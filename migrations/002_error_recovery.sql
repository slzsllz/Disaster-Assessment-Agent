CREATE TABLE IF NOT EXISTS error_recovery_candidates (
    id bigserial PRIMARY KEY,
    tool_name text NOT NULL,
    error_signature text NOT NULL,
    error_hash text NOT NULL,
    changed_fields jsonb NOT NULL,
    occurrences integer NOT NULL DEFAULT 1,
    status text NOT NULL DEFAULT 'pending',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tool_name, error_hash, changed_fields)
);
