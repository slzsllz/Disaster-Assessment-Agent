CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS sessions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    title text,
    model_name text NOT NULL DEFAULT '',
    config_path text NOT NULL DEFAULT '',
    system_prompt text NOT NULL DEFAULT '',
    message_count integer NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id bigserial PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role text NOT NULL,
    content text,
    display_content text,
    attachments jsonb,
    images jsonb,
    tool_trace jsonb,
    elapsed_seconds double precision,
    tool_call_count integer,
    created_at timestamptz NOT NULL DEFAULT now(),
    attachment_files jsonb,
    image_files jsonb,
    legend jsonb,
    report_files jsonb
);

CREATE TABLE IF NOT EXISTS assessment_results (
    id bigserial PRIMARY KEY,
    session_id uuid REFERENCES sessions(id) ON DELETE SET NULL,
    task text,
    description text,
    raster_path text,
    geojson_path text,
    overlay_path text,
    summary_path text,
    summary jsonb,
    geom geometry(Geometry, 4326),
    model_file text,
    num_objects integer,
    created_at timestamptz NOT NULL DEFAULT now(),
    raster_file bytea,
    geojson_file bytea,
    overlay_file bytea,
    summary_file bytea
);

CREATE TABLE IF NOT EXISTS error_memory (
    pattern text PRIMARY KEY,
    fix text NOT NULL DEFAULT '',
    hits integer NOT NULL DEFAULT 0,
    updated_at timestamptz NOT NULL DEFAULT now(),
    created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS legend jsonb;
ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS report_files jsonb;

CREATE TABLE IF NOT EXISTS agent_turns (
    id uuid PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    status text NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    user_message_id bigint REFERENCES chat_messages(id) ON DELETE SET NULL,
    assistant_message_id bigint REFERENCES chat_messages(id) ON DELETE SET NULL,
    error_code text,
    error_message text,
    started_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
);

CREATE TABLE IF NOT EXISTS artifacts (
    id uuid PRIMARY KEY,
    session_id uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    kind text NOT NULL,
    original_name text NOT NULL,
    mime_type text NOT NULL,
    size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
    sha256 text NOT NULL,
    relative_path text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created ON chat_messages(session_id, created_at, id);
CREATE INDEX IF NOT EXISTS idx_assessment_created_at ON assessment_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_assessment_session ON assessment_results(session_id);
CREATE INDEX IF NOT EXISTS idx_assessment_task ON assessment_results(task);
CREATE INDEX IF NOT EXISTS idx_assessment_geom ON assessment_results USING gist(geom);
CREATE INDEX IF NOT EXISTS idx_agent_turns_session_started ON agent_turns(session_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id);

ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS turn_id uuid REFERENCES agent_turns(id) ON DELETE SET NULL;
ALTER TABLE assessment_results ADD COLUMN IF NOT EXISTS artifact_ids jsonb;
