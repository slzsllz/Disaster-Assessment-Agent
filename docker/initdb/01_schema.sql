-- =============================================================================
-- Disaster-Assessment-Agent 数据库 Schema
-- 目标库: PostgreSQL 16 + PostGIS 3.4 (docker/postgis/postgis:16-3.4)
--
-- 设计原则:
--   1. 表结构严格对齐 agent/db.py 中各方法的 INSERT/SELECT/UPDATE 字段
--   2. 幂等 -- 全部使用 IF NOT EXISTS, 可重复执行不报错
--   3. 评估结果带 PostGIS geometry, 支持空间查询
--
-- 数据流: 模型/工具输出 -> agent.db.save_assessment() -> assessment_results
--         后端 backend_api.py -> sessions / chat_messages
--         前端 App.vue <- GET /api/sessions, /api/sessions/{id}/messages
-- =============================================================================

-- PostGIS 扩展 (assessment_results.geom 需要)
CREATE EXTENSION IF NOT EXISTS postgis;

-- =============================================================================
-- 1. sessions -- 一次对话会话
--    对应 agent.db.create_session / list_recent_sessions / update_session
-- =============================================================================
CREATE TABLE IF NOT EXISTS sessions (
    id            UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    title         TEXT,
    model_name    TEXT         NOT NULL DEFAULT '',
    config_path   TEXT         NOT NULL DEFAULT '',
    system_prompt TEXT         NOT NULL DEFAULT '',
    message_count INTEGER      NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions (updated_at DESC);

-- =============================================================================
-- 2. chat_messages -- 会话内的每条消息 (user / assistant)
--    对应 agent.db.save_chat_message / get_chat_messages / delete_session_messages
-- =============================================================================
CREATE TABLE IF NOT EXISTS chat_messages (
    id               BIGSERIAL    PRIMARY KEY,
    session_id       UUID         NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role             TEXT         NOT NULL,
    content          TEXT,
    display_content  TEXT,
    attachments      JSONB,
    images           JSONB,
    tool_trace       JSONB,
    elapsed_seconds  DOUBLE PRECISION,
    tool_call_count  INTEGER,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
    ON chat_messages (session_id, created_at);

-- =============================================================================
-- 3. assessment_results -- 模型/工具评估结果 (核心: 模型输出落库)
--    对应 agent.db.save_assessment / query_assessments / query_assessments_within
--    写入方: agent/tools/utils.py::save_assessment_to_db
--      (GeoAI / FloodSegmentation / DamageAssessment 等工具调用)
-- =============================================================================
CREATE TABLE IF NOT EXISTS assessment_results (
    id           BIGSERIAL    PRIMARY KEY,
    session_id   UUID         REFERENCES sessions(id) ON DELETE SET NULL,
    task         TEXT,
    description  TEXT,
    raster_path  TEXT,
    geojson_path TEXT,
    overlay_path TEXT,
    summary_path TEXT,
    summary      JSONB,
    geom         geometry(Polygon, 4326),
    model_file   TEXT,
    num_objects  INTEGER,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_assessment_task       ON assessment_results (task);
CREATE INDEX IF NOT EXISTS idx_assessment_session    ON assessment_results (session_id);
CREATE INDEX IF NOT EXISTS idx_assessment_created_at ON assessment_results (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_assessment_geom        ON assessment_results USING GIST (geom);

-- =============================================================================
-- 4. error_memory -- 跨任务错误记忆 (pattern -> fix)
--    对应 agent.db.save_error / load_all_errors / increment_error_hit
--    写入方: agent/error_memory.py (双写 文件 + DB)
-- =============================================================================
CREATE TABLE IF NOT EXISTS error_memory (
    pattern     TEXT         PRIMARY KEY,
    fix         TEXT         NOT NULL DEFAULT '',
    hits        INTEGER      NOT NULL DEFAULT 0,
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- =============================================================================
-- updated_at 自动维护触发器 (sessions 表)
-- =============================================================================
CREATE OR REPLACE FUNCTION trg_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS sessions_set_updated_at ON sessions;
CREATE TRIGGER sessions_set_updated_at
    BEFORE UPDATE ON sessions
    FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();

DROP TRIGGER IF EXISTS error_memory_set_updated_at ON error_memory;
CREATE TRIGGER error_memory_set_updated_at
    BEFORE UPDATE ON error_memory
    FOR EACH ROW EXECUTE FUNCTION trg_set_updated_at();
