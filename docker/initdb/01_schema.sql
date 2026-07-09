-- =============================================================================
-- Disaster-Assessment-Agent 数据库初始化脚本
-- 数据库: PostgreSQL 16 + PostGIS 3.4
-- =============================================================================

-- 启用扩展
CREATE EXTENSION IF NOT EXISTS "postgis";

-- =============================================================================
-- 1. 会话表 - Streamlit 聊天会话
-- =============================================================================
CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title           TEXT,
    model_name      TEXT,
    config_path     TEXT,
    system_prompt   TEXT,
    message_count   INT DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

-- =============================================================================
-- 2. 聊天消息表 - 会话中的每条消息
-- =============================================================================
CREATE TABLE IF NOT EXISTS chat_messages (
    id              BIGSERIAL PRIMARY KEY,
    session_id      UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    content         TEXT,
    display_content TEXT,
    attachments     JSONB,
    images          JSONB,
    tool_trace      JSONB,
    elapsed_seconds REAL,
    tool_call_count INT,
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_messages(session_id, created_at);

-- =============================================================================
-- 3. 评估结果表 - 工具输出的结构化结果(建筑物检测/洪水分割/车辆检测等)
--    空间数据存 geom 列,大文件(.tif/.png)只存路径
-- =============================================================================
CREATE TABLE IF NOT EXISTS assessment_results (
    id              BIGSERIAL PRIMARY KEY,
    session_id      UUID REFERENCES sessions(id) ON DELETE SET NULL,
    task            TEXT NOT NULL,
    description     TEXT,
    raster_path     TEXT,
    geojson_path    TEXT,
    overlay_path    TEXT,
    summary_path    TEXT,
    summary         JSONB,
    geom            GEOMETRY(Geometry, 4326),
    model_file      TEXT,
    num_objects     INT,
    extra           JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_assess_geom ON assessment_results USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_assess_task ON assessment_results(task);
CREATE INDEX IF NOT EXISTS idx_assess_session ON assessment_results(session_id);
CREATE INDEX IF NOT EXISTS idx_assess_summary ON assessment_results USING GIN(summary);

-- =============================================================================
-- 4. 错误记忆表 - 跨任务错误模式与修复建议
-- =============================================================================
CREATE TABLE IF NOT EXISTS error_memory (
    pattern         TEXT PRIMARY KEY,
    fix             TEXT NOT NULL,
    hits            INT DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

-- =============================================================================
-- 更新触发器 - 自动更新 updated_at
-- =============================================================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_sessions_updated
    BEFORE UPDATE ON sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE OR REPLACE TRIGGER trg_error_memory_updated
    BEFORE UPDATE ON error_memory
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
