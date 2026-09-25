"""
Disaster-Assessment-Agent 数据库连接层

设计原则:
  1. 懒连接 —— 首次使用时才尝试连库
  2. 持久化优先 —— 聊天与评估写入失败时显式报告，不伪装为成功
  3. 连接池复用 —— 使用 psycopg 连接池，避免频繁建连
  4. 最小侵入 —— 现有代码只需调用 DB.xxx()，不关心连接细节

支持的表: sessions / chat_messages / assessment_results / error_memory
"""

from __future__ import annotations

import atexit
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DatabaseUnavailable(RuntimeError):
    pass

# ---------------------------------------------------------------------------
# psycopg 3 延迟导入
# ---------------------------------------------------------------------------
try:
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
    _HAS_PSYCOPG = True
except Exception:  # noqa: BLE001
    psycopg = None
    sql = None
    dict_row = None
    Jsonb = None
    _HAS_PSYCOPG = False

try:
    from psycopg_pool import ConnectionPool
    _HAS_POOL = True
except Exception:  # noqa: BLE001
    ConnectionPool = None
    _HAS_POOL = False


def _build_dsn() -> str:
    """从环境变量构建连接字符串"""
    user = os.getenv("DB_USER", "disaster_agent")
    password = os.getenv("DB_PASSWORD", "disaster_agent_pwd")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5433")
    dbname = os.getenv("DB_NAME", "disaster_agent")
    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


def is_enabled() -> bool:
    """数据库功能是否启用(环境变量 DB_ENABLED != false 且 psycopg 可用)"""
    return os.getenv("DB_ENABLED", "true").lower() != "false" and _HAS_PSYCOPG


class Database:
    """PostgreSQL + PostGIS 数据库访问层

    使用方式::

        from agent.db import db
        db.save_session(...)
        db.save_chat_message(...)

    Agent 的关键读写路径使用严格方法；部分兼容接口仍返回 None / False。
    """

    _instance: Optional["Database"] = None
    _pool: Any = None  # ConnectionPool
    _checked: bool = False  # 是否已检查过连接(避免反复重试)

    def __new__(cls) -> "Database":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------
    def _ensure_pool(self) -> bool:
        """惰性初始化连接池，返回是否可用

        一旦检测到 DB 不可用，标记 _checked 避免后续重复重试。
        可通过 health_check() 重置后重新探测。
        """
        if not is_enabled():
            return False
        if self._pool is not None:
            return True
        if self._checked:
            if time.monotonic() < getattr(self, "_retry_after", 0):
                return False
            self._checked = False
        try:
            dsn = _build_dsn()
            # 先用单连接快速探测，避免连接池的长时间重试噪音
            probe = psycopg.connect(dsn, connect_timeout=3)
            probe.close()
            # 探测成功，创建连接池
            if _HAS_POOL:
                # 抑制连接池的日志噪音
                pool_logger = logging.getLogger("psycopg.pool")
                pool_logger.setLevel(logging.WARNING)
                self._pool = ConnectionPool(
                    conninfo=dsn,
                    min_size=1,
                    max_size=8,
                    timeout=5,
                )
                self._pool.wait()  # 确保连接池就绪
            logger.info("Database connection pool initialized")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.info("Database unavailable, file-only mode: %s", exc)
            self._pool = None
            self._checked = True  # 标记不可用，避免反复重试
            self._retry_after = time.monotonic() + 5
            return False

    @contextmanager
    def _conn(self):
        """获取数据库连接上下文管理器"""
        if not self._ensure_pool():
            raise RuntimeError("Database not available")
        if _HAS_POOL and self._pool is not None:
            with self._pool.connection() as conn:
                yield conn
        else:
            conn = psycopg.connect(_build_dsn())
            try:
                yield conn
            finally:
                conn.close()

    def migrate(self) -> bool:
        """Apply checked-in migrations once, in order, before serving requests."""
        if not self._ensure_pool():
            return False
        migration_dir = Path(__file__).resolve().parents[1] / "migrations"
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute("""CREATE TABLE IF NOT EXISTS schema_migrations (
                    version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now()
                )""")
                cur.execute("SELECT version FROM schema_migrations")
                applied = {row[0] for row in cur.fetchall()}
                for migration in sorted(migration_dir.glob("[0-9]*.sql")):
                    if migration.name in applied:
                        continue
                    cur.execute(migration.read_text(encoding="utf-8"))
                    cur.execute("INSERT INTO schema_migrations(version) VALUES (%s)", (migration.name,))
                conn.commit()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("Database migration failed: %s", exc)
            return False

    def require_available(self) -> None:
        if not self._ensure_pool():
            raise DatabaseUnavailable("Persistent database is unavailable")

    # ==================================================================
    # 1. Sessions
    # ==================================================================
    def create_session(
        self,
        session_id: str = None,
        model_name: str = "",
        config_path: str = "",
        system_prompt: str = "",
        title: str = "",
    ) -> Optional[str]:
        """创建(或刷新)会话，返回 session_id (UUID)

        传入 session_id 时以其作主键做 upsert, 便于与前端生成的 sessionId
        对齐; 不传时由数据库 gen_random_uuid() 生成。已存在则刷新元数据。
        """
        if not self._ensure_pool():
            return None
        try:
            with self._conn() as conn, conn.cursor(row_factory=dict_row) as cur:
                if session_id:
                    cur.execute(
                        """INSERT INTO sessions
                           (id, title, model_name, config_path, system_prompt)
                           VALUES (%s, %s, %s, %s, %s)
                           ON CONFLICT (id) DO UPDATE
                           SET system_prompt = EXCLUDED.system_prompt,
                               model_name    = EXCLUDED.model_name,
                               config_path   = EXCLUDED.config_path,
                               title         = COALESCE(sessions.title, EXCLUDED.title),
                               updated_at    = now()
                           RETURNING id""",
                        (session_id, title or None, model_name, config_path, system_prompt),
                    )
                else:
                    cur.execute(
                        """INSERT INTO sessions (title, model_name, config_path, system_prompt)
                           VALUES (%s, %s, %s, %s) RETURNING id""",
                        (title or None, model_name, config_path, system_prompt),
                    )
                row = cur.fetchone()
                conn.commit()
                return str(row["id"]) if row else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("create_session failed: %s", exc)
            return None

    def update_session(self, session_id: str, **kwargs) -> bool:
        """更新会话字段"""
        if not self._ensure_pool():
            return False
        allowed = {"model_name", "config_path", "system_prompt", "title"}
        sets = {k: v for k, v in kwargs.items() if k in allowed}
        if not sets:
            return False
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cols = sql.SQL(", ").join(
                    sql.SQL("{} = %s").format(sql.Identifier(k)) for k in sets
                )
                cur.execute(
                    sql.SQL("UPDATE sessions SET {} WHERE id = %s").format(cols),
                    (*sets.values(), session_id),
                )
                conn.commit()
                return cur.rowcount > 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("update_session failed: %s", exc)
            return False

    def rename_session(self, session_id: str, title: str) -> bool:
        """重命名会话标题"""
        return self.update_session(session_id, title=title)

    def delete_session(self, session_id: str) -> bool:
        """删除会话及其全部消息(级联删除)"""
        if not self._ensure_pool():
            return False
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM assessment_results WHERE session_id = %s", (session_id,))
                cur.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
                conn.commit()
                return cur.rowcount > 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("delete_session failed: %s", exc)
            return False

    def get_session(self, session_id: str) -> Optional[dict]:
        if not self._ensure_pool():
            return None
        try:
            with self._conn() as conn, conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT * FROM sessions WHERE id = %s", (session_id,))
                return cur.fetchone()
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_session failed: %s", exc)
            return None

    def list_recent_sessions(self, limit: int = 20) -> List[dict]:
        if not self._ensure_pool():
            return []
        try:
            with self._conn() as conn, conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """SELECT s.id, s.title, s.model_name, s.config_path,
                              s.message_count, s.created_at, s.updated_at,
                              (SELECT content FROM chat_messages
                               WHERE session_id = s.id AND role = 'user'
                               ORDER BY created_at ASC LIMIT 1) AS first_message
                       FROM sessions s
                       ORDER BY s.updated_at DESC LIMIT %s""",
                    (limit,),
                )
                return cur.fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("list_recent_sessions failed: %s", exc)
            raise DatabaseUnavailable("Could not load sessions") from exc

    # ==================================================================
    # 2. Chat Messages
    # ==================================================================
    def save_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        display_content: str = "",
        attachments: list = None,
        images: list = None,
        tool_trace: list = None,
        elapsed_seconds: float = None,
        tool_call_count: int = None,
        attachment_files: list = None,
        image_files: list = None,
        legend: list = None,
        report_files: list = None,
    ) -> Optional[int]:
        """保存一条聊天消息，返回消息 ID

        Args:
            attachment_files: 用户上传的附件文件二进制数据列表 [{name, mime_type, data_base64}]
            image_files: 模型输出的图片二进制数据列表 [{name, mime_type, data_base64}]
            legend: 图例数据列表 [{label, color}]
            report_files: AI 生成的 PDF 报告二进制数据列表 [{name, mime_type, data_base64}]
        """
        if not self._ensure_pool():
            return None
        try:
            with self._conn() as conn, conn.cursor() as cur:
                base_cols = [
                    "session_id", "role", "content", "display_content", "attachments",
                    "images", "tool_trace", "elapsed_seconds", "tool_call_count",
                    "attachment_files", "image_files", "legend", "report_files",
                ]
                base_vals = [
                    session_id,
                    role,
                    content,
                    display_content or None,
                    Jsonb(attachments) if attachments else None,
                    Jsonb(images) if images else None,
                    Jsonb(tool_trace) if tool_trace else None,
                    elapsed_seconds,
                    tool_call_count,
                    Jsonb(attachment_files) if attachment_files else None,
                    Jsonb(image_files) if image_files else None,
                    Jsonb(legend) if legend else None,
                    Jsonb(report_files) if report_files else None,
                ]

                cols_sql = sql.SQL(", ").join(sql.Identifier(c) for c in base_cols)
                placeholders = sql.SQL(", ").join(sql.Placeholder() * len(base_vals))
                cur.execute(
                    sql.SQL("INSERT INTO chat_messages ({}) VALUES ({}) RETURNING id").format(
                        cols_sql, placeholders
                    ),
                    base_vals,
                )
                row = cur.fetchone()
                # 维护 sessions 表的 message_count 和 updated_at
                cur.execute(
                    """UPDATE sessions
                       SET message_count = message_count + 1,
                           updated_at = now()
                       WHERE id = %s""",
                    (session_id,),
                )
                conn.commit()
                return row[0] if row else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("save_chat_message failed: %s", exc)
            return None

    def start_turn(
        self, session_id: str, turn_id: str, content: str,
        display_content: str, attachments: list[dict],
    ) -> int:
        """Record the user message and running turn atomically."""
        self.require_available()
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO agent_turns(id, session_id, status)
                   VALUES (%s, %s, 'running')""",
                (turn_id, session_id),
            )
            cur.execute(
                """INSERT INTO chat_messages
                   (session_id, role, content, display_content, attachments)
                   VALUES (%s, 'user', %s, %s, %s) RETURNING id""",
                (session_id, content, display_content, Jsonb(attachments)),
            )
            message_id = cur.fetchone()[0]
            cur.execute(
                "UPDATE agent_turns SET user_message_id = %s WHERE id = %s",
                (message_id, turn_id),
            )
            cur.execute(
                """UPDATE sessions SET message_count = message_count + 1,
                   updated_at = now() WHERE id = %s""",
                (session_id,),
            )
            conn.commit()
            return message_id

    def complete_turn(
        self, session_id: str, turn_id: str, content: str,
        display_content: str, attachments: list[dict], images: list[dict],
        tool_trace: list[dict], elapsed_seconds: float,
        legend: list[dict], retrieval_refs: list[dict] | None = None,
    ) -> int:
        """Write the answer and mark its turn complete in one transaction."""
        self.require_available()
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO chat_messages
                   (session_id, role, content, display_content, attachments,
                    images, tool_trace, elapsed_seconds, tool_call_count,
                    legend, retrieval_refs)
                   VALUES (%s, 'assistant', %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (session_id, content, display_content, Jsonb(attachments),
                 Jsonb(images), Jsonb(tool_trace), elapsed_seconds,
                 len(tool_trace), Jsonb(legend), Jsonb(retrieval_refs or [])),
            )
            message_id = cur.fetchone()[0]
            cur.execute(
                """UPDATE agent_turns SET status = 'completed',
                   assistant_message_id = %s, updated_at = now(),
                   finished_at = now() WHERE id = %s AND status = 'running'""",
                (message_id, turn_id),
            )
            if cur.rowcount != 1:
                raise RuntimeError("Turn is missing or already finished")
            cur.execute(
                """UPDATE sessions SET message_count = message_count + 1,
                   updated_at = now() WHERE id = %s""",
                (session_id,),
            )
            conn.commit()
            return message_id

    def fail_turn(self, session_id: str, turn_id: str, message: str, code: str = "execution_error") -> bool:
        """Persist a short user-visible failure; never store a traceback here."""
        if not self._ensure_pool():
            return False
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """UPDATE agent_turns SET status = 'failed', error_code = %s,
                       error_message = %s, updated_at = now(), finished_at = now()
                       WHERE id = %s AND session_id = %s AND status = 'running'""",
                    (code, message[:500], turn_id, session_id),
                )
                if cur.rowcount != 1:
                    return False
                cur.execute("DELETE FROM assessment_results WHERE turn_id = %s", (turn_id,))
                cur.execute(
                    """INSERT INTO chat_messages(session_id, role, content, display_content)
                       VALUES (%s, 'assistant', %s, %s)""",
                    (session_id, message, message),
                )
                cur.execute(
                    """UPDATE sessions SET message_count = message_count + 1,
                       updated_at = now() WHERE id = %s""",
                    (session_id,),
                )
                conn.commit()
                return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("fail_turn failed: %s", exc)
            return False

    def save_artifact(self, record: dict) -> None:
        self.require_available()
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO artifacts
                   (id, session_id, kind, original_name, mime_type, size_bytes,
                    sha256, relative_path)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (record["id"], record["session_id"], record["kind"],
                 record["original_name"], record["mime_type"],
                 record["size_bytes"], record["sha256"], record["relative_path"]),
            )
            conn.commit()

    def get_artifact(self, artifact_id: str) -> Optional[dict]:
        self.require_available()
        try:
            with self._conn() as conn, conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT * FROM artifacts WHERE id = %s", (artifact_id,))
                return cur.fetchone()
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_artifact failed: %s", exc)
            raise DatabaseUnavailable("Could not load artifact") from exc

    def list_session_artifacts(self, session_id: str, limit: int = 200) -> List[dict]:
        """Only return files owned by this session, newest first."""
        self.require_available()
        with self._conn() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT id, session_id, kind, original_name, mime_type,
                          size_bytes, sha256, relative_path, created_at
                   FROM artifacts WHERE session_id = %s
                   ORDER BY created_at DESC, id DESC LIMIT %s""",
                (session_id, max(1, min(limit, 500))),
            )
            return cur.fetchall()

    def load_retrieval_embeddings(
        self, source_type: str, source_ids: list[str], session_id: str | None = None,
    ) -> dict[str, dict]:
        """Load only vectors for the requested source scope."""
        if not source_ids:
            return {}
        self.require_available()
        with self._conn() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT source_id, content_sha256, model_name, dimensions,
                          embedding FROM retrieval_embeddings
                   WHERE source_type = %s AND source_id = ANY(%s)
                     AND session_id IS NOT DISTINCT FROM %s::uuid""",
                (source_type, source_ids, session_id),
            )
            return {row["source_id"]: row for row in cur.fetchall()}

    def upsert_retrieval_embeddings(self, rows: list[dict]) -> None:
        """Persist one embedding version per source and content hash."""
        if not rows:
            return
        self.require_available()
        with self._conn() as conn, conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO retrieval_embeddings
                   (source_type, source_id, session_id, content_sha256,
                    model_name, dimensions, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (source_type, source_id) DO UPDATE SET
                     session_id = EXCLUDED.session_id,
                     content_sha256 = EXCLUDED.content_sha256,
                     model_name = EXCLUDED.model_name,
                     dimensions = EXCLUDED.dimensions,
                     embedding = EXCLUDED.embedding,
                     updated_at = now()""",
                [
                    (row["source_type"], row["source_id"], row["session_id"],
                     row["content_sha256"], row["model_name"],
                     row["dimensions"], row["embedding"])
                    for row in rows
                ],
            )
            conn.commit()

    def update_message_artifact_refs(
        self, message_id: int, attachments: list[dict],
        images: list[dict], reports: list[dict],
    ) -> None:
        """Replace legacy inline file data after copies have been verified."""
        self.require_available()
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """UPDATE chat_messages SET attachments = %s, images = %s,
                   report_files = %s, attachment_files = NULL, image_files = NULL
                   WHERE id = %s""",
                (Jsonb(attachments), Jsonb(images), Jsonb(reports), message_id),
            )
            if cur.rowcount != 1:
                raise RuntimeError("Legacy message disappeared during migration")
            conn.commit()

    def clear_session_data(self, session_id: str) -> None:
        self.require_available()
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM assessment_results WHERE session_id = %s", (session_id,))
            cur.execute("DELETE FROM agent_turns WHERE session_id = %s", (session_id,))
            cur.execute("DELETE FROM chat_messages WHERE session_id = %s", (session_id,))
            cur.execute("DELETE FROM retrieval_embeddings WHERE session_id = %s", (session_id,))
            cur.execute("DELETE FROM artifacts WHERE session_id = %s", (session_id,))
            cur.execute(
                "UPDATE sessions SET message_count = 0, updated_at = now() WHERE id = %s",
                (session_id,),
            )
            conn.commit()

    def list_turns(self, session_id: str) -> List[dict]:
        self.require_available()
        with self._conn() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT id, session_id, status, user_message_id,
                   assistant_message_id, error_code, error_message,
                   started_at, updated_at, finished_at
                   FROM agent_turns WHERE session_id = %s
                   ORDER BY started_at ASC, id ASC""",
                (session_id,),
            )
            return cur.fetchall()

    def reconcile_stale_turns(self) -> int:
        """Close turns abandoned by a process that stopped over an hour ago."""
        self.require_available()
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT id, session_id FROM agent_turns
                   WHERE status = 'running' AND started_at < now() - interval '1 hour'
                   FOR UPDATE SKIP LOCKED"""
            )
            rows = cur.fetchall()
            for turn_id, session_id in rows:
                message = "任务中断，结果未完成。"
                cur.execute(
                    """UPDATE agent_turns SET status = 'failed',
                       error_code = 'interrupted', error_message = %s,
                       updated_at = now(), finished_at = now() WHERE id = %s""",
                    (message, turn_id),
                )
                cur.execute("DELETE FROM assessment_results WHERE turn_id = %s", (turn_id,))
                cur.execute(
                    """INSERT INTO chat_messages(session_id, role, content, display_content)
                       VALUES (%s, 'assistant', %s, %s)""",
                    (session_id, message, message),
                )
                cur.execute(
                    """UPDATE sessions SET message_count = message_count + 1,
                       updated_at = now() WHERE id = %s""",
                    (session_id,),
                )
            conn.commit()
            return len(rows)

    def update_message_report_files(self, message_id: int, report_files: list) -> bool:
        """更新某条消息的 report_files 字段（用于流式响应后异步生成 PDF 报告）"""
        if not self._ensure_pool():
            return False
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE chat_messages SET report_files = %s WHERE id = %s",
                    (Jsonb(report_files) if report_files else None, message_id),
                )
                conn.commit()
                return cur.rowcount > 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("update_message_report_files failed: %s", exc)
            return False

    def get_chat_messages(self, session_id: str) -> List[dict]:
        """获取会话的全部消息(按时间排序)"""
        if not self._ensure_pool():
            return []
        try:
            with self._conn() as conn, conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """SELECT * FROM chat_messages
                       WHERE session_id = %s
                       ORDER BY created_at ASC, id ASC""",
                    (session_id,),
                )
                return cur.fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_chat_messages failed: %s", exc)
            return []

    def load_chat_messages_strict(self, session_id: str) -> List[dict]:
        """History used by the agent must fail visibly if it cannot be loaded."""
        self.require_available()
        with self._conn() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT * FROM chat_messages WHERE session_id = %s
                   ORDER BY created_at ASC, id ASC""",
                (session_id,),
            )
            return cur.fetchall()

    def delete_session_messages(self, session_id: str) -> bool:
        """清空会话消息并重置计数"""
        if not self._ensure_pool():
            return False
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chat_messages WHERE session_id = %s",
                    (session_id,),
                )
                cur.execute(
                    """UPDATE sessions
                       SET message_count = 0, updated_at = now()
                       WHERE id = %s""",
                    (session_id,),
                )
                conn.commit()
                return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("delete_session_messages failed: %s", exc)
            return False

    # ==================================================================
    # 3. Assessment Results (工具输出结构化存储)
    # ==================================================================
    def save_assessment(
        self,
        task: str,
        summary: dict,
        session_id: str = None,
        description: str = "",
        geom_geojson: str = None,
        turn_id: str | None = None,
        artifact_ids: list[str] | None = None,
    ) -> Optional[int]:
        """保存工具评估结果及持久化产物引用

        Args:
            task: MCP 工具名
            summary: 工具输出的 summary dict
            session_id: 关联的会话 ID
            description: 描述
            geom_geojson: GeoJSON 字符串(检测区域外接多边形)
        """
        if not self._ensure_pool():
            return None
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO assessment_results
                       (session_id, task, description, raster_path, geojson_path,
                        overlay_path, summary_path, summary, geom, model_file,
                        num_objects, turn_id, artifact_ids)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NULL, %s, %s,
                               %s, %s)
                       RETURNING id""",
                    (
                        session_id,
                        task,
                        description or summary.get("description", ""),
                        summary.get("raster_path"),
                        summary.get("geojson_path"),
                        summary.get("overlay_path"),
                        summary.get("summary_path"),
                        Jsonb(summary),
                        summary.get("model_file"),
                        summary.get("num_objects"),
                        turn_id,
                        Jsonb(artifact_ids or []),
                    ),
                )
                row = cur.fetchone()
                if row is None:
                    conn.commit()
                    return None
                assessment_id = row[0]
                if geom_geojson:
                    cur.execute(
                        """UPDATE assessment_results
                           SET geom = ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)
                           WHERE id = %s""",
                        (geom_geojson, assessment_id),
                    )
                conn.commit()
                return assessment_id
        except Exception as exc:  # noqa: BLE001
            logger.warning("save_assessment failed: %s", exc)
            return None

    def query_assessments(
        self,
        task: str = None,
        session_id: str = None,
        limit: int = 50,
    ) -> List[dict]:
        """查询评估结果"""
        if not self._ensure_pool():
            return []
        try:
            conditions = []
            params = []
            if task:
                conditions.append("task = %s")
                params.append(task)
            if session_id:
                conditions.append("session_id = %s")
                params.append(session_id)
            where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
            params.append(limit)
            with self._conn() as conn, conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""SELECT id, session_id, task, description, raster_path,
                              geojson_path, overlay_path, summary_path, summary,
                              num_objects, artifact_ids,
                              created_at, ST_AsGeoJSON(geom) as geom_geojson
                        FROM assessment_results{where}
                        ORDER BY created_at DESC LIMIT %s""",
                    params,
                )
                return cur.fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("query_assessments failed: %s", exc)
            raise DatabaseUnavailable("Could not load assessments") from exc

    def latest_assessment_geometry(self, session_id: str) -> Optional[dict]:
        self.require_available()
        with self._conn() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """SELECT id, session_id, task, description, raster_path,
                   geojson_path, overlay_path, summary_path, summary,
                   num_objects, artifact_ids, created_at,
                   ST_AsGeoJSON(geom) AS geom_geojson
                   FROM assessment_results
                   WHERE session_id = %s AND geom IS NOT NULL
                   ORDER BY created_at DESC, id DESC LIMIT 1""",
                (session_id,),
            )
            return cur.fetchone()

    def query_assessments_within(
        self, geom_geojson: str, task: str = None, limit: int = 50
    ) -> List[dict]:
        """空间查询: 查找与给定几何体相交的评估结果"""
        if not self._ensure_pool():
            return []
        try:
            task_clause = "AND task = %s" if task else ""
            params: list = [geom_geojson]
            if task:
                params.append(task)
            params.append(limit)
            with self._conn() as conn, conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""SELECT id, task, summary, raster_path, overlay_path,
                              created_at, ST_AsGeoJSON(geom) as geom_geojson
                        FROM assessment_results
                        WHERE geom IS NOT NULL
                          AND ST_Intersects(geom, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326))
                          {task_clause}
                        ORDER BY created_at DESC LIMIT %s""",
                    params,
                )
                return cur.fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("query_assessments_within failed: %s", exc)
            return []

    # ==================================================================
    # 4. Error Memory
    # ==================================================================
    def save_error(self, pattern: str, fix: str) -> bool:
        """保存/更新一条错误记忆"""
        if not self._ensure_pool():
            return False
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO error_memory (pattern, fix)
                       VALUES (%s, %s)
                       ON CONFLICT (pattern)
                       DO UPDATE SET fix = EXCLUDED.fix, updated_at = now()""",
                    (pattern, fix),
                )
                conn.commit()
                return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("save_error failed: %s", exc)
            return False

    def load_all_errors(self) -> Dict[str, str]:
        """加载全部错误记忆"""
        if not self._ensure_pool():
            return {}
        try:
            with self._conn() as conn, conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT pattern, fix FROM error_memory")
                return {row["pattern"]: row["fix"] for row in cur.fetchall()}
        except Exception as exc:  # noqa: BLE001
            logger.warning("load_all_errors failed: %s", exc)
            return {}

    def increment_error_hit(self, pattern: str) -> bool:
        """错误命中计数 +1"""
        if not self._ensure_pool():
            return False
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """UPDATE error_memory SET hits = hits + 1, updated_at = now()
                       WHERE pattern = %s""",
                    (pattern,),
                )
                conn.commit()
                return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("increment_error_hit failed: %s", exc)
            return False

    def save_error_recovery_candidate(
        self,
        tool_name: str,
        error_signature: str,
        error_hash: str,
        changed_fields: list[str],
    ) -> bool:
        """Store verified retry evidence for review, not as an active rule."""
        if not self._ensure_pool():
            return False
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO error_recovery_candidates
                           (tool_name, error_signature, error_hash, changed_fields)
                           VALUES (%s, %s, %s, %s)
                           ON CONFLICT (tool_name, error_hash, changed_fields)
                           DO UPDATE SET occurrences = error_recovery_candidates.occurrences + 1,
                                         updated_at = now()""",
                    (tool_name, error_signature, error_hash, Jsonb(changed_fields)),
                )
                conn.commit()
                return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("save_error_recovery_candidate failed: %s", exc)
            return False

    # ==================================================================
    # 通用辅助
    # ==================================================================
    def health_check(self) -> bool:
        """检查数据库是否可连接

        与 _ensure_pool 不同，此方法总是重新探测一次，
        因此即使之前标记为不可用(如 Docker 尚未就绪)，
        在 Docker 启动后也能重新连上。
        """
        if not is_enabled():
            return False
        if self._pool is not None:
            try:
                with self._conn() as conn, conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return True
            except Exception:
                pass  # 连接池可能已失效，继续尝试重建
        # 重新探测
        try:
            dsn = _build_dsn()
            probe = psycopg.connect(dsn, connect_timeout=3)
            probe.close()
            # 探测成功，重置状态并重建连接池
            self._checked = False
            self._pool = None
            return self._ensure_pool()
        except Exception:
            return False

    def close(self):
        """关闭连接池"""
        if self._pool is not None:
            try:
                self._pool.close()
            except Exception:
                pass
            self._pool = None


# 模块级单例
db = Database()
atexit.register(db.close)
