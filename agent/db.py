"""
Disaster-Assessment-Agent 数据库连接层

设计原则:
  1. 懒连接 —— 首次使用时才尝试连库
  2. 优雅降级 —— DB 不可用时自动回退到纯文件模式，不阻断业务
  3. 连接池复用 —— 使用 psycopg 连接池，避免频繁建连
  4. 最小侵入 —— 现有代码只需调用 DB.xxx()，不关心连接细节

支持的表: sessions / chat_messages / assessment_results / error_memory
"""

from __future__ import annotations

import atexit
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

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

    所有方法在 DB 不可用时静默返回 None / [] / False，不抛异常。
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
            return False  # 已确认不可用，不重复尝试
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

    # ==================================================================
    # 1. Sessions
    # ==================================================================
    def create_session(
        self,
        model_name: str = "",
        config_path: str = "",
        system_prompt: str = "",
        title: str = "",
    ) -> Optional[str]:
        """创建会话，返回 session_id (UUID)"""
        if not self._ensure_pool():
            return None
        try:
            with self._conn() as conn, conn.cursor(row_factory=dict_row) as cur:
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
            return []

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
    ) -> Optional[int]:
        """保存一条聊天消息，返回消息 ID"""
        if not self._ensure_pool():
            return None
        try:
            with self._conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO chat_messages
                       (session_id, role, content, display_content, attachments,
                        images, tool_trace, elapsed_seconds, tool_call_count)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                       RETURNING id""",
                    (
                        session_id,
                        role,
                        content,
                        display_content or None,
                        Jsonb(attachments) if attachments else None,
                        Jsonb(images) if images else None,
                        Jsonb(tool_trace) if tool_trace else None,
                        elapsed_seconds,
                        tool_call_count,
                    ),
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

    def get_chat_messages(self, session_id: str) -> List[dict]:
        """获取会话的全部消息(按时间排序)"""
        if not self._ensure_pool():
            return []
        try:
            with self._conn() as conn, conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """SELECT * FROM chat_messages
                       WHERE session_id = %s
                       ORDER BY created_at ASC""",
                    (session_id,),
                )
                return cur.fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_chat_messages failed: %s", exc)
            return []

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
    ) -> Optional[int]:
        """保存工具评估结果

        Args:
            task: 任务类型 (building/flood/car/ship/damage/solar_panel/wetland/water)
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
                        num_objects)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                        f"ST_GeomFromGeoJSON('{geom_geojson}')" if geom_geojson else None,
                        summary.get("model_file"),
                        summary.get("num_objects"),
                    ),
                )
                # 如果有空间数据，用参数化方式设置 geom
                if geom_geojson:
                    cur.execute(
                        """UPDATE assessment_results SET geom = ST_GeomFromGeoJSON(%s)
                           WHERE id = (SELECT currval('assessment_results_id_seq'))""",
                        (geom_geojson,),
                    )
                row = cur.fetchone()
                conn.commit()
                return row[0] if row else None
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
                              geojson_path, overlay_path, summary, num_objects,
                              created_at, ST_AsGeoJSON(geom) as geom_geojson
                        FROM assessment_results{where}
                        ORDER BY created_at DESC LIMIT %s""",
                    params,
                )
                return cur.fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("query_assessments failed: %s", exc)
            return []

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
                          AND ST_Intersects(geom, ST_GeomFromGeoJSON(%s))
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
