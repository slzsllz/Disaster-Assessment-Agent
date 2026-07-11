"""
FastAPI backend for the Vue Disaster Detection Agent frontend.

This keeps the existing LangGraph/MCP agent runtime in Python and exposes a
small HTTP API for the Vue app:

- POST /api/chat: send a message and optional files
- POST /api/sessions/{session_id}/clear: clear backend chat history
- GET /api/files/{file_id}: read generated output images
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.error_memory import ErrorMemory
from agent.db import db


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
CONDA_PREFIX = os.getenv("CONDA_PREFIX") or str(Path(sys.executable).resolve().parents[1])
CONDA_BIN = os.getenv("CONDA_BIN") or str(Path(CONDA_PREFIX) / "bin")
CONDA_PYTHON = os.getenv("CONDA_PYTHON") or sys.executable
PROJ_DATA_DIR = os.getenv("PROJ_DATA") or str(Path(CONDA_PREFIX) / "share" / "proj")
GDAL_DATA_DIR = os.getenv("GDAL_DATA") or str(Path(CONDA_PREFIX) / "share" / "gdal")

os.environ["CONDA_PREFIX"] = CONDA_PREFIX
if CONDA_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = f"{CONDA_BIN}{os.pathsep}{os.environ.get('PATH', '')}"
os.environ.setdefault("GTIFF_SRS_SOURCE", "EPSG")
os.environ.setdefault("GDAL_DATA", GDAL_DATA_DIR)
os.environ.setdefault("PROJ_DATA", PROJ_DATA_DIR)
os.environ.setdefault("PROJ_LIB", PROJ_DATA_DIR)

MCP_CHILD_ENV_KEYS = (
    "PATH",
    "CONDA_PREFIX",
    "PROJ_DATA",
    "PROJ_LIB",
    "PROJ_DEBUG",
    "GDAL_DATA",
    "GDAL_NUM_THREADS",
    "GTIFF_SRS_SOURCE",
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
)

load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

AGENT_DIR = PROJECT_ROOT / "agent"
BENCHMARK_DATA_DIR = PROJECT_ROOT / "benchmark" / "data"
TEMP_BASE = PROJECT_ROOT / "tmp" / "fastapi_out"
TEMP_BASE.mkdir(parents=True, exist_ok=True)
ERROR_MEMORY = ErrorMemory(AGENT_DIR / "error_memory.json")

DEFAULT_CONFIG = AGENT_DIR / next(
    (
        name
        for name in (
            os.getenv("AGENT_CONFIG"),
            "config_qwen3.json",
            "config_deepseek.json",
            "config.json",
        )
        if name and (AGENT_DIR / name).exists()
    ),
    "config.json",
)
DEFAULT_SYSTEM_PROMPT = (
    "You are a geoscientist, and you need to use tools to answer Earth "
    "observation questions. Carefully reason about which tools to use and "
    "in what order. When a tool returns 'Result saved at /path/to/file', "
    "you MUST use that full path in all subsequent tool calls. Finish your "
    "final response with a clearly labelled answer block, e.g.:\n"
    "<Answer>Your final answer</Answer>"
)

ANSWER_RE = re.compile(r"<Answer>(.*?)</Answer>", re.DOTALL | re.IGNORECASE)
LOCAL_PATH_RE = re.compile(r"`?(/(?:home\d*|tmp)/[^\s`*),;]+(?:\.[A-Za-z0-9]+))`?")
OVERLAY_RE = re.compile(r"[\w./~:-]*_overlay\.png")


def build_system_prompt(base: str, data_roots: list[str]) -> str:
    if not base:
        base = DEFAULT_SYSTEM_PROMPT
    roots_block = (
        "\n\nData access — the following directories are valid inputs to "
        "`get_filelist` and any tool that accepts a file path. Always "
        "start by calling `get_filelist` on one of these roots to find "
        "the data the user is referring to:\n"
        + "\n".join(f"  - {root}" for root in data_roots)
    )
    return base + roots_block + ERROR_MEMORY.format_prompt_block()


def sanitize_local_paths(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return f"`{Path(match.group(1)).name}`"

    return LOCAL_PATH_RE.sub(replace, text or "")


def sanitize_display_answer(text: str) -> str:
    """Hide machine-local absolute directories while preserving answer text."""
    cleaned = text or ""
    cleaned = sanitize_local_paths(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    match = ANSWER_RE.search(text)
    if not match:
        return text.strip()
    before = text[: match.start()].strip()
    answer = match.group(1).strip()
    after = text[match.end() :].strip()
    if len(before) + len(after) > 80:
        return "\n\n".join(part for part in (before, answer, after) if part)
    return answer


def build_messages(history: list[dict[str, str]], system_prompt: str | None) -> list:
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    for item in history:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            messages.append(AIMessage(content=item["content"]))
    return messages


def last_ai_message(response: dict) -> str:
    for msg in reversed(response.get("messages", [])):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
    return "(no assistant message)"


def message_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content or "")


def tool_trace(response: dict) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for msg in response.get("messages", []):
        if isinstance(msg, AIMessage):
            extra = getattr(msg, "additional_kwargs", {}) or {}
            for call in extra.get("tool_calls") or []:
                fn = call.get("function", {}) or {}
                args_raw = fn.get("arguments", "")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except Exception:
                    args = args_raw
                pending = {"name": fn.get("name", "?"), "args": args, "result": None}
        elif isinstance(msg, ToolMessage):
            entry = pending or {"name": getattr(msg, "name", "?"), "args": None}
            entry["result"] = str(msg.content)[:500]
            trace.append(entry)
            pending = None
    return trace


def extract_overlay_images(response: dict, *texts: str) -> list[str]:
    found: list[str] = []

    def add_path(value: str) -> None:
        value = value.strip().strip("`'\" ,.;")
        if not value.endswith("_overlay.png"):
            return
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists() and str(path) not in found:
            found.append(str(path))

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "overlay_path" and isinstance(value, str):
                    add_path(value)
                else:
                    visit(value)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)
        elif isinstance(obj, str):
            for match in OVERLAY_RE.findall(obj):
                add_path(match)

    for msg in response.get("messages", []):
        if isinstance(msg, ToolMessage):
            content = msg.content
            if isinstance(content, str):
                try:
                    visit(json.loads(content))
                except Exception:
                    visit(content)
            else:
                visit(content)
    for text in texts:
        visit(text)
    return found


def substitute_env(value: str) -> str:
    return re.sub(r"\$\{([^}]+)\}", lambda match: os.getenv(match.group(1), ""), value)


def load_model_config(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg = json.loads(config_path.read_text())
    if not cfg.get("models"):
        raise RuntimeError(f"No models configured in {config_path}")
    model = cfg["models"][0]
    api_key = substitute_env(model.get("api_key", "") or "")
    base_url = substitute_env((model.get("client_args") or {}).get("base_url", "") or "")
    return {
        "path": str(config_path),
        "model_name": model.get("model_name", "qwen3.7-plus"),
        "api_key": api_key,
        "base_url": base_url,
        "generate_args": model.get("generate_args", {}) or {},
        "mcp_servers": cfg.get("mcpServers", {}),
    }


def build_mcp_child_env(session_id: str = "") -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key in MCP_CHILD_ENV_KEYS}
    env["CONDA_PREFIX"] = CONDA_PREFIX
    env["PATH"] = f"{CONDA_BIN}:{env.get('PATH', os.environ.get('PATH', ''))}"
    # 让工具子进程把评估结果关联到当前会话 (见 agent/tools/utils.py)
    if session_id:
        env["DISASTER_SESSION_ID"] = session_id
    return env


def build_mcp_servers(
    mcp_servers_cfg: dict[str, Any], temp_dir: Path, session_id: str = ""
) -> dict[str, Any]:
    servers: dict[str, Any] = {}
    child_env = build_mcp_child_env(session_id)
    for name, server_cfg in mcp_servers_cfg.items():
        args: list[str] = []
        for arg in server_cfg.get("args", []):
            if "tmp/tmp/out" in arg:
                args.append(str(temp_dir / "out"))
            elif arg.startswith("tools/"):
                args.append(str(AGENT_DIR / arg))
            else:
                args.append(arg)
        servers[name] = {
            "command": CONDA_PYTHON,
            "args": args,
            "env": child_env,
            "transport": "stdio",
        }
    return servers


class AgentHandle:
    """LangGraph/MCP agent pinned to one background asyncio loop."""

    def __init__(self, config: dict[str, Any], temp_dir: Path, session_id: str = ""):
        self.loop = asyncio.new_event_loop()
        self.ready: queue.Queue[bool] = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self.ready.get(timeout=20)

        async def setup() -> tuple[Any, Any, list[Any]]:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            from langchain_openai import ChatOpenAI
            from langgraph.prebuilt import create_react_agent

            llm = ChatOpenAI(
                model=config["model_name"],
                api_key=config["api_key"] or "EMPTY",
                base_url=config["base_url"] or None,
                temperature=0.1,
                request_timeout=180,
                extra_body=config["generate_args"] or None,
            )
            client = MultiServerMCPClient(
                build_mcp_servers(config["mcp_servers"], temp_dir, session_id)
            )
            tools = await client.get_tools()
            agent = create_react_agent(llm, tools)
            return agent, client, tools

        self.agent, self.client, self.tools = self.run(setup())

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.ready.put(True)
        self.loop.run_forever()

    def run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def invoke(self, messages: list, config: dict | None = None) -> dict:
        async def run_agent():
            return await self.agent.ainvoke({"messages": messages}, config=config or {})

        return self.run(run_agent())

    def stream(self, messages: list, config: dict | None = None):
        output_queue: queue.Queue[Any] = queue.Queue()
        sentinel = object()

        async def run_agent():
            final_response: dict[str, Any] | None = None
            try:
                async for event in self.agent.astream_events(
                    {"messages": messages},
                    config=config or {},
                    version="v2",
                ):
                    event_name = event.get("event")
                    data = event.get("data") or {}
                    if event_name == "on_chat_model_stream":
                        chunk = data.get("chunk")
                        text = message_content_text(getattr(chunk, "content", ""))
                        if text:
                            output_queue.put({"type": "delta", "text": text})
                    elif event_name == "on_tool_start":
                        output_queue.put(
                            {
                                "type": "status",
                                "message": f"Calling tool: {event.get('name', 'tool')}",
                            }
                        )
                    elif event_name == "on_chain_end":
                        output = data.get("output")
                        if isinstance(output, dict) and "messages" in output:
                            final_response = output

                if final_response is None:
                    final_response = await self.agent.ainvoke(
                        {"messages": messages},
                        config=config or {},
                    )
                output_queue.put({"type": "final", "response": final_response})
            except Exception as exc:  # noqa: BLE001
                output_queue.put({"type": "error", "error": exc})
            finally:
                output_queue.put(sentinel)

        asyncio.run_coroutine_threadsafe(run_agent(), self.loop)
        while True:
            item = output_queue.get()
            if item is sentinel:
                break
            yield item

    def close(self) -> None:
        async def close_client():
            if hasattr(self.client, "aclose"):
                await self.client.aclose()
            elif hasattr(self.client, "close"):
                await self.client.close()

        try:
            self.run(close_client())
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)


@dataclass
class ChatSession:
    session_id: str
    temp_dir: Path
    messages: list[dict[str, str]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    handle: AgentHandle | None = None

    @property
    def uploads_dir(self) -> Path:
        return self.temp_dir / "uploads"

    @property
    def output_dir(self) -> Path:
        return self.temp_dir / "out"


app = FastAPI(title="Disaster Detection Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_CONFIG = load_model_config()
SESSIONS: dict[str, ChatSession] = {}
FILES: dict[str, Path] = {}
SESSIONS_LOCK = threading.Lock()


def get_session(session_id: str) -> ChatSession:
    session_id = session_id.strip() or uuid.uuid4().hex
    with SESSIONS_LOCK:
        session = SESSIONS.get(session_id)
        if session is None:
            temp_dir = TEMP_BASE / session_id
            temp_dir.mkdir(parents=True, exist_ok=True)
            (temp_dir / "out").mkdir(parents=True, exist_ok=True)
            session = ChatSession(session_id=session_id, temp_dir=temp_dir)
            SESSIONS[session_id] = session
        return session


def file_payload(path: str) -> dict[str, str]:
    file_id = uuid.uuid4().hex
    file_path = Path(path)
    FILES[file_id] = file_path
    return {"name": file_path.name, "url": f"/api/files/{file_id}"}


def sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# DB row serializers -- convert dict_row results to JSON-friendly dicts.
# FastAPI handles datetime/UUID encoding; we only reshape images/assessments.
# ---------------------------------------------------------------------------
def _image_descriptor(path: str) -> dict[str, str]:
    """Image descriptor stored in chat_messages.images JSONB.

    Keeps the absolute path so a history row loaded after a backend restart can
    re-register the file (the /api/files/{id} mapping lives in-memory only).
    """
    return {"name": Path(path).name, "path": str(path)}


def _serialize_message(row: dict) -> dict[str, Any]:
    images: list[dict[str, str]] = []
    for img in row.get("images") or []:
        if not isinstance(img, dict):
            continue
        path = img.get("path")
        if path and Path(path).exists():
            images.append(file_payload(path))  # fresh /api/files/{id} url
        elif img.get("url"):
            images.append({"name": img.get("name", "image"), "url": img["url"]})
    return {
        "id": row["id"],
        "role": row["role"],
        "content": row.get("display_content") or row.get("content") or "",
        "attachments": row.get("attachments") or [],
        "images": images,
        "tool_trace": row.get("tool_trace") or [],
        "elapsed_seconds": row.get("elapsed_seconds"),
        "tool_call_count": row.get("tool_call_count"),
        "created_at": row.get("created_at"),
    }


def _serialize_session(row: dict) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "title": row.get("title") or "",
        "model_name": row.get("model_name") or "",
        "message_count": row.get("message_count") or 0,
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "first_message": row.get("first_message") or "",
    }


def _serialize_assessment(row: dict) -> dict[str, Any]:
    overlay_path = row.get("overlay_path")
    return {
        "id": row["id"],
        "session_id": str(row["session_id"]) if row.get("session_id") else None,
        "task": row.get("task"),
        "description": row.get("description"),
        "raster_path": row.get("raster_path"),
        "geojson_path": row.get("geojson_path"),
        "overlay_path": overlay_path,
        "summary_path": row.get("summary_path"),
        "summary": row.get("summary") or {},
        "num_objects": row.get("num_objects"),
        "geom": row.get("geom_geojson"),
        "created_at": row.get("created_at"),
        "overlay_url": file_payload(overlay_path)["url"] if overlay_path and Path(overlay_path).exists() else None,
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": MODEL_CONFIG["model_name"],
        "tools": list(MODEL_CONFIG["mcp_servers"].keys()),
        "db_ok": db.health_check(),
    }


@app.get("/api/sessions")
def list_sessions(limit: int = 30) -> dict[str, Any]:
    """列出最近的会话 -- 前端历史侧栏的数据来源 (DB)"""
    rows = db.list_recent_sessions(limit=limit)
    return {"sessions": [_serialize_session(r) for r in rows]}


@app.get("/api/sessions/{session_id}/messages")
def get_session_messages(session_id: str) -> dict[str, Any]:
    """加载某会话的全部消息 -- 切换历史会话时从 DB 读取"""
    rows = db.get_chat_messages(session_id)
    return {"session_id": session_id, "messages": [_serialize_message(r) for r in rows]}


@app.get("/api/sessions/{session_id}/assessments")
def get_session_assessments(session_id: str) -> dict[str, Any]:
    """某会话产生的评估结果 (模型输出)"""
    rows = db.query_assessments(session_id=session_id)
    return {"session_id": session_id, "assessments": [_serialize_assessment(r) for r in rows]}


@app.get("/api/assessments")
def list_assessments(task: str = "", limit: int = 50) -> dict[str, Any]:
    """全局评估结果列表 (可按 task 过滤)"""
    rows = db.query_assessments(task=task or None, limit=limit)
    return {"assessments": [_serialize_assessment(r) for r in rows]}


@app.post("/api/sessions/{session_id}/clear")
def clear_session(session_id: str) -> dict[str, bool]:
    session = get_session(session_id)
    with session.lock:
        session.messages.clear()
    db.delete_session_messages(session_id)
    return {"ok": True}


@app.get("/api/files/{file_id}")
def get_file(file_id: str) -> FileResponse:
    path = FILES.get(file_id)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@app.post("/api/chat")
def chat(
    session_id: str = Form(...),
    message: str = Form(""),
    system_prompt: str = Form(DEFAULT_SYSTEM_PROMPT),
    recursion_limit: int = Form(40),
    max_execution_time: int = Form(600),
    show_trace: bool = Form(False),
    files: list[UploadFile] | None = File(default=None),
) -> dict[str, Any]:
    session = get_session(session_id)
    with session.lock:
        try:
            session.uploads_dir.mkdir(parents=True, exist_ok=True)
            uploaded_paths: list[str] = []
            for upload in files or []:
                if not upload.filename:
                    continue
                safe_name = Path(upload.filename).name
                save_path = session.uploads_dir / safe_name
                with open(save_path, "wb") as out:
                    out.write(upload.file.read())
                uploaded_paths.append(str(save_path))

            content_parts = [message.strip()] if message.strip() else []
            if uploaded_paths:
                content_parts.append(
                    "Uploaded files:\n" + "\n".join(f"- `{path}`" for path in uploaded_paths)
                )
            user_content = "\n\n".join(content_parts) or "Please analyze the uploaded file(s)."
            session.messages.append({"role": "user", "content": user_content})

            # Persist session + user message to DB (upsert; idempotent)
            db.create_session(
                session_id=session_id,
                model_name=MODEL_CONFIG["model_name"],
                config_path=MODEL_CONFIG["path"],
                system_prompt=system_prompt,
                title=(message.strip()[:80] or None),
            )
            db.save_chat_message(
                session_id,
                "user",
                content=user_content,
                attachments=[{"name": Path(p).name, "path": p} for p in uploaded_paths],
            )

            if session.handle is None:
                session.handle = AgentHandle(MODEL_CONFIG, session.temp_dir, session_id=session_id)

            data_roots = [str(BENCHMARK_DATA_DIR), str(session.uploads_dir), str(session.output_dir)]
            effective_prompt = build_system_prompt(system_prompt, data_roots)
            lc_messages = build_messages(session.messages, effective_prompt)

            started = time.time()
            response = session.handle.invoke(
                lc_messages,
                config={
                    "recursion_limit": recursion_limit,
                    "max_execution_time": max_execution_time,
                },
            )
            elapsed = time.time() - started
            raw_answer = last_ai_message(response)
            final_answer = extract_final_answer(raw_answer)
            display_answer = sanitize_display_answer(final_answer or raw_answer)
            trace = tool_trace(response)
            images = extract_overlay_images(response, raw_answer, final_answer)

            session.messages.append(
                {"role": "assistant", "content": final_answer or raw_answer}
            )
            db.save_chat_message(
                session_id,
                "assistant",
                content=final_answer or raw_answer,
                display_content=display_answer,
                images=[_image_descriptor(p) for p in images],
                tool_trace=trace,
                elapsed_seconds=elapsed,
                tool_call_count=len(trace),
            )

            return {
                "answer": display_answer or "(empty response)",
                "elapsed": elapsed,
                "tool_calls": len(trace),
                "images": [file_payload(path) for path in images],
                "trace": trace if show_trace else [],
            }
        except Exception as exc:  # noqa: BLE001
            error_text = "".join(traceback.format_exception(exc))
            matches = ERROR_MEMORY.lookup_all(error_text)
            return {
                "answer": f"后端调用失败：{exc}",
                "elapsed": 0,
                "tool_calls": 0,
                "images": [],
                "trace": [],
                "error": error_text,
                "memory_suggestions": [
                    {"pattern": pattern, "fix": fix} for pattern, fix in matches
                ],
            }


@app.post("/api/chat/stream")
def chat_stream(
    session_id: str = Form(...),
    message: str = Form(""),
    system_prompt: str = Form(DEFAULT_SYSTEM_PROMPT),
    recursion_limit: int = Form(40),
    max_execution_time: int = Form(600),
    show_trace: bool = Form(False),
    files: list[UploadFile] | None = File(default=None),
) -> StreamingResponse:
    session = get_session(session_id)
    session.lock.acquire()
    try:
        session.uploads_dir.mkdir(parents=True, exist_ok=True)
        uploaded_paths: list[str] = []
        for upload in files or []:
            if not upload.filename:
                continue
            safe_name = Path(upload.filename).name
            save_path = session.uploads_dir / safe_name
            with open(save_path, "wb") as out:
                out.write(upload.file.read())
            uploaded_paths.append(str(save_path))

        content_parts = [message.strip()] if message.strip() else []
        if uploaded_paths:
            content_parts.append(
                "Uploaded files:\n" + "\n".join(f"- `{path}`" for path in uploaded_paths)
            )
        user_content = "\n\n".join(content_parts) or "Please analyze the uploaded file(s)."
        session.messages.append({"role": "user", "content": user_content})

        # Persist session + user message to DB (upsert; idempotent)
        db.create_session(
            session_id=session_id,
            model_name=MODEL_CONFIG["model_name"],
            config_path=MODEL_CONFIG["path"],
            system_prompt=system_prompt,
            title=(message.strip()[:80] or None),
        )
        db.save_chat_message(
            session_id,
            "user",
            content=user_content,
            attachments=[{"name": Path(p).name, "path": p} for p in uploaded_paths],
        )

        if session.handle is None:
            session.handle = AgentHandle(MODEL_CONFIG, session.temp_dir, session_id=session_id)

        data_roots = [str(BENCHMARK_DATA_DIR), str(session.uploads_dir), str(session.output_dir)]
        effective_prompt = build_system_prompt(system_prompt, data_roots)
        lc_messages = build_messages(session.messages, effective_prompt)
    except Exception:
        session.lock.release()
        raise

    def generate():
        started = time.time()
        streamed_text = ""
        try:
            yield sse_event("status", {"message": "正在思考..."})
            for item in session.handle.stream(
                lc_messages,
                config={
                    "recursion_limit": recursion_limit,
                    "max_execution_time": max_execution_time,
                },
            ):
                if item["type"] == "delta":
                    text = item["text"]
                    streamed_text += text
                    yield sse_event("delta", {"text": text})
                elif item["type"] == "status":
                    yield sse_event("status", {"message": item["message"]})
                elif item["type"] == "error":
                    raise item["error"]
                elif item["type"] == "final":
                    elapsed = time.time() - started
                    response = item["response"]
                    raw_answer = last_ai_message(response)
                    final_answer = extract_final_answer(raw_answer)
                    display_answer = sanitize_display_answer(
                        final_answer or raw_answer or streamed_text
                    )
                    trace = tool_trace(response)
                    images = extract_overlay_images(response, raw_answer, final_answer, streamed_text)

                    session.messages.append(
                        {"role": "assistant", "content": final_answer or raw_answer or streamed_text}
                    )
                    db.save_chat_message(
                        session_id,
                        "assistant",
                        content=final_answer or raw_answer or streamed_text,
                        display_content=display_answer,
                        images=[_image_descriptor(p) for p in images],
                        tool_trace=trace,
                        elapsed_seconds=elapsed,
                        tool_call_count=len(trace),
                    )

                    yield sse_event(
                        "done",
                        {
                            "answer": display_answer or "(empty response)",
                            "elapsed": elapsed,
                            "tool_calls": len(trace),
                            "images": [file_payload(path) for path in images],
                            "trace": trace if show_trace else [],
                        },
                    )
        except Exception as exc:  # noqa: BLE001
            error_text = "".join(traceback.format_exception(exc))
            matches = ERROR_MEMORY.lookup_all(error_text)
            yield sse_event(
                "error",
                {
                    "answer": f"后端调用失败：{exc}",
                    "elapsed": 0,
                    "tool_calls": 0,
                    "images": [],
                    "trace": [],
                    "error": error_text,
                    "memory_suggestions": [
                        {"pattern": pattern, "fix": fix} for pattern, fix in matches
                    ],
                },
            )
        finally:
            session.lock.release()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
