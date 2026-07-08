"""
Disaster Detection Agent – Streamlit interaction UI.

Features
--------
1. Pick a model from `agent/config_*.json` (API key / base URL pulled from `.env`).
2. Load a benchmark question (`benchmark/question.json`) into the chat.
3. Upload data files; they are made available to the agent via a session
   directory it can call `get_filelist` on.
4. Run the LangChain ReAct agent (over the existing MCP tools) and show
   only the final analysis / answer in the UI – intermediate tool calls
   are hidden behind a toggle.
"""

from __future__ import annotations

import asyncio
import base64
import html
import json
import os
import re
import shutil
import sys
import time
import traceback
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Environment – match what the existing langchain_*.py scripts do, otherwise
# the rasterio / GDAL stack raises warnings about missing SRS data.
#
# Two paths exist for the PROJ database in this env:
#   1. $CONDA_PREFIX/share/proj                              (system, PROJ 9.7)
#   2. $CONDA_PREFIX/lib/python3.10/site-packages/pyproj/
#      proj_dir/share/proj                                   (pyproj 3.7 bundle)
#
# pyproj 3.7.1 ships a proj.db whose DATABASE.LAYOUT.VERSION.MINOR = 4,
# but the conda env's libproj (9.7) expects >= 6 – using the pyproj path
# triggers "Open of …proj.db contains DATABASE.LAYOUT.VERSION.MINOR = 4
# whereas a number >= 6 is expected" warnings under `from osgeo import gdal`.
# We point PROJ_DATA/PROJ_LIB at the system path so GDAL, rasterio, and
# the MCP tool subprocesses all agree on one up-to-date database.
# ---------------------------------------------------------------------------
_CONDA_PREFIX = "/home2/llz/miniconda3/envs/earthagent_cpython"
_PROJ_DATA_DIR = f"{_CONDA_PREFIX}/share/proj"
_GDAL_DATA_DIR = f"{_CONDA_PREFIX}/share/gdal"

os.environ.setdefault("GTIFF_SRS_SOURCE", "EPSG")
os.environ.setdefault("GDAL_DATA", _GDAL_DATA_DIR)
os.environ.setdefault("PROJ_DATA", _PROJ_DATA_DIR)
os.environ.setdefault("PROJ_LIB", _PROJ_DATA_DIR)

# Subset of the parent env that each MCP tool subprocess needs. The
# `langchain_mcp_adapters` client passes a (mostly) empty env to stdio
# subprocesses (just PATH), so any env var we don't include here is gone
# in the child process. PROJ needs PROJ_DATA / PROJ_LIB; GDAL needs
# GDAL_DATA and GTIFF_SRS_SOURCE; keep CONDA_PREFIX too because several
# conda-installed tools rely on it.
_MCP_CHILD_ENV_KEYS = (
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

from dotenv import load_dotenv  # noqa: E402

load_dotenv()  # project root .env

import streamlit as st  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from agent.error_memory import ErrorMemory  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)  # MCP tools resolve relative paths from cwd

AGENT_DIR = PROJECT_ROOT / "agent"
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
QUESTION_FILE = BENCHMARK_DIR / "question.json"
BENCHMARK_DATA_DIR = BENCHMARK_DIR / "data"
TEMP_BASE = PROJECT_ROOT / "tmp" / "streamlit_out"
TEMP_BASE.mkdir(parents=True, exist_ok=True)
ERROR_MEMORY = ErrorMemory(AGENT_DIR / "error_memory.json")
WATERMARK_LOGO = PROJECT_ROOT / "imgs" / "深圳大学-logo-1024px.png"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are a geoscientist, and you need to use tools to answer Earth "
    "observation questions. Carefully reason about which tools to use and "
    "in what order. When a tool returns 'Result saved at /path/to/file', "
    "you MUST use that full path in all subsequent tool calls. Finish your "
    "final response with a clearly labelled answer block, e.g.:\n"
    "<Answer>Your final answer</Answer>"
)


def build_system_prompt(base: str, data_roots: list[str]) -> str:
    """Append a 'valid data roots' block to a base system prompt.

    This tells the agent exactly which directory prefixes are valid
    inputs to ``get_filelist`` and other filesystem tools, so it does
    not have to guess paths like ``/home/ubuntu`` that the OS will
    reject.
    """
    if not base:
        base = (
            "You are a geoscientist, and you need to use tools to answer "
            "Earth observation questions. Carefully reason about which "
            "tools to use and in what order. When a tool returns "
            "'Result saved at /path/to/file', you MUST use that full path "
            "in all subsequent tool calls. Finish your final response with "
            "a clearly labelled answer block, e.g.:\n"
            "<Answer>Your final answer</Answer>"
        )
    roots_block = (
        "\n\nData access — the following directories are valid inputs to "
        "`get_filelist` and any tool that accepts a file path. Always "
        "start by calling `get_filelist` on one of these roots to find "
        "the data the user is referring to:\n"
        + "\n".join(f"  - {r}" for r in data_roots)
    )
    return base + roots_block + ERROR_MEMORY.format_prompt_block()


# ---------------------------------------------------------------------------
# Helpers – configuration loading
# ---------------------------------------------------------------------------
def _mask_key(value: str) -> str:
    if not value:
        return "(empty)"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}…{value[-4:]}"


@st.cache_data
def list_model_configs() -> list[dict[str, Any]]:
    """Discover every `agent/config_*.json` model configuration."""
    configs: list[dict[str, Any]] = []
    for p in sorted(AGENT_DIR.glob("config_*.json")):
        try:
            cfg = json.loads(p.read_text())
        except Exception:
            continue
        for model in cfg.get("models", []):
            api_key = model.get("api_key", "") or ""
            client_args = model.get("client_args", {}) or {}
            base_url = client_args.get("base_url", "") or ""

            # Substitute ${ENV_VAR} placeholders that the existing
            # `langchain_*.py` scripts rely on, so the UI does not have
            # to be edited when a new environment variable is added.
            def _sub(s: str) -> str:
                return re.sub(
                    r"\$\{([^}]+)\}",
                    lambda m: os.getenv(m.group(1), ""),
                    s,
                )

            api_key = _sub(api_key)
            base_url = _sub(base_url)

            # If still empty, look for an env variable based on the
            # `config_name`/`model_name` (mirrors config_utils.py).
            if not api_key or not base_url:
                _try_fill_credentials(model.get("config_name", ""), model)

            configs.append(
                {
                    "path": str(p),
                    "name": model.get("config_name", p.stem),
                    "model_name": model.get("model_name", "?"),
                    "model_type": model.get("model_type", "?"),
                    "api_key": api_key,
                    "base_url": base_url,
                    "generate_args": model.get("generate_args", {}) or {},
                    "mcp_servers": cfg.get("mcpServers", {}),
                }
            )
    return configs


def _try_fill_credentials(name: str, model: dict[str, Any]) -> None:
    """Best-effort fill api_key/base_url from env, like config_utils.py."""
    aliases = {
        # "openai": "OPENAI",
        # "gpt": "OPENAI",
        # "gpt-5": "OPENAI",
        # "gpt-4": "OPENAI",
        "deepseek": "DEEPSEEK",
        "qwen": "QWEN",
        "qwen3": "QWEN",
        "qwen3-32b": "QWEN",
        # "kimi": "KIMI",
        # "kimi_k2": "KIMI",
        # "gemini": "GEMINI",
        # "glm": "GLM",
        # "glm-4.5": "GLM",
    }
    prefix = aliases.get(name.lower(), name.upper())
    if not model.get("api_key"):
        model["api_key"] = os.getenv(f"{prefix}_API_KEY", "") or ""
    if not (model.get("client_args") or {}).get("base_url"):
        model.setdefault("client_args", {})["base_url"] = (
            os.getenv(f"{prefix}_BASE_URL", "") or ""
        )


def _find_enhanced_system_prompt(model_name: str) -> str | None:
    """Look for a learned `system_prompt` that matches this model."""
    slug = model_name.lower().replace("-", "_")
    candidates = list(PROJECT_ROOT.glob("training_free_results_*/earth_agent_practice_*_enhanced_config.json"))
    for p in candidates:
        if slug in p.name.lower() or any(
            tok in p.name.lower() for tok in model_name.lower().split("-")
        ):
            try:
                cfg = json.loads(p.read_text())
                prompt = cfg.get("system_prompt")
                if prompt:
                    return prompt
            except Exception:
                continue
    return None


@st.cache_data
def load_questions() -> dict[str, Any]:
    return json.loads(QUESTION_FILE.read_text())


def _question_summary(qid: str, info: dict[str, Any]) -> str:
    user_text = (info.get("dialogs") or [{}])[0].get("content", "")
    short = user_text.splitlines()[0] if user_text else "(no text)"
    return f"Q{qid} — {short[:70]}"


# ---------------------------------------------------------------------------
# Helpers – MCP server setup
# ---------------------------------------------------------------------------
def _build_mcp_child_env() -> dict[str, str]:
    """Project only the env vars the MCP tool subprocesses need.

    `langchain_mcp_adapters` (and the underlying `mcp` stdio client) does
    NOT inherit the parent process env by default – it only injects
    `PATH` if absent. Without our env vars, each tool subprocess starts
    with an empty env, PROJ falls back to a compile-time default path
    that is not readable from this env, and we get the dreaded
    "PROJ: proj_create_from_database: Open of <…>/share/proj failed"
    warnings every time a tool touches a CRS.
    """
    env = {k: v for k, v in os.environ.items() if k in _MCP_CHILD_ENV_KEYS}
    # Belt-and-braces – the stdio client re-asserts PATH if missing.
    env.setdefault("PATH", os.environ.get("PATH", ""))
    return env


def _build_mcp_servers(
    mcp_servers_cfg: dict[str, Any], temp_dir: Path
) -> dict[str, Any]:
    """Mirror the path-rewriting logic from langchain_*.py."""
    servers: dict[str, Any] = {}
    child_env = _build_mcp_child_env()
    for name, scfg in mcp_servers_cfg.items():
        args: list[str] = []
        for a in scfg.get("args", []):
            if "tmp/tmp/out" in a:
                args.append(str(temp_dir / "out"))
            elif a.startswith("tools/"):
                # The tool scripts live under agent/tools/, not cwd/tools/
                args.append(str(AGENT_DIR / a))
            else:
                args.append(a)
        servers[name] = {
            "command": scfg.get("command", "python"),
            "args": args,
            "env": child_env,
            "transport": "stdio",
        }
    return servers


# ---------------------------------------------------------------------------
# Helpers – agent lifecycle (event-loop aware)
# ---------------------------------------------------------------------------
class _AgentHandle:
    """A ReAct agent bound to a single, persistent event loop.

    `MultiServerMCPClient` spawns stdio subprocesses that are tied to the
    loop that created them, so we must keep using the same loop for every
    invocation. `st.cache_resource` keeps this object alive across reruns.
    """

    def __init__(self, llm, mcp_servers: dict[str, Any]):
        self.loop = asyncio.new_event_loop()

        async def _setup() -> tuple[Any, Any]:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            from langgraph.prebuilt import create_react_agent

            client = MultiServerMCPClient(mcp_servers)
            tools = await client.get_tools()
            agent = create_react_agent(llm, tools)
            return agent, client, tools

        try:
            asyncio.set_event_loop(self.loop)
            self.agent, self.client, self.tools = self.loop.run_until_complete(
                _setup()
            )
        except Exception:
            self.loop.close()
            raise

    def invoke(self, messages: list, config: dict | None = None) -> dict:
        from langchain_core.messages import HumanMessage, SystemMessage

        # `agent.ainvoke` already accepts {"messages": [...]}; we wrap
        # the user-provided history and prepend the system prompt if any.
        async def _run():
            return await self.agent.ainvoke(
                {"messages": messages}, config=config or {}
            )

        return self.loop.run_until_complete(_run())

    def close(self) -> None:
        try:
            if hasattr(self.client, "aclose"):

                async def _close():
                    await self.client.aclose()

                self.loop.run_until_complete(_close())
            elif hasattr(self.client, "close"):

                async def _close():
                    await self.client.close()

                self.loop.run_until_complete(_close())
        finally:
            self.loop.close()


@st.cache_resource(show_spinner="Loading model and MCP tools…")
def _build_agent(
    cfg_path: str,
    api_key: str,
    base_url: str,
    model_name: str,
    generate_args_json: str,
    temp_dir: str,
) -> _AgentHandle:
    """Build / cache the agent. Keyed on (cfg, credentials, model, temp_dir)."""
    from langchain_openai import ChatOpenAI

    cfg = json.loads(Path(cfg_path).read_text())
    servers_cfg = cfg.get("mcpServers", {})

    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)
    (temp_path / "out").mkdir(parents=True, exist_ok=True)

    generate_args = json.loads(generate_args_json) if generate_args_json else {}

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key or "EMPTY",
        base_url=base_url or None,
        temperature=0.1,
        request_timeout=180,
        extra_body=generate_args or None,
    )

    servers = _build_mcp_servers(servers_cfg, temp_path)
    return _AgentHandle(llm, servers)


# ---------------------------------------------------------------------------
# Helpers – agent execution + answer extraction
# ---------------------------------------------------------------------------
ANSWER_RE = re.compile(r"<Answer>(.*?)</Answer>", re.DOTALL | re.IGNORECASE)
LOCAL_PATH_RE = re.compile(
    r"`?(/(?:home\d*|tmp)/[^\s`*),;]+(?:\.[A-Za-z0-9]+))`?"
)
IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    m = ANSWER_RE.search(text)
    if m:
        before = text[: m.start()].strip()
        answer = m.group(1).strip()
        after = text[m.end() :].strip()
        if len(before) + len(after) > 80:
            return "\n\n".join(part for part in (before, answer, after) if part)
        return answer
    return text.strip()


def _build_messages(
    history: list[dict[str, str]], system_prompt: str | None
) -> list:
    msgs = []
    if system_prompt:
        msgs.append(SystemMessage(content=system_prompt))
    for m in history:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            content = m["content"]
            if isinstance(content, str):
                msgs.append(AIMessage(content=content))
    return msgs


def _last_ai_message(response: dict) -> str:
    for msg in reversed(response.get("messages", [])):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Some langchain versions return a list of content blocks
                return "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content
                )
    return "(no assistant message)"


def _tool_trace(response: dict) -> list[dict[str, Any]]:
    """Summarise the tool-call steps (only shown when the user opts in)."""
    trace: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for msg in response.get("messages", []):
        if isinstance(msg, AIMessage):
            extra = getattr(msg, "additional_kwargs", {}) or {}
            calls = extra.get("tool_calls") or []
            for call in calls:
                fn = call.get("function", {}) or {}
                args_raw = fn.get("arguments", "")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except Exception:
                    args = args_raw
                pending = {"name": fn.get("name", "?"), "args": args, "result": None}
        elif isinstance(msg, ToolMessage):
            entry = pending or {"name": getattr(msg, "name", "?"), "args": None, "result": None}
            entry["result"] = str(msg.content)[:500]
            trace.append(entry)
            pending = None
    return trace


def _extract_overlay_images(response: dict, *texts: str) -> list[str]:
    """Find generated overlay/preview images in tool outputs / answer text.

    Matches any ``*_overlay.png`` file (covers damage_overlay.png,
    flood_overlay.png, and GeoAI tool outputs like car_overlay.png,
    water_unet_overlay.png, similarity_overlay.png, features_overlay.png).
    """
    found: list[str] = []

    def _add_path(value: str) -> None:
        value = value.strip().strip("`'\" ,.;")
        if not value.endswith("_overlay.png"):
            return
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists() and str(path) not in found:
            found.append(str(path))

    def _visit(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "overlay_path" and isinstance(value, str):
                    _add_path(value)
                else:
                    _visit(value)
        elif isinstance(obj, list):
            for item in obj:
                _visit(item)
        elif isinstance(obj, str):
            for match in re.findall(r"[\w./~:-]*_overlay\.png", obj):
                _add_path(match)

    for msg in response.get("messages", []):
        if isinstance(msg, ToolMessage):
            content = msg.content
            if isinstance(content, str):
                try:
                    _visit(json.loads(content))
                except Exception:
                    _visit(content)
            else:
                _visit(content)

    for text in texts:
        if text:
            _visit(text)

    return found


def _is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def _sanitize_local_paths_for_display(text: str) -> str:
    """Hide machine-local paths from the chat transcript."""
    if not text:
        return ""

    def _replace(match: re.Match[str]) -> str:
        path = match.group(1)
        return f"`{Path(path).name}`"

    return LOCAL_PATH_RE.sub(_replace, text)


def _render_attachments(attachments: list[dict[str, str]]) -> None:
    if not attachments:
        return

    image_items = [a for a in attachments if a.get("type") == "image"]
    file_items = [a for a in attachments if a.get("type") != "image"]

    if image_items:
        visible = image_items[:4]
        cols = st.columns([1] * len(visible) + [max(1, 4 - len(visible))], gap="small")
        for idx, item in enumerate(visible):
            path = item.get("path", "")
            name = item.get("name") or Path(path).name or "image"
            safe_name = html.escape(name)
            with cols[idx]:
                if path and Path(path).exists():
                    try:
                        st.image(path, caption=name, width=128)
                        continue
                    except Exception:
                        pass
                st.markdown(
                    f"<div class='attachment-card'>Image · {safe_name}</div>",
                    unsafe_allow_html=True,
                )

    for item in file_items:
        name = item.get("name") or Path(item.get("path", "")).name or "file"
        suffix = Path(name).suffix.upper().lstrip(".") or "FILE"
        safe_name = html.escape(name)
        safe_suffix = html.escape(suffix)
        st.markdown(
            f"""
            <div class="attachment-card">
              <span class="attachment-icon">📄</span>
              <span class="attachment-name">{safe_name}</span>
              <span class="attachment-type">{safe_suffix}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_chat_message(message: dict[str, Any]) -> None:
    display_content = message.get("display_content", message.get("content", ""))
    display_content = _sanitize_local_paths_for_display(display_content)
    if display_content:
        st.markdown(display_content)
    _render_attachments(message.get("attachments", []) or [])
    for image_path in message.get("images", []) or []:
        if Path(image_path).exists():
            st.image(image_path, caption=Path(image_path).name, width=420)


# ---------------------------------------------------------------------------
# Streamlit – session state bootstrap
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "messages": [],                 # chat history [{role, content}, ...]
        "model_signature": None,        # identifies the active (model, cfg, temp_dir)
        "uploads_dir": None,            # path of saved uploaded files
        "uploaded_files": [],           # filenames
        "session_temp_dir": None,       # stable temp dir for this Streamlit session
        "pending_question": None,       # question loaded from benchmark, awaiting send
        "pending_assistant": False,     # run the agent after the next rerun
        "last_answer_meta": None,       # info about the last assistant turn
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# Streamlit – sidebar (model + advanced settings)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Disaster Detection Agent",
    page_icon="🌍",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1040px;
        padding-top: 0.75rem;
        padding-bottom: 13.5rem;
        position: relative;
        box-sizing: border-box;
        min-height: 0;
    }
    [data-testid="stSidebar"] {
        background: #f7f7f8;
        border-right: 1px solid #e5e5e5;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #202123;
    }
    div[data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.85rem 0;
        border-bottom: 1px solid #f0f0f0;
    }
    div[data-testid="stChatMessage"]:last-of-type {
        border-bottom: 0;
    }
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        line-height: 1.65;
    }
    div[data-testid="stChatInput"] {
        position: fixed !important;
        left: calc(240px + (100vw - 240px) / 2) !important;
        bottom: 1.4rem !important;
        width: min(980px, calc(100vw - 240px - 4rem)) !important;
        transform: translateX(-50%);
        z-index: 50 !important;
        margin: 0 !important;
    }
    div[data-testid="stElementContainer"]:has(div[data-testid="stChatInput"]),
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stChatInput"]) {
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: visible !important;
        background: transparent !important;
    }
    div[data-testid="stBottom"],
    div[data-testid="stBottomBlockContainer"],
    div[data-testid="stBottom"] > div,
    div[data-testid="stBottomBlockContainer"] > div,
    section[data-testid="stMain"] div:has(> div[data-testid="stChatInput"]) {
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }
    div[data-testid="stBottom"]::before,
    div[data-testid="stBottom"]::after,
    div[data-testid="stBottomBlockContainer"]::before,
    div[data-testid="stBottomBlockContainer"]::after,
    section[data-testid="stMain"] div:has(> div[data-testid="stChatInput"])::before,
    section[data-testid="stMain"] div:has(> div[data-testid="stChatInput"])::after {
        background: transparent !important;
        background-image: none !important;
        box-shadow: none !important;
    }
    div[data-testid="stChatInput"] > div {
        border-radius: 26px;
        background: #ffffff;
        border: 1px solid #d6d6dd;
        box-shadow: 0 18px 48px rgba(0, 0, 0, 0.10);
        padding: 0.25rem 0.5rem;
    }
    div[data-testid="stChatInput"] > div:focus-within {
        border-color: #d6d6dd !important;
        box-shadow: 0 18px 48px rgba(0, 0, 0, 0.10) !important;
        outline: none !important;
    }
    div[data-testid="stChatInput"] textarea {
        border: 0;
        box-shadow: none;
        background: #ffffff !important;
        background-color: #ffffff !important;
        min-height: 68px;
        font-size: 0.98rem;
        padding: 1rem 0.6rem;
    }
    div[data-testid="stChatInput"] textarea:focus,
    div[data-testid="stChatInput"] textarea:focus-visible,
    div[data-testid="stChatInput"] [contenteditable="true"],
    div[data-testid="stChatInput"] [contenteditable="true"]:focus {
        background: #ffffff !important;
        background-color: #ffffff !important;
        border-color: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }
    div[data-testid="stChatInput"] button {
        color: #4f4f5f;
        background: #ffffff;
        border-color: transparent;
        min-width: 38px;
        min-height: 38px;
        border-radius: 999px;
        opacity: 1;
        visibility: visible;
    }
    div[data-testid="stChatInput"] button:hover,
    div[data-testid="stChatInput"] button:focus,
    div[data-testid="stChatInput"] button:active {
        color: #202123;
        background: #f4f4f5;
        border-color: #ececf1;
        box-shadow: none;
    }
    div[data-testid="stChatInput"] svg {
        color: #4f4f5f;
        opacity: 1;
        visibility: visible;
    }
    div[data-testid="stChatInput"] [data-testid="stBaseButton-secondary"],
    div[data-testid="stChatInput"] [data-testid="stBaseButton-primary"] {
        background: #ffffff;
        border-color: transparent;
        color: #4f4f5f;
        opacity: 1;
        visibility: visible;
    }
    div[data-testid="stChatInput"] [data-testid="stBaseButton-secondary"]:hover,
    div[data-testid="stChatInput"] [data-testid="stBaseButton-primary"]:hover {
        background: #f4f4f5;
        border-color: #ececf1;
        color: #202123;
    }
    div[data-testid="stChatInput"] [data-testid="stFileUploaderDropzone"],
    div[data-testid="stChatInput"] [data-testid="stFileUploader"] {
        background: #ffffff;
        border-color: #d6d6dd;
        color: #202123;
    }
    div[data-testid="stChatInput"] [data-testid="stFileUploaderFile"] {
        background: #f7f7f8;
        border-color: #e5e5e5;
        color: #202123;
    }
    div[data-testid="stChatInput"] [role="progressbar"] > div,
    div[data-testid="stChatInput"] progress::-webkit-progress-value {
        background-color: #6e6e80;
    }
    .earth-agent-empty {
        text-align: center;
        color: #202123;
        min-height: calc(100vh - 15rem);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 0 0 4rem;
    }
    .earth-agent-empty h2 {
        font-size: 1.8rem;
        font-weight: 650;
        margin-bottom: 0.4rem;
    }
    .earth-agent-empty p {
        color: #6e6e80;
        margin: 0;
    }
    .upload-note {
        color: #6e6e80;
        font-size: 0.88rem;
        margin-top: 0.35rem;
    }
    .attachment-card {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        max-width: 360px;
        margin: 0.35rem 0.35rem 0.15rem 0;
        padding: 0.62rem 0.75rem;
        border: 1px solid #e5e5e5;
        border-radius: 10px;
        background: #ffffff;
        color: #202123;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        vertical-align: top;
    }
    .attachment-icon {
        flex: 0 0 auto;
        font-size: 1.05rem;
        line-height: 1;
    }
    .attachment-name {
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-size: 0.92rem;
    }
    .attachment-type {
        flex: 0 0 auto;
        color: #6e6e80;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .chat-bottom-spacer {
        height: 10rem;
    }
    @media (max-width: 768px) {
        div[data-testid="stChatInput"] {
            left: 50% !important;
            width: calc(100vw - 1.5rem) !important;
            bottom: 0.75rem !important;
        }
    }
    div[data-testid="stAlert"] {
        background: #f7f7f8;
        color: #202123;
        border: 1px solid #e5e5e5;
        border-radius: 10px;
    }
    div[data-testid="stStatusWidget"] {
        border: 1px solid #e5e5e5;
        background: #ffffff;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if WATERMARK_LOGO.exists():
    watermark_data = base64.b64encode(WATERMARK_LOGO.read_bytes()).decode("ascii")
    st.markdown(
        f"""
        <style>
        .stApp::before {{
            content: "";
            position: fixed;
            left: calc(240px + (100vw - 240px) / 2);
            top: 50%;
            width: min(52vw, 620px);
            aspect-ratio: 1;
            transform: translate(-50%, -50%);
            pointer-events: none;
            background-image: url("data:image/png;base64,{watermark_data}");
            background-repeat: no-repeat;
            background-position: center;
            background-size: contain;
            opacity: 0.055;
            z-index: 0;
        }}
        section[data-testid="stMain"],
        section[data-testid="stMain"] > div {{
            position: relative;
            z-index: 1;
        }}
        [data-testid="stSidebar"] {{
            z-index: 2;
        }}
        @media (max-width: 768px) {{
            .stApp::before {{
                left: 50%;
                width: min(78vw, 480px);
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.title("Disaster Detection Agent")

configs = list_model_configs()
if not configs:
    st.sidebar.error(
        "No model configurations found under `agent/config_*.json`."
    )
    st.stop()

default_cfg_idx = next(
    (
        i
        for i, cfg in enumerate(configs)
        if "qwen" in f"{cfg['name']} {cfg['model_name']}".lower()
    ),
    0,
)
cfg_idx = default_cfg_idx
selected = configs[cfg_idx]
api_key = selected["api_key"]
base_url = selected["base_url"]
model_name = selected["model_name"]

# System prompt
enhanced_prompt = _find_enhanced_system_prompt(selected["name"]) or _find_enhanced_system_prompt(
    selected["model_name"]
)
prompt_options = ["Default (concise)"]
if enhanced_prompt:
    prompt_options.append("Enhanced (learned experiences)")
prompt_options.append("Custom")

prompt_choice = st.sidebar.radio("System prompt", prompt_options, index=0)

if prompt_choice == "Default (concise)":
    default_prompt = DEFAULT_SYSTEM_PROMPT
elif prompt_choice == "Enhanced (learned experiences)":
    default_prompt = enhanced_prompt or DEFAULT_SYSTEM_PROMPT
else:
    default_prompt = ""

system_prompt = st.sidebar.text_area(
    "System prompt",
    value=default_prompt,
    height=240,
    help=(
        "You can edit this freely. The agent will receive it as the first "
        "message of every turn."
    ),
)

# Advanced settings
st.sidebar.subheader("Advanced")
recursion_limit = st.sidebar.slider("Recursion limit", 10, 100, 40, step=5)
max_exec_seconds = st.sidebar.slider("Max execution time (s)", 60, 1800, 600, step=60)
show_tool_trace = st.sidebar.checkbox(
    "Show tool-call trace", value=False, help="Off by default – the UI shows only the final answer."
)
st.session_state["show_tool_trace"] = show_tool_trace

st.sidebar.divider()
if st.sidebar.button("🗑️ Clear chat history"):
    st.session_state.messages = []
    st.session_state.pending_assistant = False
    st.session_state.last_answer_meta = None
    st.rerun()


# Build (or rebuild) the agent when the signature changes.
signature = (
    selected["path"],
    api_key,
    base_url,
    model_name,
    json.dumps(selected.get("generate_args", {}), sort_keys=True),
)
if st.session_state.model_signature != signature:
    # Wipe cached agent if model changed
    _build_agent.clear()
    st.session_state.model_signature = signature

# Unique temp dir for this session so file outputs don't collide
if not st.session_state.session_temp_dir:
    st.session_state.session_temp_dir = str(TEMP_BASE / str(uuid.uuid4().hex[:12]))
session_temp = Path(st.session_state.session_temp_dir)
session_temp.mkdir(parents=True, exist_ok=True)
(session_temp / "out").mkdir(parents=True, exist_ok=True)

handle = None  # populated below; default keeps the rest of the script safe
try:
    handle = _build_agent(
        cfg_path=selected["path"],
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        generate_args_json=json.dumps(selected.get("generate_args", {})),
        temp_dir=str(session_temp),
    )
    st.sidebar.caption(f"Ready · {len(handle.tools)} tools loaded")
except Exception as exc:  # noqa: BLE001
    st.sidebar.error(f"Failed to initialise agent: {exc}")
    st.stop()


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
pending = st.session_state.pop("pending_question", None)
if pending:
    st.session_state.messages.append({"role": "user", "content": pending})
    st.session_state.pending_assistant = True

if st.session_state.messages:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            _render_chat_message(m)
else:
    st.markdown(
        """
        <div class="earth-agent-empty">
          <h2>What can I help analyze?</h2>
          <p>Ask about disaster damage, flood inundation, indices, statistics, or attach files below.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

ran_assistant = bool(st.session_state.pending_assistant)
if ran_assistant:
    data_roots: list[str] = [str(BENCHMARK_DATA_DIR)]
    if st.session_state.uploads_dir:
        data_roots.append(st.session_state.uploads_dir)
    data_roots.append(str(session_temp / "out"))
    effective_system_prompt = build_system_prompt(system_prompt, data_roots)
    lc_messages = _build_messages(
        st.session_state.messages, effective_system_prompt
    )

    with st.chat_message("assistant"):
        placeholder = st.empty()
        t0 = time.time()
        try:
            with st.spinner("Thinking..."):
                response = handle.invoke(
                    lc_messages,
                    config={
                        "recursion_limit": recursion_limit,
                        "max_execution_time": max_exec_seconds,
                    },
                )
        except Exception as exc:  # noqa: BLE001
            error_text = "".join(traceback.format_exception(exc))
            st.error(error_text)
            memory_matches = ERROR_MEMORY.lookup_all(error_text)
            if memory_matches:
                st.warning("Matched error-memory suggestion(s):")
                for pattern, fix in memory_matches:
                    st.markdown(f"- `{pattern}`: {fix}")
            st.stop()
        elapsed = time.time() - t0

        raw_answer = _last_ai_message(response)
        final_answer = extract_final_answer(raw_answer)
        trace = _tool_trace(response)
        images = _extract_overlay_images(response, raw_answer, final_answer)
        answer_for_display = _sanitize_local_paths_for_display(
            final_answer or raw_answer
        )

        placeholder.markdown(answer_for_display or "_(empty response)_")
        st.caption(f"{elapsed:.1f}s · {len(trace)} tool call(s)")
        for image_path in images:
            st.image(image_path, caption=Path(image_path).name, width=420)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": final_answer or raw_answer,
                "images": images,
            }
        )
        st.session_state.last_answer_meta = {
            "elapsed": elapsed,
            "tool_calls": len(trace),
            "raw_answer": raw_answer,
            "images": images,
        }
        st.session_state.pending_assistant = False

    if show_tool_trace:
        with st.expander("Tool-call trace (advanced)", expanded=False):
            for i, step in enumerate(trace, 1):
                st.markdown(f"**Step {i} — `{step['name']}`**")
                st.json(step.get("args") or {})
                st.code(step.get("result") or "", language="text")

if st.session_state.last_answer_meta and not ran_assistant:
    meta = st.session_state.last_answer_meta
    st.caption(
        f"Last response: {meta['elapsed']:.1f}s · "
        f"{meta['tool_calls']} tool call(s)."
    )

if st.session_state.messages:
    st.markdown('<div class="chat-bottom-spacer"></div>', unsafe_allow_html=True)

chat_value = st.chat_input(
    "Ask anything, or attach raster/image files",
    key="chat_input",
    accept_file="multiple",
)

if chat_value:
    if isinstance(chat_value, str):
        user_text = chat_value
        uploaded_files = []
    else:
        user_text = getattr(chat_value, "text", "") or ""
        uploaded_files = list(getattr(chat_value, "files", []) or [])

    uploaded_paths: list[str] = []
    attachments: list[dict[str, str]] = []
    if uploaded_files:
        target = session_temp / "uploads"
        target.mkdir(parents=True, exist_ok=True)
        for f in uploaded_files:
            save_path = target / f.name
            with open(save_path, "wb") as out:
                out.write(f.getbuffer())
            uploaded_paths.append(str(save_path))
            attachments.append(
                {
                    "name": f.name,
                    "path": str(save_path),
                    "type": "image" if _is_image_file(str(save_path)) else "file",
                }
            )
        st.session_state.uploads_dir = str(target)
        st.session_state.uploaded_files = sorted(
            {*(st.session_state.uploaded_files or []), *(f.name for f in uploaded_files)}
        )

    content_parts = [user_text.strip()] if user_text.strip() else []
    if uploaded_paths:
        content_parts.append(
            "Uploaded files:\n" + "\n".join(f"- `{path}`" for path in uploaded_paths)
        )
    user_content = "\n\n".join(content_parts) or "Please analyze the uploaded file(s)."
    display_content = user_text.strip() or "Please analyze the uploaded file(s)."

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_content,
            "display_content": display_content,
            "attachments": attachments,
        }
    )
    st.session_state.pending_assistant = True
    st.rerun()
