"""
Earth Agent – Streamlit interaction UI.

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
    return base + roots_block


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
                    lambda m: os.getenv(m.group(1), m.group(0)),
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


def _extract_damage_overlay_paths(response: dict, *texts: str) -> list[str]:
    """Find generated damage overlay images in tool outputs / answer text."""
    found: list[str] = []

    def _add_path(value: str) -> None:
        value = value.strip().strip("`'\" ,.;")
        if not value.endswith("damage_overlay.png"):
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
            for match in re.findall(r"[\w./~:-]*damage_overlay\.png", obj):
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


def _render_chat_message(message: dict[str, Any]) -> None:
    st.markdown(message["content"])
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
        "pending_question": None,       # question loaded from benchmark, awaiting send
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
    page_title="Earth Agent",
    page_icon="🌍",
    layout="wide",
)

st.sidebar.title("⚙️ Configuration")

configs = list_model_configs()
if not configs:
    st.sidebar.error(
        "No model configurations found under `agent/config_*.json`."
    )
    st.stop()

cfg_idx = st.sidebar.selectbox(
    "Model configuration",
    range(len(configs)),
    format_func=lambda i: f"{configs[i]['name']} — {configs[i]['model_name']}",
    help="Models are loaded from `agent/config_*.json`.",
)
selected = configs[cfg_idx]

# Override credentials in the UI
with st.sidebar.expander("API credentials (override)"):
    st.caption(
        f"From config: **{selected['model_name']}** — "
        f"key `{_mask_key(selected['api_key'])}`, "
        f"endpoint `{selected['base_url'] or '(default)'}`"
    )
    api_key = st.text_input("API key", value=selected["api_key"], type="password")
    base_url = st.text_input("Base URL", value=selected["base_url"])
    model_name = st.text_input("Model name", value=selected["model_name"])

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
    st.session_state.last_answer_meta = None
    st.rerun()


# ---------------------------------------------------------------------------
# Streamlit – main layout
# ---------------------------------------------------------------------------
st.title("🌍 Earth Agent — interactive UI")
st.caption(
    "Run the LangChain ReAct agent over the MCP tool stack. Pick a model, "
    "load a benchmark question (or write your own), upload data, and chat."
)

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
session_temp = TEMP_BASE / str(uuid.uuid4().hex[:12])
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
    st.sidebar.success(
        f"Agent ready · {len(handle.tools)} tools loaded", icon="✅"
    )
except Exception as exc:  # noqa: BLE001
    st.sidebar.error(f"Failed to initialise agent: {exc}")
    st.stop()

tabs = st.tabs(["💬 Chat", "📋 Benchmark question", "📂 Upload data", "🛠️ Model info"])


# ---------------------------------------------------------------------------
# Tab 2 – Benchmark question loader
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Load a benchmark question")
    questions = load_questions()
    qids = list(questions.keys())
    qid = st.selectbox(
        "Question ID",
        qids,
        index=0,
        format_func=lambda k: _question_summary(k, questions[k]),
    )
    info = questions[qid]
    user_text = (info.get("dialogs") or [{}])[0].get("content", "")
    eval_block = info.get("evaluation") or [{}]
    data_path = next(
        (e.get("data") for e in eval_block if e.get("data")),
        None,
    )
    choices = info.get("choices") or []

    st.markdown("**Question text**")
    st.text_area("Question text area", value=user_text, height=160, label_visibility="collapsed", key=f"q_text_{qid}")

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Data directory**")
        st.code(data_path or "(not specified)", language="text")
    with cols[1]:
        st.markdown("**Choices**")
        if choices:
            for i, c in enumerate(choices):
                st.write(f"{chr(65 + i)}. {c}")
        else:
            st.write("(open-ended)")

    st.caption(
        "Clicking *Load* adds the question to the chat input. You can also "
        "edit the chat text directly before sending."
    )

    if st.button("➕ Load into chat", key="load_q"):
        text = user_text
        if data_path:
            text += f"\n\nRelevant data directory: `{data_path}`"
        if choices:
            text += "\n\nChoices:\n" + "\n".join(
                f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)
            )
        st.session_state.pending_question = text
        st.toast("Question loaded into the chat tab ✉️", icon="✅")


# ---------------------------------------------------------------------------
# Tab 3 – File uploads
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Upload data files")
    st.write(
        "Upload raster / vector / image files for the agent to work with. "
        "They will be saved into a session-specific directory the agent can "
        "call `get_filelist` on."
    )
    uploads = st.file_uploader(
        "Select one or more files",
        accept_multiple_files=True,
        type=None,
    )
    cols = st.columns([1, 1, 3])
    with cols[0]:
        if st.button("💾 Save uploads", disabled=not uploads):
            target = session_temp / "uploads"
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
            for f in uploads:
                with open(target / f.name, "wb") as out:
                    out.write(f.getbuffer())
            st.session_state.uploads_dir = str(target)
            st.session_state.uploaded_files = sorted(p.name for p in uploads)
            st.success(f"Saved {len(uploads)} file(s) to {target}")
    with cols[1]:
        if st.button("♻️ Reset uploads"):
            st.session_state.uploads_dir = None
            st.session_state.uploaded_files = []
            st.info("Cleared upload directory")

    if st.session_state.uploads_dir:
        st.markdown("**Current upload directory**")
        st.code(st.session_state.uploads_dir, language="text")
        st.markdown("**Files**")
        for name in st.session_state.uploaded_files:
            st.write(f"- {name}")
    else:
        st.info("No files uploaded yet.")


# ---------------------------------------------------------------------------
# Tab 4 – Model info
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Active model")
    if handle is None:
        st.warning("Agent not initialised – see the error in the sidebar.")
    else:
        st.json(
            {
                "config_file": selected["path"],
                "config_name": selected["name"],
                "model_name": model_name,
                "base_url": base_url or "(default OpenAI)",
                "api_key": _mask_key(api_key),
                "mcp_servers": list(selected["mcp_servers"].keys()),
                "tools_loaded": [t.name for t in handle.tools],
                "session_temp_dir": str(session_temp),
                "recursion_limit": recursion_limit,
                "max_execution_time_s": max_exec_seconds,
            }
        )
        st.subheader("Loaded tool list")
        for t in handle.tools:
            with st.expander(f"🔧 {t.name}", expanded=False):
                st.write(t.description or "(no description)")


# ---------------------------------------------------------------------------
# Tab 1 – Chat
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Chat with the agent")

    # A pending benchmark question can be injected via the "load" button.
    pending = st.session_state.pop("pending_question", None)
    if pending:
        # Persist into the chat history so it survives subsequent reruns and
        # is rendered by the loop below.
        st.session_state.messages.append({"role": "user", "content": pending})

    user_text = st.chat_input(
        "Ask Earth Agent… (Shift+Enter for newline)",
        key="chat_input",
    )

    if user_text:
        # Append user turn
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        # Build the message list the agent will see. We enrich the system
        # prompt with the list of directories the agent is allowed to
        # inspect via `get_filelist`, so it does not guess paths like
        # /home/ubuntu that the OS will reject.
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
            status = st.status("Agent is thinking…", expanded=False)
            t0 = time.time()
            try:
                response = handle.invoke(
                    lc_messages,
                    config={
                        "recursion_limit": recursion_limit,
                        "max_execution_time": max_exec_seconds,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                status.update(label="Agent error", state="error")
                st.error("".join(traceback.format_exception(exc)))
                st.stop()
            elapsed = time.time() - t0

            raw_answer = _last_ai_message(response)
            final_answer = extract_final_answer(raw_answer)
            trace = _tool_trace(response)
            images = _extract_damage_overlay_paths(response, raw_answer, final_answer)

            status.update(
                label=f"Done in {elapsed:.1f}s · {len(trace)} tool call(s)",
                state="complete",
            )

            placeholder.markdown(final_answer or "_(empty response)_")
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

        if show_tool_trace:
            with st.expander("Tool-call trace (advanced)", expanded=False):
                for i, step in enumerate(trace, 1):
                    st.markdown(f"**Step {i} — `{step['name']}`**")
                    st.json(step.get("args") or {})
                    st.code(step.get("result") or "", language="text")

    # Render previous messages on rerun. During a fresh submit, the current
    # user/assistant turns have already been rendered above.
    if not user_text:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                _render_chat_message(m)

    if st.session_state.last_answer_meta and not user_text:
        meta = st.session_state.last_answer_meta
        st.caption(
            f"Last response: {meta['elapsed']:.1f}s · "
            f"{meta['tool_calls']} tool call(s)."
        )
