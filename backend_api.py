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
import base64
import json
import mimetypes
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
    "DB_ENABLED",
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
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
    "you MUST use that full path in all subsequent tool calls. Do not list "
    "generated output file paths in the final answer; the frontend will "
    "display images and downloads separately. Finish your final response "
    "with a clearly labelled answer block, e.g.:\n"
    "<Answer>Your final answer</Answer>"
)
ARTIFACT_SELECTION_PROMPT = (
    "\n\nArtifact selection — after using a disaster tool, choose only the generated "
    "artifacts that are useful for the user. If you can judge from the tool output, "
    "include an optional machine-readable block after the answer:\n"
    "<Artifacts>{\"display\":[\"useful image filename or path\"],"
    "\"download\":[\"useful file filename or path\"]}</Artifacts>\n"
    "Do not select every generated intermediate file by default. Prefer one clear "
    "overlay/diagnostic image for display and only downloadable GIS/report files "
    "that help the user verify or reuse the result. In the final answer, explain "
    "each selected downloadable file in user-facing terms: what it contains, what "
    "the user can do with it, and which software/workflow it is useful for. Do not "
    "only repeat the filename."
)

ANSWER_RE = re.compile(r"<Answer>(.*?)</Answer>", re.DOTALL | re.IGNORECASE)
ARTIFACTS_RE = re.compile(r"<Artifacts>(.*?)</Artifacts>", re.DOTALL | re.IGNORECASE)
LOCAL_PATH_RE = re.compile(r"`?(/(?:home\d*|tmp)/[^\s`*),;]+(?:\.[A-Za-z0-9]+))`?")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
MULTIMODAL_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
TEXT_PREVIEW_EXTENSIONS = {".json", ".txt", ".csv", ".geojson", ".md"}
EXCLUDED_ARTIFACT_EXTENSIONS = {".pt", ".pth", ".th", ".ckpt", ".safetensors", ".py", ".bak"}
MAX_MULTIMODAL_IMAGE_BYTES = int(os.getenv("MAX_MULTIMODAL_IMAGE_BYTES", str(6 * 1024 * 1024)))
MAX_TEXT_PREVIEW_CHARS = int(os.getenv("MAX_TEXT_PREVIEW_CHARS", "2500"))
MASK_OUTPUT_MARKERS = ("_mask.", "mask_path")
DOWNLOAD_ONLY_IMAGE_MARKERS = (
    "_comparison.",
    "true_color_rgb.",
    "mndwi_heatmap.",
    "water_ndci_heatmap.",
    "ndci_histogram.",
)
OUTPUT_PATH_KEYS = {
    "outputs",
    "output_paths",
    "mask_path",
    "vis_path",
    "visualization_path",
    "overlay_path",
    "comparison_path",
    "summary_path",
    "metrics_csv_path",
    "geojson_path",
    "raster_path",
    "output_mask_path",
    "flood_mask_path",
    "flood_mask_png_path",
    "building_mask_path",
    "damage_mask_path",
    "burned_mask_path",
}
OUTPUT_FILES_SECTION_RE = re.compile(
    r"(?ims)^\s{0,3}#{1,6}\s*"
    r"(?:[^\n#]*?(?:输出文件|生成文件|可视化输出|output files|generated files|visualization files)[^\n]*)"
    r"\n.*?(?=^\s{0,3}#{1,6}\s+|\Z)"
)


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
    return base + ARTIFACT_SELECTION_PROMPT + roots_block + ERROR_MEMORY.format_prompt_block()


def sanitize_local_paths(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return f"`{Path(match.group(1)).name}`"

    return LOCAL_PATH_RE.sub(replace, text or "")


def sanitize_display_answer(text: str) -> str:
    """Hide machine-local absolute directories while preserving answer text."""
    cleaned = text or ""
    cleaned = ARTIFACTS_RE.sub("", cleaned)
    cleaned = sanitize_local_paths(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    text = ARTIFACTS_RE.sub("", text).strip()
    match = ANSWER_RE.search(text)
    if not match:
        return text.strip()
    before = text[: match.start()].strip()
    answer = match.group(1).strip()
    after = text[match.end() :].strip()
    return "\n\n".join(part for part in (before, answer, after) if part)


def file_kind(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in {".tif", ".tiff"}:
        return "geotiff"
    if suffix in TEXT_PREVIEW_EXTENSIONS:
        return suffix.lstrip(".")
    return suffix.lstrip(".") or "file"


def text_preview(path: str) -> str:
    file_path = Path(path)
    if file_path.suffix.lower() not in TEXT_PREVIEW_EXTENSIONS:
        return ""
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    text = text.strip()
    if len(text) > MAX_TEXT_PREVIEW_CHARS:
        return text[:MAX_TEXT_PREVIEW_CHARS] + "\n... [truncated]"
    return text


def image_data_url(path: str) -> str | None:
    file_path = Path(path)
    if file_path.suffix.lower() not in MULTIMODAL_IMAGE_EXTENSIONS:
        return None
    try:
        if file_path.stat().st_size > MAX_MULTIMODAL_IMAGE_BYTES:
            return None
        mime_type = mimetypes.guess_type(file_path.name)[0] or "image/png"
        encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
    except Exception:
        return None


def describe_files_block(title: str, paths: list[str]) -> str:
    if not paths:
        return f"{title}: none"
    lines = [f"{title}:"]
    for index, path in enumerate(paths, 1):
        file_path = Path(path)
        size_kb = 0.0
        try:
            size_kb = file_path.stat().st_size / 1024
        except Exception:
            pass
        lines.append(
            f"{index}. name={file_path.name}; kind={file_kind(path)}; "
            f"size_kb={size_kb:.1f}; path={path}"
        )
        preview = text_preview(path)
        if preview:
            lines.append(f"   text_preview:\n```text\n{preview}\n```")
    return "\n".join(lines)


def build_multimodal_review_content(
    raw_answer: str,
    uploaded_paths: list[str],
    output_paths: list[str],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Review the agent's answer, user input files, and generated tool artifacts. "
                "Use the attached images when available. Decide which generated artifacts are actually useful "
                "for the frontend to display or offer as downloads.\n\n"
                "Return exactly two blocks:\n"
                "<Answer>你的最终中文回答。所有给用户看的解释都必须写在这个块里。不要列出本机路径；可以说明关键图像/文件会在回答底部提供。</Answer>\n"
                "<Artifacts>{\"display\":[\"exact generated filename or path\"],"
                "\"download\":[\"exact generated filename or path\"]}</Artifacts>\n\n"
                "Rules:\n"
                "- display: only generated images that are useful for visual inspection.\n"
                "- download: only generated files that are useful to the user for verification, GIS, or reports.\n"
                "- Do not include model weights, checkpoints, source files, or unhelpful intermediate artifacts.\n"
                "- If two artifacts contain the same information, keep the clearer one.\n\n"
                "In <Answer>, include a short section explaining the selected downloadable files. "
                "For each file, describe what it contains and what the user can do with it "
                "(for example GIS loading, quantitative checking, report archiving, or downstream analysis). "
                "Do not merely repeat filenames.\n\n"
                "Put <Artifacts> last. Do not write any user-facing explanation after </Artifacts>.\n\n"
                f"Original agent answer:\n{raw_answer}\n\n"
                f"{describe_files_block('User uploaded files', uploaded_paths)}\n\n"
                f"{describe_files_block('Generated tool artifacts', output_paths)}"
            ),
        }
    ]

    for path in uploaded_paths:
        data_url = image_data_url(path)
        if data_url:
            content.append({"type": "text", "text": f"User uploaded image: {Path(path).name}"})
            content.append({"type": "image_url", "image_url": {"url": data_url, "detail": "low"}})

    for path in output_paths:
        data_url = image_data_url(path)
        if data_url:
            content.append({"type": "text", "text": f"Generated artifact image: {Path(path).name}"})
            content.append({"type": "image_url", "image_url": {"url": data_url, "detail": "low"}})

    return content


def build_user_message_content(user_text: str, uploaded_paths: list[str]) -> str | list[dict[str, Any]]:
    if not uploaded_paths:
        return user_text
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"{user_text}\n\n"
                "The following uploaded files are available to tools and to you. "
                "For image files, inspect the attached image content directly when deciding file roles "
                "such as pre-disaster vs post-disaster:\n"
                f"{describe_files_block('Uploaded files', uploaded_paths)}"
            ),
        }
    ]
    for path in uploaded_paths:
        data_url = image_data_url(path)
        if data_url:
            content.append({"type": "text", "text": f"Uploaded image: {Path(path).name}"})
            content.append({"type": "image_url", "image_url": {"url": data_url, "detail": "low"}})
    return content


def _artifact_refs(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        refs: list[str] = []
        for item in value:
            if isinstance(item, str):
                refs.append(item)
            elif isinstance(item, dict):
                ref = item.get("path") or item.get("name") or item.get("file")
                if ref:
                    refs.append(str(ref))
        return refs
    return []


def _match_artifact_paths(refs: list[str], output_paths: list[str]) -> list[str]:
    matched: list[str] = []
    by_name = {Path(path).name: path for path in output_paths}
    by_lower_name = {Path(path).name.lower(): path for path in output_paths}
    normalized = {_normalize_existing_file(path) or path: path for path in output_paths}
    for ref in refs:
        clean = ref.strip().strip("`'\" ")
        path = _normalize_existing_file(clean)
        candidate = normalized.get(path or clean) or by_name.get(clean) or by_lower_name.get(clean.lower())
        if candidate and candidate not in matched:
            matched.append(candidate)
    return matched


def extract_artifact_selection(text: str, output_paths: list[str]) -> tuple[list[str], list[str]] | None:
    match = ARTIFACTS_RE.search(text or "")
    if not match:
        return None
    try:
        data = json.loads(match.group(1).strip())
    except Exception:
        return None
    display_refs = _artifact_refs(data.get("display") or data.get("images") or data.get("show"))
    download_refs = _artifact_refs(data.get("download") or data.get("files") or data.get("downloads"))
    display_candidates = _match_artifact_paths(display_refs, output_paths)
    download_candidates = _match_artifact_paths(download_refs, output_paths)

    display: list[str] = []
    download: list[str] = []
    for path in display_candidates:
        if Path(path).suffix.lower() in IMAGE_EXTENSIONS:
            display.append(path)
        elif path not in download:
            download.append(path)
    for path in download_candidates:
        if path not in display and path not in download:
            download.append(path)
    return display, download


def choose_frontend_artifacts(answer_text: str, output_paths: list[str]) -> tuple[list[str], list[str]]:
    selection = extract_artifact_selection(answer_text, output_paths)
    if selection is not None:
        return selection
    return split_output_files(output_paths)


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


def _normalize_existing_file(value: str) -> str | None:
    value = value.strip().strip("`'\" ,.;")
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        path = path.resolve()
    except Exception:
        return None
    return str(path) if path.exists() and path.is_file() else None


def extract_tool_output_files(response: dict) -> list[str]:
    """Collect every real file path returned by tools.

    The frontend decides how to render each file: browser-friendly image
    formats are previewed, other outputs are exposed as downloads.
    """
    found: list[str] = []

    def add_path(value: str) -> None:
        path = _normalize_existing_file(value)
        if path:
            file_path = Path(path)
            if file_path.suffix.lower() in EXCLUDED_ARTIFACT_EXTENSIONS:
                return
            try:
                file_path.relative_to(PROJECT_ROOT / "model")
                return
            except ValueError:
                pass
        if path and path not in found:
            found.append(path)

    def visit(obj: Any, collect_strings: bool = False) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_name = str(key)
                should_collect = collect_strings or key_name in OUTPUT_PATH_KEYS or key_name.endswith("_path")
                visit(value, should_collect)
        elif isinstance(obj, list):
            for item in obj:
                visit(item, collect_strings)
        elif isinstance(obj, str):
            if collect_strings:
                add_path(obj)
                for match in LOCAL_PATH_RE.findall(obj):
                    add_path(match)
            try:
                visit(json.loads(obj), collect_strings)
            except Exception:
                pass

    for msg in response.get("messages", []):
        if isinstance(msg, ToolMessage):
            visit(msg.content)
    return found


def extract_tool_legend(response: dict) -> list[dict[str, Any]]:
    """Collect color legend entries returned by tools."""
    legend: list[dict[str, Any]] = []

    def add_entries(value: Any) -> None:
        items = value.values() if isinstance(value, dict) and "label" not in value else value
        if isinstance(items, dict):
            items = [items]
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            label = item.get("label") or item.get("meaning") or item.get("name")
            color = item.get("color")
            if label and color:
                entry = {
                    "label": str(label),
                    "color": str(color),
                    "value": item.get("value"),
                }
                if entry not in legend:
                    legend.append(entry)

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "legend":
                    add_entries(value)
                else:
                    visit(value)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)
        elif isinstance(obj, str):
            try:
                visit(json.loads(obj))
            except Exception:
                pass

    for msg in response.get("messages", []):
        if isinstance(msg, ToolMessage):
            visit(msg.content)
    return legend


def split_output_files(paths: list[str]) -> tuple[list[str], list[str]]:
    images: list[str] = []
    files: list[str] = []
    for path in paths:
        name = Path(path).name.lower()
        suffix = Path(path).suffix.lower()
        if (
            suffix in IMAGE_EXTENSIONS
            and not any(marker in name for marker in MASK_OUTPUT_MARKERS)
            and not any(marker in name for marker in DOWNLOAD_ONLY_IMAGE_MARKERS)
        ):
            images.append(path)
        else:
            files.append(path)
    return images, files


def extract_bbox_geojson_from_raster(path: str) -> str | None:
    """Extract a WGS84 bbox GeoJSON from a georeferenced raster."""
    try:
        from osgeo import gdal, osr

        ds = gdal.Open(path)
        if ds is None:
            return None
        geo = ds.GetGeoTransform()
        proj = ds.GetProjection()
        width, height = ds.RasterXSize, ds.RasterYSize
        ds = None
        if geo == (0, 1.0, 0, 0, 0, 1.0):
            return None

        def pixel_to_map(px: float, py: float) -> tuple[float, float]:
            return (
                geo[0] + px * geo[1] + py * geo[2],
                geo[3] + px * geo[4] + py * geo[5],
            )

        coords = [
            pixel_to_map(0, 0),
            pixel_to_map(width, 0),
            pixel_to_map(width, height),
            pixel_to_map(0, height),
            pixel_to_map(0, 0),
        ]

        def looks_like_lonlat(points: list[tuple[float, float]]) -> bool:
            return all(-180 <= x <= 180 and -90 <= y <= 90 for x, y in points)

        if proj and not looks_like_lonlat(coords):
            source = osr.SpatialReference()
            if source.ImportFromWkt(proj) == 0:
                target = osr.SpatialReference()
                target.ImportFromEPSG(4326)
                if hasattr(source, "SetAxisMappingStrategy"):
                    source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                transform = osr.CoordinateTransformation(source, target)
                transformed = []
                for x, y in coords:
                    lon, lat, *_ = transform.TransformPoint(x, y)
                    transformed.append((lon, lat))
                coords = transformed

        return json.dumps({
            "type": "Polygon",
            "coordinates": [[list(point) for point in coords]],
        })
    except Exception:
        return None


def extract_first_output_geometry(paths: list[str]) -> str | None:
    for path in paths:
        if Path(path).suffix.lower() in {".tif", ".tiff"}:
            geom = extract_bbox_geojson_from_raster(path)
            if geom:
                return geom
    return None


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
            return agent, llm, client, tools

        self.agent, self.llm, self.client, self.tools = self.run(setup())

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

    def review_answer_and_artifacts(
        self,
        raw_answer: str,
        uploaded_paths: list[str],
        output_paths: list[str],
    ) -> str:
        if not output_paths and not uploaded_paths:
            return raw_answer

        async def run_review():
            messages = [
                SystemMessage(
                    content=(
                        "You are a careful geospatial result reviewer. Use multimodal image understanding "
                        "when images are attached. Improve the final Chinese answer and select only useful "
                        "generated artifacts for display/download. Never expose local absolute paths to users."
                    )
                ),
                HumanMessage(
                    content=build_multimodal_review_content(raw_answer, uploaded_paths, output_paths)
                ),
            ]
            result = await self.llm.ainvoke(messages)
            return message_content_text(getattr(result, "content", ""))

        try:
            reviewed = self.run(run_review())
        except Exception:
            return raw_answer
        return reviewed or raw_answer

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
# AI session title summarization -- prevent long user input from producing
# overly long titles by asking the LLM for a concise summary.
# ---------------------------------------------------------------------------
TITLE_SUMMARIZE_THRESHOLD = 20


def generate_session_title(message: str) -> str | None:
    """Use the LLM to summarize a long user message into a concise title.

    Returns ``None`` on failure so callers can keep the existing fallback.
    """
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=MODEL_CONFIG["model_name"],
            api_key=MODEL_CONFIG["api_key"] or "EMPTY",
            base_url=MODEL_CONFIG["base_url"] or None,
            temperature=0.1,
            request_timeout=30,
            extra_body=MODEL_CONFIG["generate_args"] or None,
        )
        result = llm.invoke([
            SystemMessage(content=(
                "你是一个标题生成助手。请将用户输入的内容总结为一个简短的中文标题，"
                "不超过15个字，直接输出标题文字，不要包含引号、书名号或句号等标点符号。"
            )),
            HumanMessage(content=message[:2000]),
        ])
        title = message_content_text(getattr(result, "content", "")).strip()
        title = title.strip("\"'“”‘’「」【】·-— ")
        return title[:40] if title else None
    except Exception:
        return None


def maybe_generate_session_title(session_id: str, message: str) -> None:
    """Best-effort: summarize a long user message into a session title via AI.

    Only replaces the title when the current one is our own truncated version
    (set by ``db.create_session``). User-renamed titles are preserved.
    """
    text = (message or "").strip()
    if len(text) <= TITLE_SUMMARIZE_THRESHOLD:
        return
    try:
        existing = db.get_session(session_id)
        existing_title = (existing.get("title") or "").strip() if existing else ""
        truncated = text[:80]
        # Only replace if the current title is our auto-generated truncated version
        if existing_title and existing_title != truncated:
            return  # User has set a custom title, don't overwrite
    except Exception:
        return
    title = generate_session_title(text)
    if title:
        db.rename_session(session_id, title)


# ---------------------------------------------------------------------------
# AI-powered PDF report generation -- after each conversation the LLM
# summarises the answer into a structured report which is rendered as a
# downloadable PDF using reportlab (with CJK font support).
# ---------------------------------------------------------------------------
import io as _io  # noqa: E402  (local import to avoid top-level clutter)
from datetime import datetime as _datetime  # noqa: E402

_CJK_FONT_REGISTERED = False


def _ensure_cjk_font() -> str:
    """Register the STSong-Light CJK font for reportlab (idempotent)."""
    global _CJK_FONT_REGISTERED
    if not _CJK_FONT_REGISTERED:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont

        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        _CJK_FONT_REGISTERED = True
    return "STSong-Light"


def generate_report_content(
    answer_text: str, user_question: str
) -> tuple[str, str, list[dict[str, str]]] | None:
    """Ask the LLM to produce a structured report from the conversation.

    Returns ``(title, summary, sections)`` where *sections* is a list of
    ``{"heading": ..., "content": ...}`` dicts, or ``None`` on failure.
    """
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=MODEL_CONFIG["model_name"],
            api_key=MODEL_CONFIG["api_key"] or "EMPTY",
            base_url=MODEL_CONFIG["base_url"] or None,
            temperature=0.3,
            request_timeout=60,
            extra_body=MODEL_CONFIG["generate_args"] or None,
        )
        prompt = (
            "你是一个灾害评估报告生成助手。请根据以下用户问题与AI回答，生成一份结构化的评估报告。\n\n"
            f"用户问题：\n{user_question[:2000]}\n\n"
            f"AI回答：\n{answer_text[:4000]}\n\n"
            "请返回纯JSON（不要包含```json标记或其他文字），格式如下：\n"
            '{"title":"报告标题（简短，不超过20字）",'
            '"summary":"报告简要说明（1-2句话，说明报告的主要内容和目的）",'
            '"sections":[{"heading":"章节标题","content":"章节正文（可含多段，用\\n分隔）"}]}\n\n'
            "要求：\n"
            "1. 报告语言与对话语言一致（通常为中文）\n"
            "2. 至少包含'分析概述'、'主要发现'、'结论与建议'三个章节\n"
            "3. 内容基于AI回答，不要编造信息\n"
            "4. 只返回JSON"
        )
        result = llm.invoke([
            SystemMessage(content="你是一个专业的报告生成助手，只返回有效的JSON。"),
            HumanMessage(content=prompt),
        ])
        raw = message_content_text(getattr(result, "content", "")).strip()
        # Strip possible markdown fences
        if raw.startswith("```"):
            raw = raw.split("```", 2)
            raw = raw[1] if len(raw) >= 2 else raw[0]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        data = json.loads(raw)
        title = str(data.get("title", "灾害评估报告")).strip()[:50]
        summary = str(data.get("summary", "")).strip()
        sections = []
        for sec in data.get("sections", []):
            heading = str(sec.get("heading", "")).strip()
            content = str(sec.get("content", "")).strip()
            if heading or content:
                sections.append({"heading": heading, "content": content})
        if not sections:
            sections = [{"heading": "报告内容", "content": answer_text[:2000]}]
        return title, summary, sections
    except Exception:
        return None


def _escape_xml(text: str) -> str:
    """Escape characters that are special in reportlab's Paragraph markup."""
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_pdf_report(
    title: str,
    summary: str,
    sections: list[dict[str, str]],
    user_question: str = "",
) -> bytes | None:
    """Render a structured report as a PDF using reportlab with CJK fonts."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            HRFlowable,
        )

        font = _ensure_cjk_font()
        buffer = _io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=25 * mm,
            rightMargin=25 * mm,
            topMargin=25 * mm,
            bottomMargin=25 * mm,
            title=title,
        )

        title_style = ParagraphStyle(
            "ReportTitle",
            fontName=font,
            fontSize=20,
            alignment=TA_CENTER,
            spaceAfter=6 * mm,
            leading=26,
        )
        meta_style = ParagraphStyle(
            "ReportMeta",
            fontName=font,
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.grey,
            spaceAfter=4 * mm,
        )
        summary_style = ParagraphStyle(
            "ReportSummary",
            fontName=font,
            fontSize=10,
            alignment=TA_LEFT,
            leading=16,
            spaceAfter=6 * mm,
            textColor=colors.HexColor("#333333"),
            leftIndent=6 * mm,
            rightIndent=6 * mm,
        )
        heading_style = ParagraphStyle(
            "ReportHeading",
            fontName=font,
            fontSize=14,
            alignment=TA_LEFT,
            spaceBefore=8 * mm,
            spaceAfter=3 * mm,
            leading=18,
            textColor=colors.HexColor("#1a1a1a"),
        )
        body_style = ParagraphStyle(
            "ReportBody",
            fontName=font,
            fontSize=10,
            alignment=TA_LEFT,
            leading=16,
            spaceAfter=2 * mm,
        )

        story: list[Any] = []
        story.append(Paragraph(_escape_xml(title), title_style))
        story.append(
            Paragraph(
                f"生成时间：{_datetime.now().strftime('%Y-%m-%d %H:%M')}",
                meta_style,
            )
        )
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
        story.append(Spacer(1, 4 * mm))

        if summary:
            story.append(Paragraph(_escape_xml(summary), summary_style))
            story.append(Spacer(1, 2 * mm))

        for section in sections:
            heading = section.get("heading", "")
            content = section.get("content", "")
            if heading:
                story.append(Paragraph(_escape_xml(heading), heading_style))
            for para in content.split("\n"):
                para = para.strip()
                if para:
                    story.append(Paragraph(_escape_xml(para), body_style))
            story.append(Spacer(1, 2 * mm))

        doc.build(story)
        return buffer.getvalue()
    except Exception:
        return None


def generate_and_store_pdf_report(
    answer_text: str,
    user_question: str,
    session_id: str,
    message_id: int | None = None,
) -> dict[str, Any] | None:
    """Generate a PDF report from the assistant answer and register it.

    If *message_id* is provided the report binary is persisted to the DB via
    ``db.update_message_report_files`` so that history reloads show the
    report card.  Returns a dict suitable for the frontend ``report`` field,
    or ``None`` on failure.
    """
    result = generate_report_content(answer_text, user_question)
    if result is None:
        # Fallback: generate a minimal report without LLM
        title = "灾害评估报告"
        summary = "基于对话内容自动生成的评估报告。"
        sections = [{"heading": "分析结果", "content": answer_text[:3000]}]
    else:
        title, summary, sections = result

    pdf_bytes = render_pdf_report(title, summary, sections, user_question)
    if not pdf_bytes:
        return None

    file_id = uuid.uuid4().hex
    safe_title = re.sub(r'[\\/:*?"<>|]', "_", title)[:40]
    file_name = f"{safe_title}.pdf"
    temp_path = TEMP_BASE / f"{file_id}_{file_name}"
    temp_path.write_bytes(pdf_bytes)
    FILES[file_id] = temp_path

    report_description = summary or "基于本次对话生成的评估报告"
    report_file_data = {
        "name": file_name,
        "mime_type": "application/pdf",
        "data_base64": base64.b64encode(pdf_bytes).decode("utf-8"),
        "description": report_description,
    }

    if message_id is not None:
        db.update_message_report_files(message_id, [report_file_data])

    return {
        "url": f"/api/files/{file_id}",
        "name": file_name,
        "description": report_description,
    }


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


def _file_descriptor(path: str) -> dict[str, str]:
    return {"name": Path(path).name, "path": str(path)}


def _serialize_message(row: dict) -> dict[str, Any]:
    images: list[dict[str, str]] = []

    # 优先从二进制字段读取图片
    image_files = row.get("image_files") or []
    for img_file in image_files:
        if not isinstance(img_file, dict):
            continue
        name = img_file.get("name", "image")
        mime_type = img_file.get("mime_type", "image/png")
        data_b64 = img_file.get("data_base64")
        if data_b64:
            # 生成一个临时 file_id 用于访问
            file_id = uuid.uuid4().hex
            # 将 base64 解码并写入临时文件
            import base64
            data = base64.b64decode(data_b64)
            temp_path = TEMP_BASE / f"{file_id}_{name}"
            temp_path.write_bytes(data)
            FILES[file_id] = temp_path
            images.append({"name": name, "url": f"/api/files/{file_id}"})

    # 如果二进制字段为空，回退到路径方式
    if not images:
        for img in row.get("images") or []:
            if not isinstance(img, dict):
                continue
            path = img.get("path")
            if path and Path(path).exists():
                images.append(file_payload(path))
            elif img.get("url"):
                images.append({"name": img.get("name", "image"), "url": img["url"]})

    # 处理附件文件（从二进制字段）
    attachments = []
    attachment_files = row.get("attachment_files") or []
    for att_file in attachment_files:
        if not isinstance(att_file, dict):
            continue
        name = att_file.get("name", "file")
        mime_type = att_file.get("mime_type", "application/octet-stream")
        data_b64 = att_file.get("data_base64")
        if data_b64:
            file_id = uuid.uuid4().hex
            import base64
            data = base64.b64decode(data_b64)
            temp_path = TEMP_BASE / f"{file_id}_{name}"
            temp_path.write_bytes(data)
            FILES[file_id] = temp_path
            attachments.append({
                "name": name,
                "url": f"/api/files/{file_id}",
                "mime_type": mime_type,
            })

    # 如果附件二进制为空，回退到路径方式
    if not attachments:
        for item in row.get("attachments") or []:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            if path and Path(path).exists():
                attachments.append(file_payload(path))
            elif item.get("url"):
                attachments.append({"name": item.get("name", "file"), "url": item["url"]})
            else:
                attachments.append(item)

    # 处理 PDF 报告文件（从二进制字段读取，生成前端可访问的 URL）
    report = None
    report_files = row.get("report_files") or []
    for rf in report_files:
        if not isinstance(rf, dict):
            continue
        name = rf.get("name", "report.pdf")
        data_b64 = rf.get("data_base64")
        if data_b64:
            file_id = uuid.uuid4().hex
            data = base64.b64decode(data_b64)
            temp_path = TEMP_BASE / f"{file_id}_{name}"
            temp_path.write_bytes(data)
            FILES[file_id] = temp_path
            report = {
                "url": f"/api/files/{file_id}",
                "name": name,
                "description": rf.get("description") or "基于本次对话生成的评估报告",
            }
            break

    raw_content = row.get("content") or ""
    display_content = row.get("display_content")
    if display_content is not None:
        rendered_content = display_content
    elif row["role"] == "user":
        # 旧数据没有 display_content，去掉后端自动追加的 Uploaded files 路径块
        rendered_content = re.sub(
            r"\n?\n?Uploaded files:\n(?:- `[^`]+`\n?)+",
            "",
            raw_content,
        ).strip()
    else:
        rendered_content = raw_content

    return {
        "id": row["id"],
        "role": row["role"],
        "content": rendered_content,
        "attachments": attachments,
        "images": images,
        "legend": row.get("legend") or [],
        "tool_trace": row.get("tool_trace") or [],
        "elapsed_seconds": row.get("elapsed_seconds"),
        "tool_call_count": row.get("tool_call_count"),
        "created_at": row.get("created_at"),
        "report": report,
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


@app.get("/api/sessions/{session_id}/latest-geometry")
def get_session_latest_geometry(session_id: str) -> dict[str, Any]:
    """返回某会话最新的空间范围, 用于前端地图自动定位。"""
    rows = db.query_assessments(session_id=session_id, limit=20)
    for row in rows:
        if row.get("geom_geojson"):
            assessment = _serialize_assessment(row)
            return {
                "session_id": session_id,
                "found": True,
                "assessment": assessment,
                "geom": assessment["geom"],
            }
    return {"session_id": session_id, "found": False, "assessment": None, "geom": None}


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


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, bool]:
    """删除整个会话及其所有消息"""
    success = db.delete_session(session_id)
    # 清理内存中的会话缓存
    with SESSIONS_LOCK:
        SESSIONS.pop(session_id, None)
    return {"ok": success}


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
            attachment_files: list[dict] = []
            for upload in files or []:
                if not upload.filename:
                    continue
                safe_name = Path(upload.filename).name
                save_path = session.uploads_dir / safe_name
                file_data = upload.file.read()
                with open(save_path, "wb") as out:
                    out.write(file_data)
                uploaded_paths.append(str(save_path))

                # 读取文件二进制数据用于数据库存储
                import base64
                attachment_files.append({
                    "name": safe_name,
                    "mime_type": upload.content_type or "application/octet-stream",
                    "data_base64": base64.b64encode(file_data).decode("utf-8"),
                })

            content_parts = [message.strip()] if message.strip() else []
            if uploaded_paths:
                content_parts.append(
                    "Uploaded files:\n" + "\n".join(f"- `{path}`" for path in uploaded_paths)
                )
            user_content = "\n\n".join(content_parts) or "Please analyze the uploaded file(s)."
            session.messages.append({
                "role": "user",
                "content": build_user_message_content(user_content, uploaded_paths),
            })

            # Persist session + user message to DB (upsert; idempotent)
            db.create_session(
                session_id=session_id,
                model_name=MODEL_CONFIG["model_name"],
                config_path=MODEL_CONFIG["path"],
                system_prompt=system_prompt,
                title=(message.strip()[:80] or None),
            )
            # AI 总结长输入为会话标题 (后台执行，不阻塞主流程)
            threading.Thread(
                target=maybe_generate_session_title,
                args=(session_id, message),
                daemon=True,
            ).start()
            db.save_chat_message(
                session_id,
                "user",
                content=user_content,
                display_content=message.strip(),
                attachments=[{"name": Path(p).name, "path": p} for p in uploaded_paths],
                attachment_files=attachment_files,
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
            trace = tool_trace(response)
            output_paths = extract_tool_output_files(response)
            reviewed_answer = session.handle.review_answer_and_artifacts(
                raw_answer,
                uploaded_paths,
                output_paths,
            )
            final_answer = extract_final_answer(reviewed_answer)
            display_answer = sanitize_display_answer(final_answer or reviewed_answer)
            images, output_files = choose_frontend_artifacts(reviewed_answer, output_paths)
            geometry = extract_first_output_geometry(output_paths)
            legend = extract_tool_legend(response)

            session.messages.append(
                {"role": "assistant", "content": final_answer or reviewed_answer or raw_answer}
            )
            # 读取输出图片的二进制数据
            image_files: list[dict] = []
            for img_path in images:
                try:
                    img_data = Path(img_path).read_bytes()
                    import base64
                    image_files.append({
                        "name": Path(img_path).name,
                        "mime_type": "image/png",
                        "data_base64": base64.b64encode(img_data).decode("utf-8"),
                    })
                except Exception:
                    pass

            _msg_id = db.save_chat_message(
                session_id,
                "assistant",
                content=final_answer or reviewed_answer or raw_answer,
                display_content=display_answer,
                attachments=[_file_descriptor(p) for p in output_files],
                images=[_image_descriptor(p) for p in images],
                tool_trace=trace,
                elapsed_seconds=elapsed,
                tool_call_count=len(trace),
                image_files=image_files,
                legend=legend,
            )

            # 生成 PDF 报告
            report = generate_and_store_pdf_report(
                display_answer or final_answer or raw_answer,
                message,
                session_id,
                message_id=_msg_id,
            )

            return {
                "answer": display_answer or "(empty response)",
                "elapsed": elapsed,
                "tool_calls": len(trace),
                "images": [file_payload(path) for path in images],
                "files": [file_payload(path) for path in output_files],
                "geometry": geometry,
                "legend": legend,
                "trace": trace if show_trace else [],
                "report": report,
            }
        except Exception as exc:  # noqa: BLE001
            error_text = "".join(traceback.format_exception(exc))
            matches = ERROR_MEMORY.lookup_all(error_text)
            return {
                "answer": f"后端调用失败：{exc}",
                "elapsed": 0,
                "tool_calls": 0,
                "images": [],
                "files": [],
                "geometry": None,
                "legend": [],
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
        attachment_files: list[dict] = []
        for upload in files or []:
            if not upload.filename:
                continue
            safe_name = Path(upload.filename).name
            save_path = session.uploads_dir / safe_name
            file_data = upload.file.read()
            with open(save_path, "wb") as out:
                out.write(file_data)
            uploaded_paths.append(str(save_path))

            # 读取文件二进制数据用于数据库存储
            import base64
            attachment_files.append({
                "name": safe_name,
                "mime_type": upload.content_type or "application/octet-stream",
                "data_base64": base64.b64encode(file_data).decode("utf-8"),
            })

        content_parts = [message.strip()] if message.strip() else []
        if uploaded_paths:
            content_parts.append(
                "Uploaded files:\n" + "\n".join(f"- `{path}`" for path in uploaded_paths)
            )
        user_content = "\n\n".join(content_parts) or "Please analyze the uploaded file(s)."
        session.messages.append({
            "role": "user",
            "content": build_user_message_content(user_content, uploaded_paths),
        })

        # Persist session + user message to DB (upsert; idempotent)
        db.create_session(
            session_id=session_id,
            model_name=MODEL_CONFIG["model_name"],
            config_path=MODEL_CONFIG["path"],
            system_prompt=system_prompt,
            title=(message.strip()[:80] or None),
        )
        # AI 总结长输入为会话标题 (后台执行，不阻塞主流程)
        threading.Thread(
            target=maybe_generate_session_title,
            args=(session_id, message),
            daemon=True,
        ).start()
        db.save_chat_message(
            session_id,
            "user",
            content=user_content,
            display_content=message.strip(),
            attachments=[{"name": Path(p).name, "path": p} for p in uploaded_paths],
            attachment_files=attachment_files,
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
        # 保存 final 块的上下文，用于 for 循环结束后生成 PDF 报告
        _final_answer_text = ""
        _final_msg_id: int | None = None
        _final_user_question = message
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
                    trace = tool_trace(response)
                    output_paths = extract_tool_output_files(response)
                    reviewed_answer = session.handle.review_answer_and_artifacts(
                        raw_answer or streamed_text,
                        uploaded_paths,
                        output_paths,
                    )
                    final_answer = extract_final_answer(reviewed_answer)
                    display_answer = sanitize_display_answer(
                        final_answer or reviewed_answer or raw_answer or streamed_text
                    )
                    images, output_files = choose_frontend_artifacts(reviewed_answer, output_paths)
                    geometry = extract_first_output_geometry(output_paths)
                    legend = extract_tool_legend(response)

                    session.messages.append(
                        {"role": "assistant", "content": final_answer or reviewed_answer or raw_answer or streamed_text}
                    )
                    # 读取输出图片的二进制数据
                    image_files: list[dict] = []
                    for img_path in images:
                        try:
                            img_data = Path(img_path).read_bytes()
                            import base64
                            image_files.append({
                                "name": Path(img_path).name,
                                "mime_type": "image/png",
                                "data_base64": base64.b64encode(img_data).decode("utf-8"),
                            })
                        except Exception:
                            pass

                    _final_msg_id = db.save_chat_message(
                        session_id,
                        "assistant",
                        content=final_answer or reviewed_answer or raw_answer or streamed_text,
                        display_content=display_answer,
                        attachments=[_file_descriptor(p) for p in output_files],
                        images=[_image_descriptor(p) for p in images],
                        tool_trace=trace,
                        elapsed_seconds=elapsed,
                        tool_call_count=len(trace),
                        image_files=image_files,
                        legend=legend,
                    )
                    _final_answer_text = display_answer or final_answer or streamed_text

                    yield sse_event(
                        "done",
                        {
                            "answer": display_answer or "(empty response)",
                            "elapsed": elapsed,
                            "tool_calls": len(trace),
                            "images": [file_payload(path) for path in images],
                            "files": [file_payload(path) for path in output_files],
                            "geometry": geometry,
                            "legend": legend,
                            "trace": trace if show_trace else [],
                        },
                    )

            # 对话结束后生成 PDF 报告
            if _final_answer_text:
                yield sse_event("status", {"message": "正在生成报告..."})
                report = generate_and_store_pdf_report(
                    _final_answer_text,
                    _final_user_question,
                    session_id,
                    message_id=_final_msg_id,
                )
                if report:
                    yield sse_event("report", report)
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
                    "files": [],
                    "geometry": None,
                    "legend": [],
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
