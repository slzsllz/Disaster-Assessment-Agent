"""Give tool failures back to the agent and learn observed execution recoveries."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import StructuredTool, ToolException

from agent.error_memory import ErrorMemory, normalize_error


logger = logging.getLogger(__name__)
MAX_FAILURES = 3
# Only execution settings may be learned automatically. Changes to input data,
# model choice or scientific parameters do not prove a reusable repair.
EXECUTION_ARGS = {
    "device", "batch_size", "tile_size", "patch_size", "chunk_size",
    "num_workers", "use_gpu",
}
OUTPUT_ARGS = {"output_dir", "output_path", "save_path", "output_prefix", "output_name"}


@dataclass
class Attempt:
    failures: int = 0
    error: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    last_failure: str = ""
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class RecoveryRun:
    attempts: dict[tuple[str, str], Attempt] = field(default_factory=dict)


_current_run: ContextVar[RecoveryRun | None] = ContextVar("tool_recovery_run", default=None)


@contextmanager
def recovery_run():
    """Isolate retry budgets and repair evidence to a single user request."""
    token = _current_run.set(RecoveryRun())
    try:
        yield
    finally:
        _current_run.reset(token)


def _attempt(tool_name: str, arguments: dict[str, Any]) -> Attempt:
    run = _current_run.get()
    if run is None:
        return Attempt()
    identity = {
        key: value for key, value in arguments.items()
        if key not in EXECUTION_ARGS | OUTPUT_ARGS
    }
    key = (tool_name, json.dumps(identity, sort_keys=True, ensure_ascii=False, default=str))
    return run.attempts.setdefault(key, Attempt())


def _exception_text(exc: Exception) -> str:
    children = getattr(exc, "exceptions", None)
    if children:
        return "\n".join(_exception_text(child) for child in children)
    return f"{type(exc).__name__}: {exc}"


def _result_error(content: Any) -> str:
    if isinstance(content, str):
        try:
            return _result_error(json.loads(content))
        except (ValueError, TypeError):
            if re.match(r"^(?:error|\w*Error|exception|错误|失败)\s*[:：]", content.strip(), re.I):
                return content
            return ""
    if isinstance(content, list):
        for block in content:
            error = _result_error(block.get("text", "") if isinstance(block, dict) else block)
            if error:
                return error
    if isinstance(content, dict):
        if (
            content.get("success") is False
            or content.get("isError") is True
            or str(content.get("status", "")).lower() in {"error", "failed", "failure"}
            or (content.get("error") and content.get("success") is not True)
        ):
            detail = content.get("error") or content.get("message") or content
            return "Tool returned failure: " + json.dumps(detail, ensure_ascii=False, default=str)
    return ""


def _failure(
    memory: ErrorMemory, name: str, arguments: dict[str, Any], error: str, attempt: Attempt,
    *, retrieve_experiences: bool = True,
) -> str:
    attempt.failures += 1
    attempt.error = error
    attempt.arguments = dict(arguments)
    matches = memory.lookup_all(error, tool_name=name) if retrieve_experiences else []
    payload = json.dumps(
        {
            "success": False,
            "error": error[:6000],
            "error_recovery": {
                "tool": name,
                "remaining_attempts": max(0, MAX_FAILURES - attempt.failures),
                "matched_experiences": [
                    {"pattern": pattern, "fix": fix} for pattern, fix in matches
                ],
                "instruction": (
                    "Read the error and matched experiences. Retry only with a justified "
                    "correction supported by the tool schema and uploaded inputs. If no "
                    "valid correction exists, explain the blocker or ask for missing input. "
                    "Do not change the requested model or scientific parameters silently."
                    if attempt.failures < MAX_FAILURES else
                    "Retry budget exhausted. Do not call this tool again for this input. "
                    "Explain the failure without inventing a successful result."
                ),
            },
        }, ensure_ascii=False,
    )
    attempt.last_failure = payload
    logger.info(
        "Tool %s failed (attempt %s/%s); matched %s error experiences",
        name, attempt.failures, MAX_FAILURES, len(matches),
    )
    return payload


def _learn(memory: ErrorMemory, name: str, arguments: dict[str, Any], attempt: Attempt) -> None:
    if not attempt.error:
        return
    before = {key: value for key, value in attempt.arguments.items() if key not in EXECUTION_ARGS}
    after = {key: value for key, value in arguments.items() if key not in EXECUTION_ARGS}
    if before != after:
        attempt.error = ""
        return
    changes = {
        key: value for key, value in arguments.items()
        if key in EXECUTION_ARGS and attempt.arguments.get(key) != value
        and isinstance(value, (bool, int, float, str))
        and (not isinstance(value, str) or value in {"cpu", "cuda", "cuda:0", "mps", "auto"})
    }
    if changes:
        # No filenames, image contents or user-specific paths are saved.
        pattern = f"{name}::{normalize_error(attempt.error)[:400]}"
        fix = (
            "For the same tool and input, execution succeeded after changing these "
            f"execution arguments: {json.dumps(changes, sort_keys=True)}. "
            "Use only if appropriate for the current task. This verifies execution "
            "recovery, not the scientific accuracy of the detection."
        )
        if memory.record(pattern, fix):
            logger.info("Recorded execution recovery for %s; arguments: %s", name, sorted(changes))
    attempt.error = ""


def wrap_tool(tool: StructuredTool, memory: ErrorMemory) -> StructuredTool:
    """Keep the MCP schema, content and artifact contract unchanged on success."""
    if not isinstance(tool, StructuredTool) or tool.coroutine is None:
        raise TypeError(f"Expected an asynchronous MCP StructuredTool: {tool.name}")
    schema = tool.args_schema
    if not isinstance(schema, dict):
        schema = schema.model_json_schema()
    defaults = {
        key: specification["default"]
        for key, specification in schema.get("properties", {}).items()
        if "default" in specification
    }

    async def call(**arguments):
        effective_arguments = {**deepcopy(defaults), **arguments}
        attempt = _attempt(tool.name, effective_arguments)
        async with attempt.lock:
            if attempt.failures >= MAX_FAILURES:
                raise ToolException(attempt.last_failure)
            try:
                result = await tool.coroutine(**arguments)
                if tool.response_format == "content_and_artifact":
                    if not isinstance(result, tuple) or len(result) != 2:
                        raise ValueError("Expected a (content, artifact) tuple from the MCP adapter")
                    content = result[0]
                else:
                    content = result
                error = _result_error(content)
                if error:
                    raise ToolException(error)
            except Exception as exc:
                payload = await asyncio.to_thread(
                    _failure, memory, tool.name, effective_arguments, _exception_text(exc), attempt,
                )
                raise ToolException(payload) from exc
            try:
                await asyncio.to_thread(_learn, memory, tool.name, effective_arguments, attempt)
            except Exception:
                logger.exception("Could not learn tool recovery for %s", tool.name)
            return result

    def validation_error(exc) -> str:
        # LangChain runs this callback synchronously on the event loop. Schema
        # failures already explain the correction; do no database/embedding I/O.
        return _failure(
            memory, tool.name, {}, _exception_text(exc), _attempt(tool.name, {}),
            retrieve_experiences=False,
        )

    return tool.model_copy(update={
        "coroutine": call,
        "handle_tool_error": lambda exc: str(exc),
        "handle_validation_error": validation_error,
    })
