"""Tool-level error hints and evidence of successful corrective retries."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from agent.error_memory import ErrorMemory


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content or "")


def tool_error(message: ToolMessage) -> str | None:
    """Recognize explicit tool failures, without treating arbitrary output as an error."""
    content = _content_text(message.content).strip()
    if message.status == "error":
        return content[:1000] or "Tool execution failed"
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        payload = None
    if isinstance(payload, dict):
        if payload.get("isError") is True or payload.get("success") is False:
            return content[:1000]
        if payload.get("error") and not payload.get("result"):
            return content[:1000]
    if re.match(r"^(error|exception|traceback)\s*[:\n]", content, re.I):
        return content[:1000]
    return None


def _tool_calls(messages: list[Any]) -> dict[str, tuple[str, dict[str, Any]]]:
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            if call.get("id"):
                calls[call["id"]] = (call.get("name") or "", call.get("args") or {})
    return calls


def _signature(error: str) -> str:
    """Persist only the exception class, never user data from the error text."""
    match = re.search(r"\b[A-Za-z_]\w*(?:Error|Exception)\b", error)
    return match.group(0) if match else "Tool reported an error"


class ToolErrorFeedback:
    def __init__(self, memory: ErrorMemory, db: Any = None):
        self.memory = memory
        self.db = db
        self._recorded: set[str] = set()

    def before_model(self, state: dict[str, Any]) -> dict[str, list[Any]]:
        messages = list(state.get("messages") or [])
        calls = _tool_calls(messages)
        errors: list[tuple[int, str, str, dict[str, Any]]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, ToolMessage):
                continue
            name, args = calls.get(message.tool_call_id, (message.name or "", {}))
            error = tool_error(message)
            if error:
                errors.append((index, name, error, args))
                continue
            # A later successful call to the *same* tool is evidence of recovery.
            # Save only a pending candidate, never an unreviewed prompt rule.
            previous = next((entry for entry in reversed(errors) if entry[1] == name), None)
            if previous is None or previous[3] == args:
                continue
            key = message.id or message.tool_call_id
            if not key or key in self._recorded:
                continue
            changed = sorted(
                field for field in previous[3].keys() | args.keys()
                if previous[3].get(field) != args.get(field)
            )
            if self.db is not None and changed:
                self.db.save_error_recovery_candidate(
                    tool_name=name,
                    error_signature=_signature(previous[2]),
                    error_hash=hashlib.sha256(previous[2].encode()).hexdigest(),
                    changed_fields=changed,
                )
            self._recorded.add(key)
            if len(self._recorded) > 2048:
                self._recorded.clear()

        # Only the latest tool batch needs hints. Earlier errors have already
        # been seen by the model, so do not repeatedly inflate the context.
        last_ai = max((i for i, msg in enumerate(messages) if isinstance(msg, AIMessage)), default=-1)
        recent_errors = [item for item in errors if item[0] > last_ai]
        hints: list[str] = []
        for _, name, error, _ in recent_errors:
            matches = self.memory.lookup_all(error, limit=2)
            for pattern, fix in matches:
                hints.append(f"Tool {name}, error pattern {pattern}: {fix}")
        if not hints:
            return {"llm_input_messages": messages}
        advice = SystemMessage(content=(
            "The latest tool call failed. Use only relevant approved error memory "
            "below; correct the tool arguments or workflow and retry at most once "
            "if safe. Do not retry irreversible actions automatically.\n" +
            "\n".join(hints[:3])
        ))
        # Keep the existing tool-call/tool-result sequence intact and retain
        # the application's system prompt as the first message.
        position = 1 if messages and isinstance(messages[0], SystemMessage) else 0
        return {"llm_input_messages": [*messages[:position], advice, *messages[position:]]}
