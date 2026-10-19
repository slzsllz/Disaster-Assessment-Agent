"""Restrict model-supplied file inputs to the current conversation's data."""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool, ToolException


OUTPUT_ARGUMENTS = {
    "output_dir", "output_path", "output_paths", "output_prefix", "output_name",
    "save_path", "save_name",
}
USER_PATH_RE = re.compile(
    r'''`([^`\n]+)`|"([^"\n]+)"|'([^'\n]+)'|(?<![\w:/])((?:/|~/|\./)[^\s`"'<>，。；、（）()\[\]{}]+)'''
)
INPUT_ERROR = (
    "This input was not provided by the user or generated in this conversation. "
    "Do not guess filenames, use server examples, or substitute another conversation's data. "
    "Ask the user to upload the intended file or explicitly specify its exact path. "
    "Do not invent a disaster location or query an arbitrary city's weather."
)


@dataclass
class InputScope:
    roots: tuple[Path, ...]
    explicit_paths: set[Path]

    def check(self, value: str) -> None:
        path = Path(value).expanduser().resolve()
        if path.exists() and (
            path in self.explicit_paths
            or any(path.is_relative_to(root) for root in self.roots)
        ):
            return
        raise ToolException(INPUT_ERROR)


_input_scope: ContextVar[InputScope | None] = ContextVar("tool_input_scope", default=None)


@contextmanager
def tool_input_scope(temp_dir: Path, messages: list):
    explicit_paths = set()
    for message in messages:
        if not isinstance(message, HumanMessage):
            continue
        blocks = message.content if isinstance(message.content, list) else [message.content]
        for block in blocks:
            text = block.get("text", "") if isinstance(block, dict) else block
            if not isinstance(text, str):
                continue
            for match in USER_PATH_RE.finditer(text):
                value = next(part for part in match.groups() if part).strip()
                if "/" in value and not "://" in value:
                    explicit_paths.add(Path(value).expanduser().resolve())
    token = _input_scope.set(InputScope(
        roots=((temp_dir / "uploads").resolve(), (temp_dir / "out").resolve()),
        explicit_paths=explicit_paths,
    ))
    try:
        yield
    finally:
        _input_scope.reset(token)


def _input_values(arguments: dict[str, Any]):
    for key, value in arguments.items():
        if key in OUTPUT_ARGUMENTS:
            continue
        if isinstance(value, dict):
            yield from _input_values(value)
        elif "path" in key or "file" in key or key.endswith(("_dir", "_directory")):
            values = value if isinstance(value, list) else [value]
            yield from (item for item in values if isinstance(item, str) and item.strip())


def guard_tool_inputs(tool: StructuredTool) -> StructuredTool:
    async def call(**arguments):
        scope = _input_scope.get()
        if scope is None:
            raise ToolException("Missing conversation input scope. " + INPUT_ERROR)
        if tool.name == "detect_fire_burned_area_change" and not (
            arguments.get("pre_image_path") or arguments.get("post_image_path")
            or arguments.get("dataset_dir")
        ):
            raise ToolException("No input images or explicit dataset directory were supplied. " + INPUT_ERROR)
        for value in _input_values(arguments):
            scope.check(value)
        return await tool.coroutine(**arguments)

    return tool.model_copy(update={"coroutine": call})
