"""Route the reviewed disaster-tool catalogue without binding every schema."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


MAX_TOOLS_PER_TURN = 16
FALLBACK_TOOLS = (
    "assess_building_damage", "extract_flood_inundation",
    "detect_fire_burned_area_change", "extract_oil_spill_area",
    "detect_algal_bloom", "geoai_semantic_segmentation",
)
LIVE_SEARCH_RE = re.compile(
    r"联网|上网|网上|搜索|检索网页|查新闻|最新|近期|实时|今天|本周|"
    r"当前.{0,12}(?:灾情|新闻|预警)|search (?:the )?web|latest|recent|today|current news",
    re.IGNORECASE,
)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def parse_tool_names(response: str, available: set[str]) -> list[str]:
    """Accept a small JSON object; ignore invented and duplicate names."""
    try:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        payload = json.loads(match.group(0) if match else response)
        raw = payload.get("tools", []) if isinstance(payload, dict) else []
    except (ValueError, TypeError):
        raw = []
    if not isinstance(raw, list):
        return []
    selected: list[str] = []
    for name in raw:
        if isinstance(name, str) and name in available and name not in selected:
            selected.append(name)
    return selected[:MAX_TOOLS_PER_TURN]


class ToolRouter:
    """Route each user turn once among tools approved for this agent."""

    def __init__(self, llm: Any, tools: Sequence[Any]):
        self.llm = llm
        self.tools = list(tools)
        self.by_name = {tool.name: tool for tool in tools}
        if len(self.by_name) != len(self.tools):
            raise ValueError("MCP tool names must be unique across servers")
        self._last_question: str | None = None
        self._selected: list[Any] = []
        self._retrieved_descriptions: list[str] = []
        self.catalogue = "\n".join(
            f"{tool.name}: {self._summary(tool.description)}" for tool in self.tools
        )

    @staticmethod
    def _summary(description: str) -> str:
        summary = re.sub(r"\s+", " ", description or "").strip()
        return summary[:110]

    @staticmethod
    def _fallback_score(name: str, question: str) -> int:
        tokens = [part for part in name.lower().split("_") if len(part) >= 3]
        return sum(len(part) for part in tokens if part in question.lower())

    def set_retrieved_tools(self, hits: Sequence[Any]) -> None:
        """Pass the current turn's full retrieved tool descriptions to routing."""
        self._retrieved_descriptions = [
            f"{hit.document.source_id}: {hit.document.metadata.get('description', '')[:1400]}"
            for hit in hits if hit.document.source_id in self.by_name
        ][:4]
        self._last_question = None

    async def select(self, messages: Sequence[Any]) -> list[Any]:
        conversational = [
            message for message in messages
            if isinstance(message, (HumanMessage, AIMessage))
            and getattr(message, "name", None) != "retrieval_context"
            and _message_text(message.content).strip()
        ]
        question = next(
            (_message_text(message.content) for message in reversed(conversational)
             if isinstance(message, HumanMessage)),
            "",
        )[:2000]
        last_user_index = max(
            (index for index, message in enumerate(conversational)
             if isinstance(message, HumanMessage)),
            default=-1,
        )
        context = "\n".join(
            f"{'User' if isinstance(message, HumanMessage) else 'Assistant'}: "
            f"{_message_text(message.content)[:700]}"
            for message in conversational[max(0, last_user_index - 3):last_user_index + 1]
        )
        if context == self._last_question:
            return self._selected

        explicit = [name for name in self.by_name if name.lower() in question.lower()]
        selected_names: list[str] = []
        routing_failed = False
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=(
                    "You select tools for a remote-sensing/disaster-assessment agent. "
                    "Choose at most 16 exact tool names from the catalogue. "
                    "Include tools needed for likely intermediate steps, but avoid unrelated tools. "
                    "Choose web_search for current events, recent warnings, news, or explicit internet search. "
                    "For a purely conversational question, return an empty list. "
                    "Reply ONLY with JSON: {\"tools\": [\"exact_name\", ...]}.\n\n"
                    "Available tools:\n" + self.catalogue +
                    ("\n\nFull descriptions of retrieved candidates:\n" +
                     "\n".join(self._retrieved_descriptions)
                     if self._retrieved_descriptions else "")
                )),
                HumanMessage(content=context),
            ])
            router_text = _message_text(response.content)
            selected_names = parse_tool_names(router_text, set(self.by_name))
            if not selected_names and not re.search(r'"tools"\s*:\s*\[\s*\]', router_text):
                routing_failed = True
        except Exception:
            # The main answer must still work if the routing LLM call fails.
            routing_failed = True

        live = ["web_search"] if "web_search" in self.by_name and LIVE_SEARCH_RE.search(question) else []
        names = list(dict.fromkeys([*live, *explicit, *selected_names]))[:MAX_TOOLS_PER_TURN]
        if not names and question:
            ranked = sorted(
                self.by_name, key=lambda name: self._fallback_score(name, question),
                reverse=True,
            )
            names = [name for name in ranked if self._fallback_score(name, question) > 0][:MAX_TOOLS_PER_TURN]
        if not names and routing_failed:
            names = [name for name in FALLBACK_TOOLS if name in self.by_name]
        self._selected = [self.by_name[name] for name in names]
        self._last_question = context
        return self._selected
