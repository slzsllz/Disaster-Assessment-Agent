"""Evidence collection and conservative checks for answer review."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

from langchain_core.messages import AIMessage, ToolMessage

from agent.error_feedback import tool_error
from agent.tool_router import LIVE_SEARCH_RE, WEB_FETCH_RE, requires_web_body
from agent.web_search import WEB_SOURCE_TOOL_NAMES, _public_url


REVIEW_VERSION = 3
MAX_EVIDENCE = 24
MAX_ARTIFACTS = 20


def _tool_payload(value: Any) -> Any:
    if isinstance(value, list):
        value = next((part.get("text") for part in value if isinstance(part, dict)
                      and part.get("type") == "text"), value)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value[:2000]
    return value


def _bounded(value: Any, limit: int = 2400) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        rendered = str(value)
    return rendered[:limit]


def _issue(code: str, severity: str, message: str, ids: list[str] | None = None) -> dict:
    return {"code": code, "severity": severity, "message": message,
            "evidence_ids": ids or []}


def _common_metric_issues(name: str, data: dict, evidence_id: str) -> list[dict]:
    issues = []
    fields = ("total_valid_pixels", "flood_pixels", "building_total", "no_damage",
              "minor_damage", "major_damage", "destroyed", "damaged", "burned_pixels",
              "total_pixels")
    for field in fields:
        value = data.get(field)
        if isinstance(value, (int, float)) and (not math.isfinite(value) or value < 0):
            issues.append(_issue("invalid_metric", "critical", f"{name}.{field} 不是有效非负数值", [evidence_id]))
    for field in ("damage_ratio", "flood_ratio", "burned_ratio"):
        value = data.get(field)
        if isinstance(value, (int, float)) and (not math.isfinite(value) or not 0 <= value <= 1):
            issues.append(_issue("invalid_ratio", "critical", f"{name}.{field} 不在 0–1 范围", [evidence_id]))
    return issues


def _damage_metric_issues(data: dict, evidence_id: str, answer: str) -> list[dict]:
    fields = ("building_total", "no_damage", "minor_damage", "major_damage",
              "destroyed", "damaged")
    if not all(isinstance(data.get(field), int) for field in fields):
        return []
    issues = []
    classes = sum(data[field] for field in ("no_damage", "minor_damage", "major_damage", "destroyed"))
    damaged = sum(data[field] for field in ("minor_damage", "major_damage", "destroyed"))
    if data["building_total"] != classes or data["damaged"] != damaged:
        issues.append(_issue("damage_count_mismatch", "critical", "建筑损毁像元分类统计不一致", [evidence_id]))
    if data.get("count_unit") == "pixels":
        building_claims = {int(number.replace(",", "")) for number in
                           re.findall(r"(?<!\d)(\d[\d,]*)\s*栋", answer)}
        pixel_counts = {data[field] for field in fields}
        if building_claims & pixel_counts:
            issues.append(_issue("pixel_count_as_buildings", "critical",
                                 "像元统计被表述为建筑物栋数", [evidence_id]))
    return issues


def _flood_metric_issues(data: dict, evidence_id: str, answer: str) -> list[dict]:
    if all(isinstance(data.get(field), int) for field in
           ("total_valid_pixels", "flood_pixels")):
        if data["flood_pixels"] > data["total_valid_pixels"]:
            return [_issue("flood_count_mismatch", "critical",
                           "洪水像元数超过有效像元数", [evidence_id])]
    return []


# Each tool contributes only its domain-specific checks; shared numeric checks run first.
TOOL_METRIC_VALIDATORS: dict[str, Callable[[dict, str, str], list[dict]]] = {
    "assess_building_damage": _damage_metric_issues,
    "extract_flood_inundation": _flood_metric_issues,
}


def _metric_issues(name: str, data: dict, evidence_id: str, answer: str = "") -> list[dict]:
    issues = _common_metric_issues(name, data, evidence_id)
    validator = TOOL_METRIC_VALIDATORS.get(name)
    if validator is not None:
        issues.extend(validator(data, evidence_id, answer))
    return issues


def _raster_metadata(path: Path) -> dict:
    import rasterio

    with rasterio.open(path) as src:
        bounds = [float(value) for value in src.bounds]
        if not all(math.isfinite(value) for value in bounds):
            raise ValueError("Raster bounds are not finite")
        return {"width": src.width, "height": src.height, "bands": src.count,
                "crs": str(src.crs) if src.crs else None,
                "bounds": [round(value, 6) for value in bounds]}


def _reported_output_paths(data: Any) -> list[str]:
    found: list[str] = []

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for field, item in value.items():
                visit(item, str(field).lower())
        elif isinstance(value, list):
            for item in value:
                visit(item, key)
        elif isinstance(value, str) and value.startswith("/") and (
            key.endswith("_path") or key in {"outputs", "output_paths"}
        ) and not key.startswith(("input", "pre_", "post_", "source_", "dataset_", "model_", "checkpoint_")):
            found.append(value)

    visit(data)
    return found[:40]


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def gather_review_evidence(
    response: dict, output_paths: list[str], references: list[dict], question: str,
    answer: str = "",
) -> tuple[list[dict], list[dict]]:
    """Create bounded evidence and identify verifiable failures before the LLM call."""
    messages = response.get("messages", [])
    names = {
        call.get("id"): call.get("name")
        for message in messages if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    evidence: list[dict] = []
    issues: list[dict] = []
    last_result_by_name: dict[str, tuple[str, bool]] = {}
    persisted = {str(Path(value).resolve()) for value in output_paths}
    for index, message in enumerate(messages):
        if not isinstance(message, ToolMessage):
            continue
        name = message.name or names.get(message.tool_call_id) or "tool"
        evidence_id = f"tool:{message.tool_call_id or index}"
        data = _tool_payload(message.content)
        failed = bool(tool_error(message))
        last_result_by_name[name] = (evidence_id, failed)
        if name in WEB_SOURCE_TOOL_NAMES and isinstance(data, dict) and data.get("success"):
            compact = dict(data)
            results = data.get("results") if isinstance(data.get("results"), list) else []
            compact["results"] = [
                {key: (value[:300] if key == "snippet" and isinstance(value, str) else value)
                 for key, value in item.items() if key in {"title", "url", "final_url", "snippet", "published_at",
                                                          "retrieved_at", "source", "text_chars", "text_sha256", "event_id", "data"}}
                for item in results[:5] if isinstance(item, dict)
            ]
            rendered = _bounded(compact, 3500 if name == "web_fetch" else 7000)
            if name == "web_fetch" and data.get("errors"):
                issues.append(_issue("web_fetch_partial", "warning", "部分网页正文未能抓取，不能视为已核实", [evidence_id]))
            if name == "disaster_feed" and (data.get("warnings") or data.get("truncated")):
                issues.append(_issue("official_feed_coverage", "warning", "官方数据源存在覆盖或条数限制，不能把查询结果当作完整灾情统计", [evidence_id]))
        else:
            rendered = _bounded(data)
        evidence.append({"id": evidence_id, "kind": "tool", "name": name,
                         "status": "failed" if failed else "succeeded",
                         "result": rendered})
        if len(evidence) > MAX_EVIDENCE:
            evidence.pop(0)
        if isinstance(data, dict) and not failed:
            issues.extend(_metric_issues(name, data, evidence_id, answer))
            for reported in _reported_output_paths(data):
                path = Path(reported)
                if _file_size(path) == 0:
                    issues.append(_issue("reported_artifact_missing", "critical",
                                         f"工具报告的产物 {path.name} 不存在或为空", [evidence_id]))
                elif str(path.resolve()) not in persisted:
                    issues.append(_issue("artifact_not_persisted", "warning",
                                         f"产物 {path.name} 未进入本轮持久化结果", [evidence_id]))

    if last_result_by_name and all(failed for _, failed in last_result_by_name.values()):
        issues.append(_issue("tools_failed", "critical", "本轮工具调用均未成功",
                             [item[0] for item in last_result_by_name.values()]))
    elif any(failed for _, failed in last_result_by_name.values()):
        issues.append(_issue("tool_failed", "warning", "部分工具最终调用失败",
                             [item[0] for item in last_result_by_name.values() if item[1]]))

    web_refs = [ref for ref in references if ref.get("source_type") == "web"]
    for ref in web_refs[:10]:
        evidence.append({"id": f"web:{ref['source_id']}", "kind": "web",
                         "title": ref.get("title"), "url": ref.get("url"),
                         "snippet": str(ref.get("snippet") or "")[:600],
                         "published_at": ref.get("published_at"),
                         "retrieved_at": ref.get("retrieved_at"),
                         "evidence_type": "page_body" if ref.get("body_verified") else ref.get("evidence_type", "search_snippet"),
                         "structured_data": ref.get("structured_data"),
                         "coverage": ref.get("coverage"), "truncated": ref.get("truncated"),
                         "body_verified": ref.get("body_verified", False),
                         "artifact_id": ref.get("artifact_id"),
                         "text_sha256": ref.get("text_sha256"), "text_chars": ref.get("text_chars"),
                         "body_excerpt": str(ref.get("body_excerpt") or "")[:3000],
                         "final_url": ref.get("final_url")})
    if LIVE_SEARCH_RE.search(question) and not web_refs:
        issues.append(_issue("live_source_missing", "critical", "实时问题没有可核验的联网来源"))
    verified_bodies = [ref for ref in web_refs if (
        ref.get("body_verified") and ref.get("artifact_id") and ref.get("body_excerpt")
    )]
    official_records = [ref for ref in web_refs if ref.get("evidence_type") == "official_record" and ref.get("structured_data")]
    # Official API fields support alert/event facts without an HTML page body.
    # Explicit requests to open/read a linked document retain the P0 body gate.
    if requires_web_body(question) and not verified_bodies and (WEB_FETCH_RE.search(question) or not official_records):
        issues.append(_issue("web_body_missing", "critical", "链接核实或预警通报缺少已抓取并保存的网页正文，搜索摘要不足以核实"))
    elif verified_bodies:
        def canonical_url(value):
            url = _public_url(value)
            if not url:
                return None
            parsed = urlsplit(url)
            return urlunsplit((parsed.scheme, parsed.netloc.lower(), parsed.path or "/", parsed.query, ""))

        requested = {canonical_url(url.rstrip(".,;!?)]}，。；！？"))
                     for url in re.findall(r"https?://[^\s<>`\"'，。；！？]+", question)} - {None}
        fetched = {canonical_url(ref.get(key)) for ref in verified_bodies
                   for key in ("url", "final_url")} - {None}
        if requested and not requested.intersection(fetched):
            issues.append(_issue("requested_web_body_missing", "critical", "已抓取的正文不对应用户要求核实的链接"))

    for index, value in enumerate(output_paths[:MAX_ARTIFACTS]):
        path = Path(value)
        evidence_id = f"artifact:{index}"
        item: dict[str, Any] = {"id": evidence_id, "kind": "artifact", "name": path.name}
        size = _file_size(path)
        if not path.is_file():
            issues.append(_issue("artifact_missing", "critical", f"产物 {path.name} 不存在", [evidence_id]))
        elif size == 0:
            issues.append(_issue("artifact_empty", "critical", f"产物 {path.name} 为空", [evidence_id]))
        else:
            item["size_bytes"] = size
            if path.suffix.lower() in {".tif", ".tiff"}:
                try:
                    item["raster"] = _raster_metadata(path)
                    if not item["raster"]["crs"]:
                        issues.append(_issue("raster_no_crs", "warning", f"栅格 {path.name} 没有坐标系", [evidence_id]))
                except Exception:
                    issues.append(_issue("raster_unreadable", "critical", f"栅格 {path.name} 无法读取", [evidence_id]))
        evidence.append(item)
    if len(output_paths) > MAX_ARTIFACTS:
        issues.append(_issue("artifact_review_limit", "warning", "本轮产物过多，仅检查前 20 个"))
    return evidence, issues


def review_required(question: str, response: dict, uploads: list[str], outputs: list[str]) -> bool:
    if uploads or outputs or LIVE_SEARCH_RE.search(question):
        return True
    return any(isinstance(message, ToolMessage) for message in response.get("messages", []))


def audit_evidence(evidence: list[dict]) -> list[dict]:
    """Persist a small evidence snapshot without machine-local paths or raw files."""
    snapshot = []
    for item in evidence[:40]:
        kind = item.get("kind")
        record = {key: item[key] for key in ("id", "kind", "name", "status", "title",
                                               "url", "snippet", "published_at", "retrieved_at",
                                               "size_bytes", "raster", "final_url", "evidence_type",
                                               "body_verified", "artifact_id", "text_sha256", "text_chars",
                                               "structured_data", "coverage", "truncated") if key in item}
        if kind == "web" and item.get("body_excerpt"):
            record["body_excerpt"] = str(item["body_excerpt"])[:1200]
        if kind == "tool":
            try:
                payload = json.loads(item.get("result", ""))
            except (TypeError, ValueError):
                payload = None
            if isinstance(payload, dict):
                if item.get("name") in WEB_SOURCE_TOOL_NAMES:
                    results = payload.get("results") if isinstance(payload.get("results"), list) else []
                    record["results"] = [{key: result[key] for key in
                                          ("title", "url", "snippet", "published_at", "retrieved_at",
                                           "final_url", "text_chars", "text_sha256", "event_id", "data")
                                          if key in result}
                                         for result in results[:5]
                                         if isinstance(result, dict)]
                else:
                    record["metrics"] = {
                        key: value for key, value in list(payload.items())[:40]
                        if not key.endswith("_path") and (
                            (isinstance(value, (int, float, bool))
                             and (not isinstance(value, float) or math.isfinite(value)))
                            or (key in {"count_unit", "model_name", "damage_level",
                                        "flood_level", "burned_level"}
                                and isinstance(value, str) and len(value) <= 80)
                        )
                    }
        snapshot.append(record)
    return snapshot


def parse_review_response(text: str, evidence_ids: set[str]) -> dict:
    """Accept only a complete JSON object with known evidence references."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I).strip()
    data = json.loads(cleaned)
    if not isinstance(data, dict) or data.get("status") not in {"passed", "revised", "blocked"}:
        raise ValueError("review status is invalid")
    issues = data.get("issues")
    artifacts = data.get("artifacts")
    if not isinstance(issues, list) or not isinstance(artifacts, dict):
        raise ValueError("review issues/artifacts are invalid")
    parsed_issues = []
    for issue in issues[:12]:
        if not isinstance(issue, dict) or issue.get("severity") not in {"warning", "critical"}:
            raise ValueError("review issue is invalid")
        refs = issue.get("evidence_ids", [])
        if not isinstance(refs, list) or any(ref not in evidence_ids for ref in refs):
            raise ValueError("review cites unknown evidence")
        message = issue.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("review issue has no message")
        parsed_issues.append(_issue(str(issue.get("code") or "review_issue")[:80],
                                    issue["severity"], message.strip()[:500], refs[:8]))
    display = artifacts.get("display")
    download = artifacts.get("download")
    if not isinstance(display, list) or not isinstance(download, list) or any(
        not isinstance(path, str) for path in [*display, *download]
    ):
        raise ValueError("review artifact selection is invalid")
    status = data["status"]
    revised = data.get("revised_answer", "")
    if status == "revised" and (not isinstance(revised, str) or not revised.strip()
                                 or not any(issue["evidence_ids"] for issue in parsed_issues)):
        raise ValueError("revision requires an answer and cited issue")
    if status == "blocked" and not any(issue["severity"] == "critical" for issue in parsed_issues):
        raise ValueError("blocked review requires a critical issue")
    if status == "passed" and any(issue["severity"] == "critical" for issue in parsed_issues):
        raise ValueError("passed review contains a critical issue")
    supplement = data.get("supplement", "")
    return {"status": status, "issues": parsed_issues,
            "revised_answer": revised.strip()[:12000] if isinstance(revised, str) else "",
            "supplement": supplement.strip()[:2000] if isinstance(supplement, str) else "",
            "artifacts": {"display": display[:MAX_ARTIFACTS], "download": download[:MAX_ARTIFACTS]}}
