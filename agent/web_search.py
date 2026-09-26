"""Bounded, source-attributed internet search for the agent."""

from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit

import requests


MAX_RESPONSE_BYTES = 512 * 1024
MAX_QUERY_CHARS = 300
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
NEWS_URL = "https://news.google.com/rss/search"


def _http_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            return None
    except ValueError:
        return None
    return value


def _public_url(value: object) -> str | None:
    url = _http_url(value)
    if not url:
        return None
    host = urlsplit(url).hostname or ""
    if host.lower() in {"localhost", "localhost.localdomain"} or host.lower().endswith(".local"):
        return None
    try:
        if not ipaddress.ip_address(host).is_global:
            return None
    except ValueError:
        pass
    return url


def _plain_text(value: object, limit: int = 600) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value))).strip()[:limit]


def _get(url: str, *, params: dict, headers: dict | None = None) -> bytes:
    timeout = min(max(float(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "10")), 2), 20)
    proxy = os.getenv("WEB_SEARCH_PROXY", "").strip()
    proxies = {"http": proxy, "https": proxy} if proxy else None
    with requests.get(url, params=params, headers=headers, timeout=timeout,
                      proxies=proxies, stream=True) as response:
        response.raise_for_status()
        body = bytearray()
        for chunk in response.iter_content(8192):
            body.extend(chunk)
            if len(body) > MAX_RESPONSE_BYTES:
                raise ValueError("Search provider response is too large")
        return bytes(body)


def _brave(query: str, count: int) -> list[dict]:
    key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
    if not key:
        raise ValueError("BRAVE_SEARCH_API_KEY is required for Brave search")
    data = json.loads(_get(
        BRAVE_URL,
        params={"q": query, "count": count},
        headers={"Accept": "application/json", "X-Subscription-Token": key},
    ))
    return [
        {
            "title": _plain_text(item.get("title"), 200),
            "url": item.get("url"),
            "snippet": _plain_text(item.get("description")),
            "source": "Brave Search",
        }
        for item in data.get("web", {}).get("results", [])
        if isinstance(item, dict)
    ]


def _searxng(query: str, count: int) -> list[dict]:
    base = os.getenv("SEARXNG_URL", "").strip().rstrip("/")
    if not _http_url(base):
        raise ValueError("SEARXNG_URL must be an HTTP(S) URL")
    endpoint = base if base.endswith("/search") else f"{base}/search"
    data = json.loads(_get(endpoint, params={"q": query, "format": "json"}))
    return [
        {
            "title": _plain_text(item.get("title"), 200),
            "url": item.get("url"),
            "snippet": _plain_text(item.get("content")),
            "source": _plain_text(item.get("engine") or "SearXNG", 80),
        }
        for item in data.get("results", [])[:count]
        if isinstance(item, dict)
    ]


def _news(query: str, count: int) -> list[dict]:
    root = ET.fromstring(_get(
        NEWS_URL, params={"q": query, "hl": "zh-CN", "gl": "CN", "ceid": "CN:zh-Hans"}
    ))
    results = []
    for item in root.findall("./channel/item")[:count]:
        published = item.findtext("pubDate")
        try:
            published_at = parsedate_to_datetime(published).astimezone(timezone.utc).isoformat() if published else None
        except (TypeError, ValueError, OverflowError):
            published_at = None
        source = item.find("source")
        results.append({
            "title": _plain_text(item.findtext("title"), 200),
            "url": item.findtext("link"),
            "snippet": _plain_text(item.findtext("description")),
            "source": _plain_text(source.text if source is not None else "Google News", 80),
            "published_at": published_at,
        })
    return results


def search_web(query: str, count: int = 5) -> dict:
    """Search live sources and return bounded snippets with URLs and fetch time.

    Auto mode uses Brave or SearXNG when configured. Its keyless fallback is
    Google News RSS, which covers news only, not the entire web.
    """
    query = (query or "").strip()
    if not query or len(query) > MAX_QUERY_CHARS:
        return {"success": False, "error": f"query must be 1-{MAX_QUERY_CHARS} characters"}
    count = max(1, min(int(count), 10))
    configured = os.getenv("WEB_SEARCH_PROVIDER", "auto").strip().lower()
    if configured == "auto":
        provider = "brave" if os.getenv("BRAVE_SEARCH_API_KEY") else "searxng" if os.getenv("SEARXNG_URL") else "news"
    elif configured in {"brave", "searxng", "news"}:
        provider = configured
    else:
        return {"success": False, "error": "WEB_SEARCH_PROVIDER must be auto, brave, searxng, or news"}
    try:
        raw = {"brave": _brave, "searxng": _searxng, "news": _news}[provider](query, count)
    except (requests.RequestException, ValueError, ET.ParseError, KeyError, TypeError) as exc:
        return {"success": False, "provider": provider, "error": f"Search unavailable: {type(exc).__name__}"}

    results = []
    seen = set()
    for item in raw:
        url = _public_url(item.get("url"))
        if not url or url in seen:
            continue
        seen.add(url)
        results.append({**item, "url": url})
        if len(results) >= count:
            break
    return {
        "success": True,
        "query": query,
        "provider": provider,
        "scope": "news" if provider == "news" else "web",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }


def web_references(messages: list) -> list[dict]:
    """Extract cited search hits from successful web_search tool messages."""
    from langchain_core.messages import AIMessage, ToolMessage

    names = {
        call.get("id"): call.get("name")
        for message in messages if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    references = []
    seen = set()
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if (message.name or names.get(message.tool_call_id)) != "web_search" or message.status == "error":
            continue
        try:
            payload = message.content
            if isinstance(payload, list):
                payload = next((part.get("text") for part in payload if isinstance(part, dict) and part.get("type") == "text"), "")
            payload = json.loads(payload) if isinstance(payload, str) else payload
            if not isinstance(payload, dict) or payload.get("success") is not True:
                continue
            for item in payload.get("results", []):
                url = _public_url(item.get("url")) if isinstance(item, dict) else None
                if not url or url in seen:
                    continue
                seen.add(url)
                references.append({
                    "source_type": "web",
                    "source_id": hashlib.sha256(url.encode()).hexdigest()[:24],
                    "title": _plain_text(item.get("title"), 200) or url,
                    "url": url,
                    "provider": payload.get("provider"),
                    "published_at": item.get("published_at"),
                    "retrieved_at": payload.get("retrieved_at"),
                })
        except (TypeError, ValueError):
            continue
    return references
