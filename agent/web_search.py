"""Bounded, source-attributed internet search for the agent."""

from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import os
import re
import socket
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests


MAX_RESPONSE_BYTES = 512 * 1024
MAX_QUERY_CHARS = 300
WEB_SOURCE_TOOL_NAMES = frozenset({"web_search", "web_fetch", "disaster_feed"})
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
        parsed.port  # Reject malformed ports before a request is constructed.
    except ValueError:
        return None
    return value


def _public_url(value: object) -> str | None:
    url = _http_url(value)
    if not url:
        return None
    host = (urlsplit(url).hostname or "").rstrip(".")
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


class _HostnameHTTPSAdapter(requests.adapters.HTTPAdapter):
    """Connect to a validated IP while retaining hostname TLS verification/SNI."""

    def __init__(self, hostname: str):
        self.hostname = hostname
        super().__init__()

    def init_poolmanager(self, connections, maxsize, block=False, **kwargs):
        super().init_poolmanager(connections, maxsize, block,
                                assert_hostname=self.hostname, server_hostname=self.hostname, **kwargs)

    def proxy_manager_for(self, proxy, **kwargs):
        return super().proxy_manager_for(proxy, assert_hostname=self.hostname,
                                         server_hostname=self.hostname, **kwargs)


def _public_address(url: str) -> str:
    """Reject private DNS answers and pin a public address against DNS rebinding."""
    if not _public_url(url):
        raise ValueError("Only public HTTP(S) URLs are allowed")
    parsed = urlsplit(url)
    answers = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80),
                                 type=socket.SOCK_STREAM)
    addresses = list(dict.fromkeys(answer[4][0] for answer in answers))
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError("URL resolves to a non-public address")
    return next((address for address in addresses if ":" not in address), addresses[0])


def _read_response(response, url: str, response_info: dict | None, deadline: float,
                   max_response_bytes: int | None = None) -> bytes:
    byte_limit = MAX_RESPONSE_BYTES if max_response_bytes is None else max_response_bytes
    response.raise_for_status()
    length = response.headers.get("Content-Length", "")
    if length.isdigit() and int(length) > byte_limit:
        raise ValueError("HTTP response is too large")
    body = bytearray()
    for chunk in response.iter_content(8192):
        if time.monotonic() > deadline:
            raise requests.Timeout("HTTP response deadline exceeded")
        body.extend(chunk)
        if len(body) > byte_limit:
            raise ValueError("HTTP response is too large")
    if response_info is not None:
        response_info.update(url=url, content_type=response.headers.get("Content-Type", ""))
    return bytes(body)


def _get(url: str, *, params: dict, headers: dict | None = None,
         public_only: bool = False, response_info: dict | None = None,
         max_response_bytes: int | None = None) -> bytes:
    """Bounded HTTP GET. Arbitrary page fetching must enable public_only.

    Search-provider configuration remains compatible with private SearXNG
    deployments. Public page fetching checks DNS and every redirect, connects
    to the checked IP, and verifies HTTPS against the original hostname.
    """
    timeout = min(max(float(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "10")), 2), 20)
    deadline = time.monotonic() + timeout
    proxy = os.getenv("WEB_SEARCH_PROXY", "").strip()
    proxies = {"http": proxy, "https": proxy} if proxy else None
    if public_only:
        current = url
        for hop in range(4):
            address = _public_address(current)
            parsed = urlsplit(current)
            hostname = (parsed.hostname or "").encode("idna").decode("ascii")
            address_host = f"[{address}]" if ":" in address else address
            authority = address_host + (f":{parsed.port}" if parsed.port else "")
            pinned = urlunsplit((parsed.scheme, authority, parsed.path, parsed.query, ""))
            host_header = (f"[{hostname}]" if ":" in hostname else hostname) + (f":{parsed.port}" if parsed.port else "")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise requests.Timeout("HTTP response deadline exceeded")
            with requests.Session() as session:
                # Explicitly inherit proxies, without inheriting netrc credentials.
                session.trust_env = False
                session.mount("https://", _HostnameHTTPSAdapter(hostname))
                with session.get(pinned, params=params if hop == 0 else None,
                                 headers={**(headers or {}), "Host": host_header},
                                 timeout=remaining, proxies=proxies or requests.utils.get_environ_proxies(current),
                                 stream=True, allow_redirects=False) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("Location")
                        if not location or hop == 3:
                            raise ValueError("Invalid or excessive redirects")
                        current = urljoin(current, location)
                        continue
                    return _read_response(response, current, response_info, deadline, max_response_bytes)
        raise ValueError("Excessive redirects")
    with requests.get(url, params=params, headers=headers, timeout=timeout,
                      proxies=proxies, stream=True) as response:
        return _read_response(response, response.url, response_info, deadline, max_response_bytes)


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


def search_result_references(payload: dict) -> list[dict]:
    """Create displayable references from one successful live search."""
    if not isinstance(payload, dict) or payload.get("success") is not True:
        return []
    references = []
    seen = set()
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
            "snippet": _plain_text(item.get("snippet")),
            "provider": payload.get("provider"),
            "published_at": item.get("published_at"),
            "retrieved_at": payload.get("retrieved_at"),
        })
    return references


def web_tool_payloads(messages: list, tool_names: set[str] | None = None):
    """Decode search/fetch MCP text blocks with tool-call name fallback."""
    from langchain_core.messages import AIMessage, ToolMessage

    names = {
        call.get("id"): call.get("name")
        for message in messages if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        name = message.name or names.get(message.tool_call_id)
        if name not in (tool_names if tool_names is not None else WEB_SOURCE_TOOL_NAMES) or message.status == "error":
            continue
        try:
            payload = message.content
            if isinstance(payload, list):
                payload = next((part.get("text") for part in payload if isinstance(part, dict) and part.get("type") == "text"), "")
            payload = json.loads(payload) if isinstance(payload, str) else payload
            if isinstance(payload, dict):
                yield name, payload
        except (TypeError, ValueError):
            continue


def fetch_result_references(payload: dict) -> list[dict]:
    """Page references carry artifact provenance; the backend verifies storage."""
    if not isinstance(payload, dict) or payload.get("success") is not True:
        return []
    references = search_result_references(payload)
    by_url = {item["url"]: item for item in payload.get("results", []) if isinstance(item, dict)
              and _public_url(item.get("url"))}
    for reference in references:
        item = by_url[reference["url"]]
        artifact = item.get("artifact") if isinstance(item.get("artifact"), dict) else {}
        reference.update({
            "provider": "web_fetch", "final_url": _public_url(item.get("final_url")),
            "retrieved_at": item.get("retrieved_at"), "artifact_id": artifact.get("id"),
            "text_chars": item.get("text_chars"), "text_sha256": item.get("text_sha256"),
        })
    return references


def web_references(messages: list) -> list[dict]:
    """Extract web citations, upgrading search hits when page evidence exists."""
    references = []
    by_url = {}
    for name, payload in web_tool_payloads(messages):
        if name == "disaster_feed":
            from agent.disaster_feeds import feed_result_references
            new = feed_result_references(payload)
        else:
            new = fetch_result_references(payload) if name == "web_fetch" else search_result_references(payload)
        for reference in new:
            previous = by_url.get(reference["url"])
            if previous:
                if reference.get("provider") == "web_fetch" or reference.get("structured_data"):
                    previous.update({key: value for key, value in reference.items() if value is not None})
                continue
            by_url[reference["url"]] = reference
            references.append(reference)
    return references
