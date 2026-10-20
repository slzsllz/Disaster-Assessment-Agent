"""Conservative semantic retrieval, separate from document RAG and exact matches."""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
import time
from collections import OrderedDict

from agent.error_memory import normalize_error


logger = logging.getLogger(__name__)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _error_text(value: str) -> str:
    # Preserve the exception at the end of long tracebacks, not just stack frames.
    return normalize_error(value, limit=None)[-2000:]


def _signature(text: str) -> str:
    """Keep incompatible known causes out of the similarity ranking."""
    text = text.casefold()
    module = re.search(r"no module named\s+['\"]([^'\"]+)", text)
    if module is None:
        module = re.search(r"(?:required )?module\s+['\"]?([\w.]+)['\"]?\s+(?:is )?(?:unavailable|missing|not installed)", text)
    if module:
        return "module:" + module[1]
    attribute = re.search(r"(?:has no attribute|missing attribute)\s+['\"]([^'\"]+)", text)
    if attribute:
        return "attribute:" + attribute[1]
    if re.search(r"invalid device ordinal|invalid device id|设备序号无效", text):
        return "gpu_device"
    if re.search(r"illegal memory access|device-side assert|kernel launch", text):
        return "gpu_kernel"
    if re.search(r"(?:cuda|gpu|显存).*(?:out of memory|memory allocation|memory.*exhaust|不足|耗尽)", text):
        return "gpu_memory"
    for name, pattern in (
        ("memory", r"memoryerror|cannot allocate memory|内存不足"),
        ("missing_file", r"filenotfounderror|no such file|file .*not found|文件不存在|找不到文件"),
        ("permission", r"permissionerror|permission denied|access denied|权限不足"),
        ("connection_refused", r"connection refused|连接被拒绝"),
        ("timeout", r"timeout|timed out|超时"),
        ("authentication", r"unauthorized|invalid (?:api )?(?:key|token)|无效.*密钥"),
        ("glibc", r"glibc"),
    ):
        if re.search(pattern, text):
            return name
    exceptions = re.findall(r"\b([a-z]+error)\b", text)
    return "exception:" + exceptions[-1] if exceptions else ""


class ErrorVectorIndex:
    """On-demand index with profile-aware persistence and bounded query caching.

    Only sanitized error descriptions are embedded. Returned fixes always come
    from the current ErrorMemory snapshot, never from stale vector rows.
    """

    def __init__(
        self, embedder, *, database=None, min_score: float = 0.52,
        ambiguity_margin: float = 0.08, retry_seconds: float = 60,
    ):
        if not 0 <= min_score <= 1 or not 0 <= ambiguity_margin <= 1:
            raise ValueError("Error memory similarity thresholds must be between 0 and 1")
        self.embedder = embedder
        self.database = database
        self.profile = embedder.settings.profile
        self.dimensions = embedder.settings.dimensions
        self.min_score = min_score
        self.ambiguity_margin = ambiguity_margin
        self.retry_seconds = retry_seconds
        self._lock = threading.RLock()
        self._vectors = {}
        self._pending = set()
        self._queries = OrderedDict()
        self._schema_ready = False
        self._db_loaded = False
        self._next_db_try = 0.0
        self._next_embedding_try = 0.0

    def _vector(self, values) -> list[float]:
        vector = [float(v) for v in values]
        if len(vector) != self.dimensions or not all(map(math.isfinite, vector)):
            raise ValueError("Invalid error embedding")
        norm = math.sqrt(sum(v * v for v in vector))
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("Invalid error embedding norm")
        return [v / norm for v in vector]

    def _sync(self) -> None:
        if self.database is None or time.monotonic() < self._next_db_try:
            return
        try:
            if not self._schema_ready:
                self._schema_ready = self.database.ensure_error_vector_schema()
                if not self._schema_ready:
                    raise RuntimeError("Vector database unavailable")
            if not self._db_loaded:
                rows = self.database.load_error_vectors(self.profile)
                if rows is None:
                    raise RuntimeError("Vector database unavailable")
                for row in rows:
                    if row["entry_id"] in self._pending:
                        continue
                    try:
                        vector = self._vector(row["embedding"])
                    except (TypeError, ValueError, OverflowError):
                        continue
                    self._vectors[row["entry_id"]] = (row["content_hash"], vector)
                self._db_loaded = True
            for entry_id in list(self._pending):
                content_hash, vector = self._vectors[entry_id]
                if not self.database.save_error_vector(entry_id, self.profile, content_hash, vector):
                    raise RuntimeError("Vector database unavailable")
                self._pending.remove(entry_id)
        except Exception:
            self._next_db_try = time.monotonic() + self.retry_seconds
            logger.warning("Error vector persistence unavailable; retaining process cache")

    def index_entries(self, entries: dict[str, str]) -> bool:
        with self._lock:
            self._sync()
            missing = []
            for key, fix in entries.items():
                entry_id = _digest(key)
                pattern = key.partition("::")[2] if "::" in key else key
                text = _error_text(pattern)
                content_hash = _digest(text + "\0" + fix)
                cached = self._vectors.get(entry_id)
                if text and (cached is None or cached[0] != content_hash):
                    missing.append((entry_id, content_hash, text))
            if not missing:
                return True
            if time.monotonic() < self._next_embedding_try:
                return False
            try:
                for start in range(0, len(missing), 10):
                    batch = missing[start:start + 10]
                    vectors = self.embedder.embed([item[2] for item in batch])
                    if len(vectors) != len(batch):
                        raise ValueError("Invalid error embedding count")
                    vectors = [self._vector(vector) for vector in vectors]
                    for (entry_id, content_hash, _), vector in zip(batch, vectors):
                        self._vectors[entry_id] = (content_hash, vector)
                        self._pending.add(entry_id)
                self._sync()
                return True
            except Exception:
                self._next_embedding_try = time.monotonic() + self.retry_seconds
                logger.warning("Error embedding unavailable; semantic lookup paused temporarily")
                return False

    def lookup(self, error: str, entries: dict[str, str], *, tool_name=None) -> list[tuple[str, str]]:
        text = _error_text(error)
        if not text:
            return []
        signature = _signature(text)
        eligible = {}
        scoped_patterns = {
            _error_text(key.partition("::")[2]).casefold()
            for key in entries if "::" in key and key.partition("::")[0] == tool_name
        }
        for key, fix in entries.items():
            scope, separator, pattern = key.partition("::")
            if separator and scope != tool_name:
                continue
            pattern = pattern if separator else key
            pattern = _error_text(pattern)
            if not pattern or (not separator and pattern.casefold() in scoped_patterns):
                continue
            candidate_signature = _signature(pattern)
            if signature and candidate_signature and signature != candidate_signature:
                continue
            eligible[key] = fix
        if not eligible:
            return []
        with self._lock:
            if not self.index_entries(eligible):
                return []
            query_id = _digest(text)
            query = self._queries.get(query_id)
            if query is None:
                if time.monotonic() < self._next_embedding_try:
                    return []
                try:
                    vectors = self.embedder.embed([text], query=True)
                    if len(vectors) != 1:
                        raise ValueError("Invalid query embedding count")
                    query = self._vector(vectors[0])
                except Exception:
                    self._next_embedding_try = time.monotonic() + self.retry_seconds
                    logger.warning("Error query embedding unavailable; no semantic suggestion returned")
                    return []
                self._queries[query_id] = query
                if len(self._queries) > 256:
                    self._queries.popitem(last=False)
            self._queries.move_to_end(query_id)
            ranked = sorted(
                ((sum(a * b for a, b in zip(query, self._vectors[_digest(key)][1])), key, fix)
                 for key, fix in eligible.items()), reverse=True,
            )
            best_score, best_key, best_fix = ranked[0]
            if best_score < self.min_score:
                return []
            # Close alternatives prescribing different repairs are not a safe match.
            for score, _, fix in ranked[1:]:
                if fix != best_fix and best_score - score < self.ambiguity_margin:
                    return []
            logger.info("Semantic error experience matched (cosine=%.3f)", best_score)
            return [(best_key, best_fix)]
