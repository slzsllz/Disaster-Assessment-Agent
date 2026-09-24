"""
Cross-task error memory: fixes learned on earlier tasks are reused on later tasks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Protocol


class ErrorMemoryStore(Protocol):
    def load_all_errors(self) -> dict[str, str]: ...
    def save_error(self, pattern: str, fix: str) -> bool: ...
    def increment_error_hit(self, pattern: str) -> bool: ...


DEFAULT_MEMORY = {
    "has no attribute 'read'": (
        "A raster-like object may already be a NumPy array, not a rasterio "
        "DatasetReader. If a tool needs band data, open file paths with "
        "rasterio.open(path).read(...) and avoid calling .read() on arrays."
    ),
    "has no attribute 'crs'": (
        "A Shapely geometry has no CRS by itself. Wrap it in "
        "gpd.GeoDataFrame(geometry=[geom], crs=source_gdf.crs) before "
        "geospatial operations that require CRS metadata."
    ),
    "geoplot": (
        "Use a non-interactive matplotlib backend such as Agg. For projection "
        "errors, prefer gcrs.PlateCarree() before trying AlbersEqualArea()."
    ),
    "MemoryError": (
        "Reduce data volume before expensive vector/raster operations. For "
        "large GeoDataFrames, sample or tile the data, e.g. "
        "gdf_sample = gdf.sample(n=10000, random_state=42)."
    ),
    "ForwardCompatibility": (
        "CUDA/NVIDIA driver is likely too old for the installed package. Use "
        "CPU inference or install a package build compatible with the driver."
    ),
    "GLIBC_2.33": (
        "A binary wheel requires a newer system glibc. Prefer a conda-forge "
        "build or pin an older compatible wheel."
    ),
    "Connection refused": (
        "This is a network/API connectivity problem, not a tool bug. Check "
        "base_url, /v1 suffix, proxy variables, and whether the server can "
        "reach the model provider."
    ),
    "No module named 'torch'": (
        "The deep model environment is missing PyTorch. Install a CPU/GPU "
        "build matching the machine before running model inference tools."
    ),
}


class ErrorMemory:
    """Small persistent map from error patterns to repair suggestions."""

    def __init__(
        self,
        path: str | Path | None = None,
        db: ErrorMemoryStore | None = None,
    ):
        self.path = Path(path) if path else None
        self.db = db
        self._lock = threading.RLock()
        self._last_refresh = time.monotonic()
        self._memory = dict(DEFAULT_MEMORY)
        self._load()

    def _load(self) -> None:
        if self.path is not None and self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            if isinstance(data, dict):
                for pattern, fix in data.items():
                    if isinstance(pattern, str) and isinstance(fix, str):
                        self._memory[pattern] = fix

        if self.db is not None:
            stored = self.db.load_all_errors()
            for pattern, fix in stored.items():
                if isinstance(pattern, str) and isinstance(fix, str):
                    self._memory[pattern] = fix
            for pattern, fix in self._memory.items():
                if pattern not in stored:
                    self.db.save_error(pattern, fix)

    def save(self) -> None:
        """Persist the current error memory to a JSON file."""
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent,
                prefix=f".{self.path.name}.", delete=False,
            ) as output:
                temporary_path = Path(output.name)
                try:
                    json.dump(self._memory, output, ensure_ascii=False, indent=2)
                    output.flush()
                    os.fsync(output.fileno())
                except BaseException:
                    temporary_path.unlink(missing_ok=True)
                    raise
            try:
                os.replace(temporary_path, self.path)
            except OSError:
                temporary_path.unlink(missing_ok=True)
                raise

    def refresh(self, interval_seconds: float = 60) -> None:
        """Merge rules written by other backend workers without restarting."""
        if self.db is None or time.monotonic() - self._last_refresh < interval_seconds:
            return
        with self._lock:
            if time.monotonic() - self._last_refresh < interval_seconds:
                return
            self._last_refresh = time.monotonic()
            self._memory.update(self.db.load_all_errors())

    def lookup(self, error_msg: str) -> str:
        for pattern, fix in self.lookup_all(error_msg, limit=1):
            return fix
        return ""

    def lookup_all(self, error_msg: str, limit: int = 5) -> list[tuple[str, str]]:
        if limit <= 0:
            return []
        self.refresh()
        text = (error_msg or "").lower()
        with self._lock:
            matches = [
                (pattern, fix) for pattern, fix in self._memory.items()
                if pattern.lower() in text
            ]
        matches.sort(key=lambda item: len(item[0]), reverse=True)
        selected = matches[:limit]
        if self.db is not None:
            for pattern, _ in selected:
                self.db.increment_error_hit(pattern)
        return selected

    def record(self, error_pattern: str, fix_suggestion: str) -> bool:
        key = error_pattern.strip()
        fix = fix_suggestion.strip()
        if len(key) <= 3 or len(fix) <= 3:
            return False
        with self._lock:
            # A temporarily unavailable DB must not prevent the file fallback.
            if self.db is not None:
                self.db.save_error(key, fix)
            self._memory[key] = fix
            self.save()
        return True

    def format_prompt_block(self, limit: int = 12) -> str:
        if not self._memory:
            return ""
        lines = [
            "\n\nKnown error memory - when a tool call fails, compare the error "
            "message with these patterns, apply the suggested fix, and retry "
            "with corrected arguments or a safer workflow before giving up:"
        ]
        for idx, (pattern, fix) in enumerate(self._memory.items()):
            if idx >= limit:
                break
            lines.append(f"  - If error contains `{pattern}`: {fix}")
        return "\n".join(lines)

    def get_all(self) -> dict:
        return dict(self._memory)

    def __len__(self):
        return len(self._memory)
