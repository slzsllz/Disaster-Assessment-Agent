"""
Cross-task error memory: fixes learned on earlier tasks are reused on later tasks.
"""

from __future__ import annotations

import json
from pathlib import Path


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

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._memory = dict(DEFAULT_MEMORY)
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        if isinstance(data, dict):
            for pattern, fix in data.items():
                if isinstance(pattern, str) and isinstance(fix, str):
                    self._memory[pattern] = fix

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._memory, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def lookup(self, error_msg: str) -> str:
        for pattern, fix in self.lookup_all(error_msg, limit=1):
            return fix
        return ""

    def lookup_all(self, error_msg: str, limit: int = 5) -> list[tuple[str, str]]:
        text = (error_msg or "").lower()
        matches: list[tuple[str, str]] = []
        for pattern, fix in self._memory.items():
            if pattern.lower() in text:
                matches.append((pattern, fix))
            if len(matches) >= limit:
                break
        return matches

    def record(self, error_pattern: str, fix_suggestion: str) -> bool:
        key = error_pattern.strip()
        fix = fix_suggestion.strip()
        if len(key) <= 3 or len(fix) <= 3:
            return False
        self._memory[key] = fix
        self.save()
        return True

    def format_prompt_block(self, limit: int = 12) -> str:
        if not self._memory:
            return ""
        lines = [
            "\n\nKnown error memory — when a tool call fails, compare the error "
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
