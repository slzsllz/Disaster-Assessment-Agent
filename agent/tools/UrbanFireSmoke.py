"""MCP bridge to the verified CITY_FIRE_008A YOLOv5l CPU handoff package.

Only the standard library and optional FastMCP are imported in the Agent runtime.
The PyTorch model runs in its separately configured Python virtual environment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import uuid

try:
    from fastmcp import FastMCP
except ImportError:
    FastMCP = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "yolov5l"
MODEL_DIRECTORY = "urban_fire_smoke_yolov5l"
ORIGINAL_CHECKPOINT_SHA256 = "b9a052a445df37451a17b38da8b39f7a7356afdf244174d577d2ffcb7034b434"
DESCRIPTION = """
Detect suspected smoke/fire in an uploaded RGB JPEG/PNG image with the
pretrained CITY_FIRE_008A YOLOv5l detector, the only supported model.
Parameters: image_path is a real server-side path from an uploaded image;
conf is a confidence threshold (0.01..0.95). There is no model selection parameter.
Output: detections (0 smoke, 1 fire), original-image xyxy pixel boxes,
annotated_image_path, results_json_path, summary_path, and conservative summary.
These are possible visual fire/smoke detections, NOT verified fire incidents,
fire severity, geographic coordinates, evacuation advice, or fire-risk ratings.
The model is for RGB photos, not raw satellite multispectral imagery.
Don't invent GPS coordinates or infer a real incident from predictions alone.
""".strip()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--temp_dir", default="tmp/urban_fire_smoke")
parser.add_argument("--fire-python", default=None,
                    help="Absolute path to verified 008A Python environment's bin/python")
parser.add_argument("--l-model-dir", default=None, help="Verified extracted 008A l model folder")
parser.add_argument("--input-root", action="append", default=[],
                    help="Additional trusted input root configured by the server operator")
parser.add_argument("--image", default=None, help="Direct CLI smoke test input")
parser.add_argument("--conf", type=float, default=0.25)
ARGS, _UNKNOWN = parser.parse_known_args()
if any(arg.split("=", 1)[0] in {"--model", "--s-model-dir"} for arg in _UNKNOWN):
    parser.error("model selection is not supported; this tool uses YOLOv5l only")

mcp = FastMCP("UrbanFireSmoke") if FastMCP is not None else None


def _project_path(raw: str | Path) -> Path:
    p = Path(raw).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def _reject_symlinks(path: Path) -> None:
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("data and model paths must not contain symlink components")


def _input_image(raw: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("image_path must be a non-empty string")
    source = _project_path(raw)
    _reject_symlinks(source)
    if source.is_symlink() or not source.is_file():
        raise ValueError("image_path must be an existing regular non-symlink file")
    resolved = source.resolve(strict=True)
    allowed_roots = [
        PROJECT_ROOT / "tmp", PROJECT_ROOT / "user_img",
        PROJECT_ROOT / "benchmark", PROJECT_ROOT / "model",
        *(_project_path(x) for x in ARGS.input_root),
    ]
    if not any(resolved.is_relative_to(root.resolve()) for root in allowed_roots):
        raise ValueError("image_path is outside the configured input data roots")
    if resolved.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("only JPEG and PNG images are accepted")
    size = resolved.stat().st_size
    if not 0 < size <= 30 * 1024 * 1024:
        raise ValueError("image file must be between 1 byte and 30 MB")
    return resolved


def _model_dir() -> Path:
    path = _project_path(ARGS.l_model_dir) if ARGS.l_model_dir else PROJECT_ROOT / "model" / MODEL_DIRECTORY
    _reject_symlinks(path)
    if path.is_symlink() or not path.is_dir():
        raise FileNotFoundError(f"008A YOLOv5l model is not deployed at {path}")
    required = ("infer.py", "fixed_model.py", "infer_cpu.py", "weights/metadata.json",
                "weights/tensor_index.json", "weights/tensors_numeric_only.npz",
                "PACKAGE_MANIFEST.json")
    for item in required:
        file = path / item
        _reject_symlinks(file)
        if file.is_symlink() or not file.is_file():
            raise FileNotFoundError(f"missing 008A YOLOv5l file: {item}")
    meta = json.loads((path / "weights" / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("source_filename") != f"{MODEL_NAME}.pt" or meta.get("source_sha256") != ORIGINAL_CHECKPOINT_SHA256:
        raise ValueError("wrong 008A YOLOv5l checkpoint identity")
    return path.resolve(strict=True)


def _runtime_python() -> str:
    raw = ARGS.fire_python or os.environ.get("URBAN_FIRE_CPU_PYTHON")
    if not raw:
        local_config = PROJECT_ROOT / "model" / "urban_fire_runtime.json"
        if local_config.is_symlink():
            raise ValueError("urban_fire_runtime.json must not be a symlink")
        if local_config.is_file():
            config = json.loads(local_config.read_text(encoding="utf-8"))
            raw = config.get("python_executable") if isinstance(config, dict) else None
    if not raw or not isinstance(raw, str):
        raise RuntimeError(
            "008A runtime not configured: use --fire-python, URBAN_FIRE_CPU_PYTHON, "
            "or a local model/urban_fire_runtime.json with python_executable"
        )
    path = _project_path(raw)
    if not path.is_absolute() or not path.is_file():
        raise FileNotFoundError("the configured 008A Python executable does not exist")
    return str(path)


def _output_dir() -> Path:
    base = _project_path(ARGS.temp_dir)
    _reject_symlinks(base)
    if base.is_symlink():
        raise ValueError("tool output parent must not be a symlink")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"urban_fire_{uuid.uuid4().hex}"


def detect_urban_fire_smoke(image_path: str, conf: float = 0.25) -> dict:
    """Run 008A CPU inference and return user-displayable paths plus detections."""
    if isinstance(conf, bool) or not isinstance(conf, (float, int)) or not 0.01 <= conf <= 0.95:
        raise ValueError("conf must be a finite number in [0.01, 0.95]")
    image = _input_image(image_path)
    package = _model_dir()
    python_bin = _runtime_python()
    out = _output_dir()
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "PYTHONNOUSERSITE": "1"})
    cmd = [python_bin, "-I", "-B", str(package / "infer.py"),
           "--image", str(image), "--output", str(out), "--conf", str(float(conf)),
           "--iou", "0.45"]
    try:
        p = subprocess.run(cmd, cwd=str(package), env=env,
                           capture_output=True, text=True, timeout=180, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("008A CPU inference timed out after 180 seconds") from exc
    if p.returncode != 0:
        raise RuntimeError(f"008A inference failed (exit {p.returncode}): {(p.stderr or p.stdout)[-600:]}")
    summary_file = out / "summary.json"
    if summary_file.is_symlink() or not summary_file.is_file() or summary_file.stat().st_size > 2_000_000:
        raise RuntimeError("008A inference succeeded but summary.json is missing or invalid")
    result = json.loads(summary_file.read_text(encoding="utf-8"))
    if (not isinstance(result, dict) or result.get("status") != "success"
            or result.get("model") != MODEL_NAME
            or result.get("summary", {}).get("fire_confirmed") is not False):
        raise RuntimeError("008A returned an unexpected result schema")
    for key, name in (("annotated_image_path", "annotated.jpg"),
                      ("results_json_path", "results.json"),
                      ("summary_path", "summary.json")):
        f = out / name
        if f.is_symlink() or not f.is_file() or Path(result.get(key, "")) != f:
            raise RuntimeError(f"008A output {key} is absent or inconsistent")
    return result


if mcp is not None:
    mcp.tool(description=DESCRIPTION)(detect_urban_fire_smoke)


if __name__ == "__main__":
    if ARGS.image:
        try:
            value = detect_urban_fire_smoke(ARGS.image, conf=ARGS.conf)
            print(json.dumps(value, ensure_ascii=False, indent=2))
        except (ValueError, OSError, RuntimeError) as exc:
            print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False))
            raise SystemExit(2)
    else:
        if mcp is None:
            raise RuntimeError("MCP server requires FastMCP in the Agent environment")
        mcp.run()
