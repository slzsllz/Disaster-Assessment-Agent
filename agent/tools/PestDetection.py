import argparse
import importlib.util
import json
from pathlib import Path
from typing import Optional

try:
    from fastmcp import FastMCP
except Exception as exc:  # noqa: BLE001
    FastMCP = None
    FASTMCP_IMPORT_ERROR = exc
else:
    FASTMCP_IMPORT_ERROR = None


mcp = FastMCP() if FastMCP is not None else None

PEST_TOOL_DESCRIPTION = """
Detect pest-affected crop plants/regions from a single RGB image using the packaged Ultralytics YOLOv8 detector.
The detected boxes represent affected plants or affected crop regions, not individual insects.

Parameters:
- image_path (str): Path to an RGB .jpg/.png image.
- output_dir (str): Relative output directory under the tool temp directory.
- weights_path (str | None): Optional YOLO weights path. If omitted, the bundled best.pt is used.
- device (str): Inference device, e.g. "cuda:0" or "cpu". CUDA falls back to CPU when unavailable.
- conf (float): Detection confidence threshold. Default is 0.25.

Returns:
- dict: Bounding boxes for pest-affected plants/regions, confidence scores, affected target count, box area ratio, visualization path, reliability metadata, and summary path.

Interpretation guidance:
- outputs.detection_count is the number of detected affected plants/regions, not the number of insects.
- outputs.box_area_ratio is the summed affected-region bounding-box area divided by image area; boxes can overlap and this is not an exact damaged-area ratio.

Answer guidance:
- Return all generated output paths in the tool result, but do not assume every output is useful to the user.
- The language model should inspect the tool result and choose which images are useful for frontend display and which files are useful downloads via the final <Artifacts> block.
- For each selected downloadable file, explain what it contains and how the user can use it, e.g. GIS overlay, quantitative summary, or downstream verification.
- Do not repeat or list output file paths in the final natural-language answer.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--temp_dir", type=str)
parser.add_argument("--image", type=str, default=None)
parser.add_argument("--output", type=str, default="pest_detection")
parser.add_argument("--weights", type=str, default=None)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--conf", type=float, default=0.25)
args, unknown = parser.parse_known_args()

TEMP_DIR = Path(args.temp_dir or "tmp/tmp/out")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "model" / "pest_yolo"


def _load_predict():
    import sys

    model_root = str(MODEL_ROOT)
    if model_root not in sys.path:
        sys.path.insert(0, model_root)
    spec = importlib.util.spec_from_file_location("pest_yolo_infer", MODEL_ROOT / "infer.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load pest infer.py from {MODEL_ROOT}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        if exc.name == "ultralytics":
            raise ModuleNotFoundError(
                "PestDetection requires ultralytics. Install the model dependencies with: "
                "pip install -r model/pest_yolo/requirements.txt"
            ) from exc
        raise
    return module.predict


def _resolve_device_name(device: str) -> str:
    if device.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
        except Exception:
            return "cpu"
    return device


def detect_crop_pests(
    image_path: str,
    output_dir: str = "pest_detection",
    weights_path: Optional[str] = None,
    device: str = "cuda:0",
    conf: float = 0.25,
) -> dict:
    out_dir = TEMP_DIR / output_dir
    predict = _load_predict()
    result = predict(
        {"image": image_path},
        weights_path=weights_path,
        output_dir=str(out_dir),
        device=_resolve_device_name(device),
        conf=conf,
    )
    result["task"] = "pest_detection"
    result["interpretation"] = {
        "target_meaning": "Detected boxes represent pest-affected plants or affected crop regions, not individual insects.",
        "detection_count_meaning": "Number of detected affected plants/regions; do not describe it as insect count.",
        "box_area_ratio_meaning": "Summed affected-region bounding-box area divided by image area; boxes can overlap and it is not an exact damaged-area ratio.",
    }

    summary_path = out_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    result["summary_path"] = str(summary_path)

    try:
        from tools.utils import save_assessment_to_db

        save_assessment_to_db("pest", result, raster_path=image_path)
    except Exception:
        pass

    return result


if mcp is not None:
    mcp.tool(description=PEST_TOOL_DESCRIPTION)(detect_crop_pests)


if __name__ == "__main__":
    if args.image:
        print(json.dumps(
            detect_crop_pests(
                image_path=args.image,
                output_dir=args.output,
                weights_path=args.weights,
                device=args.device,
                conf=args.conf,
            ),
            indent=2,
            ensure_ascii=False,
        ))
    else:
        if mcp is None:
            raise FASTMCP_IMPORT_ERROR
        mcp.run()
