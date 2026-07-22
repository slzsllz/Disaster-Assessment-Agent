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

OILSPILL_TOOL_DESCRIPTION = """
Extract marine oil spill areas from a 3-channel image using the packaged CBDNet semantic segmentation model.

Parameters:
- image_path (str): Path to a 3-channel .jpg/.png image. SOS samples are 256x256 RGB-like rendered images.
- output_dir (str): Relative output directory under the tool temp directory.
- weights_path (str | None): Optional CBDNet weights path. If omitted, the bundled cbdnet_best.th is used.
- device (str): Inference device, e.g. "cuda:0" or "cpu". CUDA automatically falls back to CPU inside the model wrapper.
- threshold (float): Segmentation probability threshold. Default is 0.5.

Returns:
- dict: Oil-spill pixel count, area ratio, mask path, visualization path, reliability metadata, and summary path.

Answer guidance:
- Return all generated output paths in the tool result, but do not assume every output is useful to the user.
- The second-pass multimodal reviewer will inspect the tool outputs and choose which images are useful for frontend display and which files are useful downloads.
- The first-pass answer should focus on the oil-spill segmentation result, not on listing or explaining output files.
- Do not repeat or list output file paths in the final natural-language answer.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--temp_dir", type=str)
parser.add_argument("--image", type=str, default=None)
parser.add_argument("--output", type=str, default="oilspill_segmentation")
parser.add_argument("--weights", type=str, default=None)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--threshold", type=float, default=0.5)
args, unknown = parser.parse_known_args()

TEMP_DIR = Path(args.temp_dir or "tmp/tmp/out")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "model" / "oilspill_cbdnet"


def _load_predict():
    import sys

    model_root = str(MODEL_ROOT)
    if model_root not in sys.path:
        sys.path.insert(0, model_root)
    spec = importlib.util.spec_from_file_location("oilspill_cbdnet_infer", MODEL_ROOT / "infer.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load oil-spill infer.py from {MODEL_ROOT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.predict


def extract_oil_spill_area(
    image_path: str,
    output_dir: str = "oilspill_segmentation",
    weights_path: Optional[str] = None,
    device: str = "cuda:0",
    threshold: float = 0.5,
) -> dict:
    out_dir = TEMP_DIR / output_dir
    predict = _load_predict()
    result = predict(
        {"image": image_path},
        weights_path=weights_path,
        output_dir=str(out_dir),
        device=device,
        threshold=threshold,
    )
    result["task"] = "oilspill_segmentation"

    summary_path = out_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    result["summary_path"] = str(summary_path)

    try:
        from tools.utils import save_assessment_to_db

        save_assessment_to_db("oil_spill", result, raster_path=image_path)
    except Exception:
        pass

    return result


if mcp is not None:
    mcp.tool(description=OILSPILL_TOOL_DESCRIPTION)(extract_oil_spill_area)


if __name__ == "__main__":
    if args.image:
        print(json.dumps(
            extract_oil_spill_area(
                image_path=args.image,
                output_dir=args.output,
                weights_path=args.weights,
                device=args.device,
                threshold=args.threshold,
            ),
            indent=2,
            ensure_ascii=False,
        ))
    else:
        if mcp is None:
            raise FASTMCP_IMPORT_ERROR
        mcp.run()
