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

LANDSLIDE_TOOL_DESCRIPTION = """
Extract landslide areas from a Landslide4Sense HDF5 image using the packaged U-Net semantic segmentation model.

Parameters:
- image_path (str): Path to a Landslide4Sense .h5 file. It must contain dataset "img" with shape 128x128x14.
- output_dir (str): Relative output directory under the tool temp directory.
- weights_path (str | None): Optional model weights path. If omitted, the bundled best.pth is used.
- device (str): Inference device, e.g. "cuda:0" or "cpu". CUDA automatically falls back to CPU inside the model wrapper.

Returns:
- dict: Landslide pixel count, area estimate, area ratio, mask path, visualization path, reliability metadata, and summary path.

Answer guidance:
- Generated images and download links are displayed at the bottom of the answer.
- Do not repeat or list output file paths in the final natural-language answer.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--temp_dir", type=str)
parser.add_argument("--image", type=str, default=None)
parser.add_argument("--output", type=str, default="landslide_segmentation")
parser.add_argument("--weights", type=str, default=None)
parser.add_argument("--device", type=str, default="cuda:0")
args, unknown = parser.parse_known_args()

TEMP_DIR = Path(args.temp_dir or "tmp/tmp/out")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "model" / "landslide_l4s"


def _load_predict():
    import sys

    model_root = str(MODEL_ROOT)
    if model_root not in sys.path:
        sys.path.insert(0, model_root)
    spec = importlib.util.spec_from_file_location("landslide_l4s_infer", MODEL_ROOT / "infer.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load landslide infer.py from {MODEL_ROOT}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        if exc.name == "h5py":
            raise ModuleNotFoundError(
                "LandslideSegmentation requires h5py. Install the model dependencies with: "
                "pip install -r model/landslide_l4s/requirements.txt"
            ) from exc
        raise
    return module.predict


def extract_landslide_area(
    image_path: str,
    output_dir: str = "landslide_segmentation",
    weights_path: Optional[str] = None,
    device: str = "cuda:0",
) -> dict:
    out_dir = TEMP_DIR / output_dir
    predict = _load_predict()
    result = predict(
        {"image": image_path},
        weights_path=weights_path,
        output_dir=str(out_dir),
        device=device,
    )
    result["task"] = "landslide_segmentation"

    summary_path = out_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    result["summary_path"] = str(summary_path)

    try:
        from tools.utils import save_assessment_to_db

        save_assessment_to_db("landslide", result, raster_path=image_path)
    except Exception:
        pass

    return result


if mcp is not None:
    mcp.tool(description=LANDSLIDE_TOOL_DESCRIPTION)(extract_landslide_area)


if __name__ == "__main__":
    if args.image:
        print(json.dumps(
            extract_landslide_area(
                image_path=args.image,
                output_dir=args.output,
                weights_path=args.weights,
                device=args.device,
            ),
            indent=2,
            ensure_ascii=False,
        ))
    else:
        if mcp is None:
            raise FASTMCP_IMPORT_ERROR
        mcp.run()
