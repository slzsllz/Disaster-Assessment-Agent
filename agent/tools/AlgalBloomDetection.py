import argparse
import importlib.util
import json
import os
from pathlib import Path

try:
    from fastmcp import FastMCP
except Exception as exc:  # noqa: BLE001
    FastMCP = None
    FASTMCP_IMPORT_ERROR = exc
else:
    FASTMCP_IMPORT_ERROR = None


mcp = FastMCP() if FastMCP is not None else None

ALGAL_BLOOM_TOOL_DESCRIPTION = """
Detect candidate algal bloom or high-chlorophyll water areas from a 7-band Sentinel-2 GeoTIFF using the standard NDCI spectral index.
This is an index-based remote-sensing method, not a trained neural network.

Parameters:
- image_path (str): Path to a single 7-band Sentinel-2 GeoTIFF. Required band order is [B2, B3, B4, B5, B8, B11, B12].
- output_dir (str): Relative output directory under the tool temp directory.
- ndci_threshold (float): Candidate bloom threshold for NDCI. Default is 0.1.
- sample_factor (int): Downsampling factor used for preview figures. Default is 4.
- cloud_percentile (float): High-percentile threshold used to remove very bright cloud-like pixels. Default is 98.0.

Returns:
- dict: Bloom candidate pixel count, area estimate, bloom ratio of detected water, georeferenced mask path, diagnostic image paths, reliability metadata, and summary path.

Interpretation guidance:
- outputs.area_ratio is bloom candidate area divided by detected water area, not the full image area.
- This tool cannot identify algal species, toxicity, or calibrated chlorophyll concentration.
- False positives may come from shallow bottom reflectance, suspended sediment, cloud edges, or sunglint.
- The input must include the B5 red-edge band; standard NDCI cannot run without it.

Answer guidance:
- Return all generated output paths in the tool result, but do not assume every output is useful to the user.
- The second-pass multimodal reviewer will inspect the tool outputs and choose which images are useful for frontend display and which files are useful downloads.
- The first-pass answer should focus on the algal-bloom candidate detection result, not on listing or explaining output files.
- Do not repeat or list output file paths in the final natural-language answer.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--temp_dir", type=str)
parser.add_argument("--image", type=str, default=None)
parser.add_argument("--output", type=str, default="algal_bloom_detection")
parser.add_argument("--ndci_threshold", type=float, default=0.1)
parser.add_argument("--sample_factor", type=int, default=4)
parser.add_argument("--cloud_percentile", type=float, default=98.0)
args, unknown = parser.parse_known_args()

TEMP_DIR = Path(args.temp_dir or "tmp/tmp/out")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TEMP_DIR / ".matplotlib"))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "model" / "algalbloom_ndci"


def _load_predict():
    import sys

    model_root = str(MODEL_ROOT)
    if model_root not in sys.path:
        sys.path.insert(0, model_root)
    spec = importlib.util.spec_from_file_location("algalbloom_ndci_infer", MODEL_ROOT / "infer.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load algal bloom infer.py from {MODEL_ROOT}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name or "required package"
        raise ModuleNotFoundError(
            f"AlgalBloomDetection requires {missing}. Install the model dependencies with: "
            "pip install -r model/algalbloom_ndci/requirements.txt"
        ) from exc
    return module.predict


def detect_algal_bloom(
    image_path: str,
    output_dir: str = "algal_bloom_detection",
    ndci_threshold: float = 0.1,
    sample_factor: int = 4,
    cloud_percentile: float = 98.0,
) -> dict:
    out_dir = TEMP_DIR / output_dir
    predict = _load_predict()
    result = predict(
        {"image": image_path},
        output_dir=str(out_dir),
        ndci_threshold=ndci_threshold,
        sample_factor=sample_factor,
        cloud_percentile=cloud_percentile,
    )
    result["task"] = "algal_bloom_detection"
    result["diagnostic_outputs"] = {
        "true_color_rgb_path": str(out_dir / "true_color_rgb.png"),
        "mndwi_water_mask_path": str(out_dir / "mndwi_water_mask.png"),
        "mndwi_heatmap_path": str(out_dir / "mndwi_heatmap.png"),
        "water_ndci_heatmap_path": str(out_dir / "water_ndci_heatmap.png"),
        "ndci_bloom_overlay_path": str(out_dir / "ndci_bloom_overlay.png"),
        "ndci_histogram_path": str(out_dir / "ndci_histogram.png"),
        "ndci_comparison_path": str(out_dir / "ndci_comparison.png"),
        "ndci_bloom_mask_path": str(out_dir / "ndci_bloom_mask.tif"),
        "stats_path": str(out_dir / "stats.json"),
    }
    result["interpretation"] = {
        "area_ratio_meaning": "Bloom candidate area divided by detected water area, not full image area.",
        "method_limitation": "NDCI is a relative spectral index and does not identify species, toxicity, or calibrated chlorophyll concentration.",
        "input_requirement": "Input must be a 7-band Sentinel-2 GeoTIFF ordered as [B2, B3, B4, B5, B8, B11, B12].",
    }
    result["legend"] = [
        {"label": "候选藻华/高叶绿素区域", "color": "#ff00ff", "value": 1},
    ]

    summary_path = out_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    result["summary_path"] = str(summary_path)

    try:
        from tools.utils import save_assessment_to_db

        save_assessment_to_db("algal_bloom", result, raster_path=image_path)
    except Exception:
        pass

    return result


if mcp is not None:
    mcp.tool(description=ALGAL_BLOOM_TOOL_DESCRIPTION)(detect_algal_bloom)


if __name__ == "__main__":
    if args.image:
        print(json.dumps(
            detect_algal_bloom(
                image_path=args.image,
                output_dir=args.output,
                ndci_threshold=args.ndci_threshold,
                sample_factor=args.sample_factor,
                cloud_percentile=args.cloud_percentile,
            ),
            indent=2,
            ensure_ascii=False,
        ))
    else:
        if mcp is None:
            raise FASTMCP_IMPORT_ERROR
        mcp.run()
