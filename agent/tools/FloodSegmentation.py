import argparse
import json
from pathlib import Path
from typing import Dict, Optional

try:
    from fastmcp import FastMCP
except Exception as exc:  # noqa: BLE001
    FastMCP = None
    FASTMCP_IMPORT_ERROR = exc
else:
    FASTMCP_IMPORT_ERROR = None


mcp = FastMCP() if FastMCP is not None else None

FLOOD_TOOL_DESCRIPTION = """
Extract flood inundation areas from a Sentinel-1 SAR GeoTIFF using the Sen1Floods11 flood segmentation model.

Parameters:
- s1_image_path (str): Path to a Sentinel-1 SAR GeoTIFF with two bands: VV and VH.
- output_dir (str): Relative output directory under the tool temp directory.
- model_name (str): Inference method. Use "unet" for the bundled deep model or "threshold" for the fixed VV dB baseline.
- checkpoint_path (str | None): Optional U-Net checkpoint path. If omitted, the bundled checkpoint is used.
- label_path (str | None): Optional label GeoTIFF for computing IoU/F1/precision/recall/accuracy.

Returns:
- dict: Flood pixel counts, flood ratio, saved mask/overlay paths, and optional metrics.

Answer guidance:
- Generated images and download links are displayed at the bottom of the answer.
- Do not repeat or list output file paths in the final natural-language answer.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--temp_dir", type=str)
parser.add_argument("--s1", type=str, default=None)
parser.add_argument("--label", type=str, default=None)
parser.add_argument("--output", type=str, default="flood_segmentation")
parser.add_argument("--model", type=str, default="unet", choices=["unet", "threshold"])
parser.add_argument("--checkpoint", type=str, default=None)
args, unknown = parser.parse_known_args()

TEMP_DIR = Path(args.temp_dir or "tmp/tmp/out")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEN1_ROOT = PROJECT_ROOT / "model" / "sen1floods11-segmentation-main"
DEFAULT_UNET_CHECKPOINT = SEN1_ROOT / "checkpoints" / "unet_flood_best.pt"

VV_MEAN, VV_STD = -10.41, 4.14
VH_MEAN, VH_STD = -17.14, 4.68
THRESHOLD_DB = -13.45

_UNET_MODEL = None
_UNET_CHECKPOINT = None


def _load_chip(s1_image_path: str, label_path: Optional[str] = None):
    import numpy as np
    import rasterio

    with rasterio.open(s1_image_path) as src:
        s1_raw = src.read().astype("float32")
    if s1_raw.shape[0] < 2:
        raise ValueError(
            f"Expected at least two Sentinel-1 bands (VV/VH), got {s1_raw.shape[0]}: {s1_image_path}"
        )
    s1_raw = s1_raw[:2]

    label = valid_mask = None
    if label_path:
        with rasterio.open(label_path) as src:
            label_raw = src.read(1).astype("float32")
        valid_mask = (label_raw != -1).astype("float32")
        label = np.clip(label_raw, 0, 1)

    s1_norm = s1_raw.copy()
    s1_norm[0] = (s1_raw[0] - VV_MEAN) / VV_STD
    s1_norm[1] = (s1_raw[1] - VH_MEAN) / VH_STD
    s1_norm = np.nan_to_num(s1_norm, nan=0.0)
    return s1_raw, s1_norm, label, valid_mask


def _load_unet(checkpoint_path: Optional[str] = None):
    global _UNET_MODEL, _UNET_CHECKPOINT

    import torch
    import segmentation_models_pytorch as smp

    ckpt_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_UNET_CHECKPOINT
    if _UNET_MODEL is not None and _UNET_CHECKPOINT == str(ckpt_path):
        return _UNET_MODEL
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Flood U-Net checkpoint not found: {ckpt_path}")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=2,
        classes=1,
        activation=None,
    )
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _UNET_MODEL = model.to(device)
    _UNET_CHECKPOINT = str(ckpt_path)
    return _UNET_MODEL


def _predict_unet(s1_norm, checkpoint_path: Optional[str] = None):
    import torch

    model = _load_unet(checkpoint_path)
    device = next(model.parameters()).device
    x = torch.from_numpy(s1_norm).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()[0, 0]
    return pred.astype("uint8")


def _predict_threshold(vv_raw):
    return (vv_raw < THRESHOLD_DB).astype("uint8")


def _compute_metrics(pred, label, valid_mask):
    if label is None or valid_mask is None:
        return None
    v = valid_mask.astype(bool)
    p = pred[v]
    t = label[v]
    tp = float(((p == 1) & (t == 1)).sum())
    fp = float(((p == 1) & (t == 0)).sum())
    fn = float(((p == 0) & (t == 1)).sum())
    tn = float(((p == 0) & (t == 0)).sum())
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {
        "IoU": round(iou, 4),
        "F1": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Accuracy": round(accuracy, 4),
    }


def _write_mask(reference_path: str, mask, out_path: Path) -> None:
    import rasterio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=255)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask.astype("uint8"), 1)


def _save_png_outputs(vv_raw, pred, output_dir: Path) -> Dict[str, str]:
    import cv2
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_png_path = output_dir / "flood_mask.png"
    overlay_path = output_dir / "flood_overlay.png"

    vv = vv_raw.astype("float32")
    lo, hi = np.nanpercentile(vv, [2, 98])
    vv_norm = np.clip((vv - lo) / (hi - lo + 1e-6), 0, 1)
    gray = (vv_norm * 255).astype("uint8")
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    color = np.zeros_like(base)
    color[pred > 0] = (255, 80, 0)
    overlay = base.copy()
    mask = pred > 0
    overlay[mask] = cv2.addWeighted(base, 0.45, color, 0.55, 0)[mask]

    cv2.imwrite(str(mask_png_path), pred.astype("uint8") * 255)
    cv2.imwrite(str(overlay_path), overlay)
    return {
        "flood_mask_png_path": str(mask_png_path),
        "overlay_path": str(overlay_path),
    }


def _summarize(pred, valid_mask=None) -> Dict[str, float | int | str]:
    if valid_mask is not None:
        valid = valid_mask.astype(bool)
    else:
        valid = pred >= 0
    total_valid_pixels = int(valid.sum())
    flood_pixels = int((pred[valid] > 0).sum()) if total_valid_pixels else 0
    flood_ratio = float(flood_pixels / total_valid_pixels) if total_valid_pixels else 0.0
    if flood_ratio < 0.05:
        flood_level = "low"
    elif flood_ratio < 0.25:
        flood_level = "moderate"
    else:
        flood_level = "high"
    return {
        "total_valid_pixels": total_valid_pixels,
        "flood_pixels": flood_pixels,
        "flood_ratio": flood_ratio,
        "flood_level": flood_level,
    }


def extract_flood_inundation(
    s1_image_path: str,
    output_dir: str = "flood_segmentation",
    model_name: str = "unet",
    checkpoint_path: str | None = None,
    label_path: str | None = None,
) -> dict:
    s1_raw, s1_norm, label, valid_mask = _load_chip(s1_image_path, label_path)
    if model_name == "threshold":
        pred = _predict_threshold(s1_raw[0])
    elif model_name == "unet":
        pred = _predict_unet(s1_norm, checkpoint_path)
    else:
        raise ValueError(f"Unsupported flood segmentation model: {model_name}")

    out_dir = TEMP_DIR / output_dir
    mask_tif_path = out_dir / "flood_mask.tif"
    _write_mask(s1_image_path, pred, mask_tif_path)
    paths = _save_png_outputs(s1_raw[0], pred, out_dir)
    result = {
        **_summarize(pred, valid_mask),
        "model_name": model_name,
        "flood_mask_path": str(mask_tif_path),
        **paths,
        "metrics": _compute_metrics(pred, label, valid_mask),
        "class_meaning": {
            "0": "non-flood",
            "1": "flood",
        },
        "legend": [
            {"value": 1, "label": "洪水淹没区域", "color": "#0050ff"},
        ],
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    result["summary_path"] = str(summary_path)

    # Persist to database
    try:
        from tools.utils import save_assessment_to_db
        save_assessment_to_db("flood", result, raster_path=s1_path)
    except Exception:
        pass

    return result


if mcp is not None:
    mcp.tool(description=FLOOD_TOOL_DESCRIPTION)(extract_flood_inundation)


if __name__ == "__main__":
    if args.s1:
        print(json.dumps(
            extract_flood_inundation(
                s1_image_path=args.s1,
                output_dir=args.output,
                model_name=args.model,
                checkpoint_path=args.checkpoint,
                label_path=args.label,
            ),
            indent=2,
        ))
    else:
        if mcp is None:
            raise FASTMCP_IMPORT_ERROR
        mcp.run()
