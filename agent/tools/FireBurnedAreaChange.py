import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path, PosixPath, WindowsPath
from typing import Any, Dict, Optional

try:
    from fastmcp import FastMCP
except Exception as exc:  # noqa: BLE001
    FastMCP = None
    FASTMCP_IMPORT_ERROR = exc
else:
    FASTMCP_IMPORT_ERROR = None


mcp = FastMCP() if FastMCP is not None else None

FIRE_TOOL_DESCRIPTION = """
Detect wildfire burned-area change from Sentinel-2 pre-fire and post-fire images using the FLOGA BAM-CD model.

Parameters:
- pre_image_path (str | None): Path to a pre-fire Sentinel-2 .npy patch. Expected shape is 9xHxW or HxWx9.
- post_image_path (str | None): Path to a post-fire Sentinel-2 .npy patch. Expected shape is 9xHxW or HxWx9.
- dataset_dir (str | None): Optional FLOGA patch dataset directory containing infer_test.pkl. If image paths are omitted, the bundled sample dataset is used.
- sample_key (str | None): Optional sample key from infer_test.pkl to run one dataset patch.
- output_dir (str): Relative output directory under the tool temp directory.
- checkpoint_path (str | None): Optional FLOGA checkpoint. If omitted, the bundled best_segmentation.pt is used.
- max_patches (int): Maximum number of dataset patches to process when dataset_dir is used. Default is 12.
- input_size (int): Patch size for direct pre/post .npy inference. Default is 256.

Returns:
- dict: Burned-area pixel counts, ratio, area estimate, optional label metrics, and saved output paths.

Answer guidance:
- Return all generated output paths in the tool result, but do not assume every output is useful to the user.
- The language model should inspect the tool result and choose which images are useful for frontend display and which files are useful downloads via the final <Artifacts> block.
- For each selected downloadable file, explain what it contains and how the user can use it, e.g. GIS overlay, quantitative summary, or downstream verification.
- Do not repeat or list output file paths in the final natural-language answer.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--temp_dir", type=str)
parser.add_argument("--pre", type=str, default=None)
parser.add_argument("--post", type=str, default=None)
parser.add_argument("--dataset_dir", type=str, default=None)
parser.add_argument("--sample_key", type=str, default=None)
parser.add_argument("--output", type=str, default="fire_burned_area_change")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--max_patches", type=int, default=12)
parser.add_argument("--input_size", type=int, default=256)
args, unknown = parser.parse_known_args()

TEMP_DIR = Path(args.temp_dir or "tmp/tmp/out")
TEMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TEMP_DIR / ".matplotlib"))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLOGA_PACKAGE_ROOT = PROJECT_ROOT / "model" / "FLOGA-main" / "FLOGA-main"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "model" / "FLOGA-main" / "v2" / "best_segmentation.pt"
DEFAULT_DATASET_DIR = FLOGA_PACKAGE_ROOT / "data" / "floga2018_infer" / "sen2_20_mod_500"

PIXEL_AREA_KM2 = 0.0004
SEN2_RGB_BANDS = (8, 2, 1)

_MODEL = None
_DEVICE = None
_CHECKPOINT = None


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "pathlib._local" and name in {"Path", "PosixPath"}:
            return PosixPath
        if module == "pathlib._local" and name == "WindowsPath":
            return WindowsPath
        return super().find_class(module, name)


def _ensure_floga_on_path() -> None:
    floga_path = str(FLOGA_PACKAGE_ROOT)
    if floga_path not in sys.path:
        sys.path.insert(0, floga_path)


def _load_model(checkpoint_path: Optional[str] = None):
    global _MODEL, _DEVICE, _CHECKPOINT

    import torch

    checkpoint = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
    if _MODEL is not None and _CHECKPOINT == str(checkpoint):
        return _MODEL, _DEVICE
    if not checkpoint.exists():
        raise FileNotFoundError(f"FLOGA checkpoint not found: {checkpoint}")

    _ensure_floga_on_path()
    from models.bam_cd.model import BAM_CD

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BAM_CD(
        encoder_name="resnet101",
        encoder_weights=None,
        in_channels=9,
        classes=2,
        fusion_mode="conc",
        activation=None,
        siamese=False,
        decoder_attention_type="scse",
        decoder_use_batchnorm=True,
    )
    payload = torch.load(str(checkpoint), map_location="cpu")
    state_dict = payload.get("model_state_dict", payload)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    _MODEL = model
    _DEVICE = device
    _CHECKPOINT = str(checkpoint)
    return _MODEL, _DEVICE


def _to_chw_9(array, name: str):
    import numpy as np

    arr = np.asarray(array)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"{name} must be a 3D array, got shape {arr.shape}")
    if arr.shape[0] == 9:
        return arr.astype("float32")
    if arr.shape[-1] == 9:
        return np.moveaxis(arr, -1, 0).astype("float32")
    raise ValueError(f"{name} must have 9 Sentinel-2 bands, got shape {arr.shape}")


def _resize_chw(arr, input_size: int):
    if input_size <= 0 or arr.shape[-2:] == (input_size, input_size):
        return arr

    import torch
    import torch.nn.functional as F

    tensor = torch.from_numpy(arr).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=(input_size, input_size), mode="bilinear", align_corners=False)
    return resized.squeeze(0).numpy()


def _scale_input(x):
    import torch

    return torch.clamp(x, max=10000) / 10000


def _rgb_like(arr):
    import numpy as np

    img = arr[list(SEN2_RGB_BANDS)].astype("float32")
    img = np.clip(img / 10000.0, 0, 1)
    return np.moveaxis(img, 0, -1)


def _predict_pair(pre_arr, post_arr, checkpoint_path: Optional[str] = None):
    import torch

    model, device = _load_model(checkpoint_path)
    before = _scale_input(torch.from_numpy(pre_arr).float().unsqueeze(0)).to(device)
    after = _scale_input(torch.from_numpy(post_arr).float().unsqueeze(0)).to(device)
    with torch.no_grad():
        output = model(before, after)
        pred = output.argmax(1).squeeze(0).to(torch.uint8).cpu().numpy()
    return pred


def _metrics(pred, label):
    if label is None:
        return None

    import numpy as np

    lab = np.asarray(label).squeeze()
    if lab.shape != pred.shape:
        return None
    valid = lab != 2
    pred_b = (pred == 1) & valid
    lab_b = (lab == 1) & valid
    tp = int((pred_b & lab_b).sum())
    fp = int((pred_b & ~lab_b).sum())
    fn = int((~pred_b & lab_b).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {
        "true_positive_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "iou": iou,
        "f1": f1,
    }


def _summarize_prediction(pred) -> Dict[str, float | int | str]:
    burned_pixels = int((pred == 1).sum())
    total_pixels = int(pred.size)
    burned_ratio = float(burned_pixels / total_pixels) if total_pixels else 0.0
    if burned_ratio < 0.02:
        burned_level = "low"
    elif burned_ratio < 0.15:
        burned_level = "moderate"
    else:
        burned_level = "high"
    return {
        "total_pixels": total_pixels,
        "burned_pixels": burned_pixels,
        "burned_ratio": burned_ratio,
        "burned_area_km2_estimate": burned_pixels * PIXEL_AREA_KM2,
        "burned_level": burned_level,
    }


def _save_outputs(pre_arr, post_arr, pred, output_dir: Path, label=None, prefix: str = "burned_area") -> Dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / f"{prefix}_mask.png"
    overlay_path = output_dir / f"{prefix}_overlay.png"
    comparison_path = output_dir / f"{prefix}_comparison.png"

    Image.fromarray((pred.astype("uint8") * 255)).save(mask_path)

    post_rgb = _rgb_like(post_arr)
    color = np.zeros((*pred.shape, 4), dtype="float32")
    color[pred == 1] = (1.0, 0.18, 0.02, 0.58)
    fig, ax = plt.subplots(figsize=(5, 5), num=1, clear=True)
    ax.imshow(post_rgb)
    ax.imshow(color)
    ax.set_title("Burned-area change overlay")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(overlay_path, dpi=140)
    plt.close(fig)

    ncols = 4 if label is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), num=1, clear=True)
    axes[0].imshow(_rgb_like(pre_arr))
    axes[0].set_title("S2 Before")
    axes[1].imshow(post_rgb)
    axes[1].set_title("S2 After")
    axes[2].imshow(pred, vmin=0, vmax=1, cmap="gray")
    axes[2].set_title("Prediction")
    if label is not None:
        axes[3].imshow(label.squeeze(), vmin=0, vmax=2, cmap="viridis")
        axes[3].set_title("Label")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(comparison_path, dpi=140)
    plt.close(fig)

    return {
        "burned_mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "comparison_path": str(comparison_path),
    }


def _resolve_record_path(dataset_dir: Path, value: Any) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = FLOGA_PACKAGE_ROOT / path
    if candidate.exists():
        return candidate
    return dataset_dir / path


def _load_infer_records(dataset_dir: Path) -> dict:
    pkl_path = dataset_dir / "infer_test.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"infer_test.pkl not found in dataset_dir: {dataset_dir}")
    with open(pkl_path, "rb") as f:
        return _CompatUnpickler(f).load()


def _run_single_pair(
    pre_image_path: str,
    post_image_path: str,
    output_dir: Path,
    checkpoint_path: Optional[str],
    input_size: int,
    label_path: Optional[str] = None,
    sample_key: str = "uploaded_pair",
) -> dict:
    import numpy as np

    pre = _resize_chw(_to_chw_9(np.load(pre_image_path), "pre_image"), input_size)
    post = _resize_chw(_to_chw_9(np.load(post_image_path), "post_image"), input_size)
    if pre.shape != post.shape:
        raise ValueError(f"pre/post shapes must match after normalization, got {pre.shape} and {post.shape}")

    label = None
    if label_path:
        label = np.load(label_path).squeeze()
    pred = _predict_pair(pre, post, checkpoint_path)
    paths = _save_outputs(pre, post, pred, output_dir, label=label, prefix="burned_area")
    result = {
        **_summarize_prediction(pred),
        **paths,
        "sample_key": sample_key,
        "metrics": _metrics(pred, label),
        "class_meaning": {
            "0": "unburned/no-change",
            "1": "burned-area change",
            "2": "other-events/ignored-label",
        },
        "legend": [
            {"value": 1, "label": "山火烧毁变化区域", "color": "#ff2e05"},
        ],
    }
    return result


def _run_dataset(
    dataset_dir: Path,
    output_dir: Path,
    checkpoint_path: Optional[str],
    max_patches: int,
    sample_key: Optional[str],
) -> dict:
    import numpy as np

    records = _load_infer_records(dataset_dir)
    items = sorted(records.items())
    if sample_key:
        items = [(key, rec) for key, rec in items if key == sample_key]
        if not items:
            raise KeyError(f"sample_key not found in infer_test.pkl: {sample_key}")
    else:
        items = items[: max(1, max_patches)]

    rows = []
    selected = None
    totals = {"burned_pixels": 0, "total_pixels": 0}
    csv_path = output_dir / "burned_area_metrics.csv"
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    for index, (key, rec) in enumerate(items):
        pre_path = _resolve_record_path(dataset_dir, rec["S2_before_image"])
        post_path = _resolve_record_path(dataset_dir, rec["S2_after_image"])
        label_path = _resolve_record_path(dataset_dir, rec["label"]) if rec.get("label") else None
        result = _run_single_pair(
            pre_image_path=str(pre_path),
            post_image_path=str(post_path),
            output_dir=samples_dir / f"{index:03d}_{key}",
            checkpoint_path=checkpoint_path,
            input_size=256,
            label_path=str(label_path) if label_path else None,
            sample_key=key,
        )
        rows.append(result)
        totals["burned_pixels"] += int(result["burned_pixels"])
        totals["total_pixels"] += int(result["total_pixels"])
        if selected is None or result["burned_pixels"] > selected["burned_pixels"]:
            selected = result

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_key",
            "burned_pixels",
            "total_pixels",
            "burned_ratio",
            "burned_area_km2_estimate",
            "burned_level",
            "overlay_path",
            "comparison_path",
            "burned_mask_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    burned_ratio = totals["burned_pixels"] / totals["total_pixels"] if totals["total_pixels"] else 0.0
    summary = {
        "mode": "dataset",
        "dataset_dir": str(dataset_dir),
        "processed_patches": len(rows),
        "total_pixels": totals["total_pixels"],
        "burned_pixels": totals["burned_pixels"],
        "burned_ratio": burned_ratio,
        "burned_area_km2_estimate": totals["burned_pixels"] * PIXEL_AREA_KM2,
        "selected_sample_key": selected["sample_key"] if selected else None,
        "burned_level": "low" if burned_ratio < 0.02 else "moderate" if burned_ratio < 0.15 else "high",
        "overlay_path": selected["overlay_path"] if selected else "",
        "comparison_path": selected["comparison_path"] if selected else "",
        "metrics_csv_path": str(csv_path),
        "top_samples": rows[:5],
        "legend": [
            {"value": 1, "label": "山火烧毁变化区域", "color": "#ff2e05"},
        ],
    }
    return summary


def detect_fire_burned_area_change(
    pre_image_path: str | None = None,
    post_image_path: str | None = None,
    dataset_dir: str | None = None,
    sample_key: str | None = None,
    output_dir: str = "fire_burned_area_change",
    checkpoint_path: str | None = None,
    max_patches: int = 12,
    input_size: int = 256,
) -> dict:
    out_dir = TEMP_DIR / output_dir

    if bool(pre_image_path) != bool(post_image_path):
        raise ValueError("pre_image_path and post_image_path must be provided together.")

    if pre_image_path and post_image_path:
        result = _run_single_pair(
            pre_image_path=pre_image_path,
            post_image_path=post_image_path,
            output_dir=out_dir,
            checkpoint_path=checkpoint_path,
            input_size=input_size,
            sample_key=sample_key or "uploaded_pair",
        )
        result["mode"] = "pair"
    else:
        ds_dir = Path(dataset_dir) if dataset_dir else DEFAULT_DATASET_DIR
        result = _run_dataset(
            dataset_dir=ds_dir,
            output_dir=out_dir,
            checkpoint_path=checkpoint_path,
            max_patches=max_patches,
            sample_key=sample_key,
        )

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    result["summary_path"] = str(summary_path)
    return result


if mcp is not None:
    mcp.tool(description=FIRE_TOOL_DESCRIPTION)(detect_fire_burned_area_change)


if __name__ == "__main__":
    if args.pre or args.post or args.dataset_dir or args.sample_key:
        print(json.dumps(
            detect_fire_burned_area_change(
                pre_image_path=args.pre,
                post_image_path=args.post,
                dataset_dir=args.dataset_dir,
                sample_key=args.sample_key,
                output_dir=args.output,
                checkpoint_path=args.checkpoint,
                max_patches=args.max_patches,
                input_size=args.input_size,
            ),
            indent=2,
            ensure_ascii=False,
        ))
    else:
        if mcp is None:
            raise FASTMCP_IMPORT_ERROR
        mcp.run()
