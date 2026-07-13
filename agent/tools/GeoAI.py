"""GeoAI-based tools for remote sensing object detection, semantic segmentation,
and image similarity analysis using the geoai library.

Supported models (stored in model/GeoAIModels/):
  - car_detection_usa.pth            (MaskRCNN, 3-channel, car detection)
  - ship_detection.pth               (MaskRCNN, 3-channel, ship detection)
  - solar_panel_detection.pth        (MaskRCNN, 3-channel, solar panel detection)
  - building_footprints_usa.pth      (MaskRCNN, 3-channel, building footprints)
  - building_footprints_usa_rgbn.pth (MaskRCNN, 4-channel RGBN, building footprints)
  - wetland_detection.pth            (MaskRCNN, 4-channel RGBN, wetland detection)
  - water_detection.pth              (MaskRCNN, 4-channel RGBN, water detection)
  - water_detection_unet_best_model.pth (UNet/ResNet34, 3-channel, water segmentation)
  - dinov3_vitl16_sat493m.pth        (DINOv3 ViT-L/16, SAT-493M, feature extraction)
"""

import argparse
import contextlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastmcp import FastMCP
except Exception as exc:  # noqa: BLE001
    FastMCP = None
    FASTMCP_IMPORT_ERROR = exc
else:
    FASTMCP_IMPORT_ERROR = None

mcp = FastMCP() if FastMCP is not None else None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI / configuration
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="GeoAI tools MCP server")
parser.add_argument("--temp_dir", type=str, default="tmp/tmp/out")
args, _unknown = parser.parse_known_args()

TEMP_DIR = Path(args.temp_dir)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "model" / "GeoAIModels"

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

#: Mapping of detection task name → model metadata
DETECTION_MODELS: Dict[str, Dict[str, Any]] = {
    "car": {
        "file": "car_detection_usa.pth",
        "detector_class": "CarDetector",
        "channels": 3,
        "description": "car detection",
    },
    "ship": {
        "file": "ship_detection.pth",
        "detector_class": "ShipDetector",
        "channels": 3,
        "description": "ship detection",
    },
    "solar_panel": {
        "file": "solar_panel_detection.pth",
        "detector_class": "SolarPanelDetector",
        "channels": 3,
        "description": "solar panel detection",
    },
    "building": {
        "file": "building_footprints_usa.pth",
        "detector_class": "BuildingFootprintExtractor",
        "channels": 3,
        "description": "building footprint extraction (RGB)",
    },
    "building_rgbn": {
        "file": "building_footprints_usa_rgbn.pth",
        "detector_class": "BuildingFootprintExtractor",
        "channels": 4,
        "description": "building footprint extraction (RGBN)",
    },
    "wetland": {
        "file": "wetland_detection.pth",
        "detector_class": "ObjectDetector",
        "channels": 4,
        "description": "wetland detection (RGBN)",
    },
    "water": {
        "file": "water_detection.pth",
        "detector_class": "ObjectDetector",
        "channels": 4,
        "description": "water body detection (RGBN)",
    },
}

#: Semantic-segmentation (UNet) models
SEGMENTATION_MODELS: Dict[str, Dict[str, Any]] = {
    "water_unet": {
        "file": "water_detection_unet_best_model.pth",
        "architecture": "unet",
        "encoder_name": "resnet34",
        "num_channels": 3,
        "num_classes": 2,
        "description": "water body segmentation (UNet/ResNet34)",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_model_path(filename: str) -> str:
    """Return the absolute path to a model file in model/GeoAIModels/."""
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. "
            f"Please download it into {MODELS_DIR}/."
        )
    return str(path)


def _create_maskrcnn_with_channels(in_channels: int):
    """Create a MaskRCNN-ResNet50-FPN model that accepts *in_channels* input bands.

    The default torchvision MaskRCNN expects 3-channel input; for 4-band RGBN
    models we replace the first conv layer before loading weights.
    """
    import torch  # noqa: F401  (ensure torch is available)

    from torchvision.models.detection import maskrcnn_resnet50_fpn
    import geoai

    image_mean = [0.485, 0.456, 0.406][:in_channels] if in_channels <= 3 else \
        [0.485, 0.456, 0.406, 0.406]
    image_std = [0.229, 0.224, 0.225][:in_channels] if in_channels <= 3 else \
        [0.229, 0.224, 0.225, 0.225]

    model = maskrcnn_resnet50_fpn(
        weights=None,
        progress=False,
        num_classes=2,
        weights_backbone=None,
        image_mean=image_mean,
        image_std=image_std,
    )

    if in_channels != 3:
        model = geoai.modify_first_conv_for_channels(
            model, in_channels=in_channels, pretrained_channels=3
        )

    return model


def _get_detector(task: str, device: Optional[str] = None):
    """Instantiate and return the appropriate geoai detector for *task*."""
    import geoai

    if task not in DETECTION_MODELS:
        raise ValueError(
            f"Unknown detection task '{task}'. "
            f"Available: {list(DETECTION_MODELS.keys())}"
        )

    cfg = DETECTION_MODELS[task]
    model_path = _resolve_model_path(cfg["file"])
    cls_name = cfg["detector_class"]
    channels = cfg["channels"]
    detector_cls = getattr(geoai, cls_name)

    if channels == 3:
        # Standard 3-channel model — let the detector initialise it internally.
        detector = detector_cls(model_path=model_path, device=device)
    else:
        # Multi-channel model — build a custom backbone and pass it in.
        custom_model = _create_maskrcnn_with_channels(in_channels=channels)
        detector = detector_cls(
            model_path=model_path, model=custom_model, device=device
        )

    return detector


def _gdf_to_summary(gdf, task: str) -> Dict[str, Any]:
    """Summarise a GeoDataFrame of detections into a JSON-serialisable dict.

    ``gdf`` may be ``None`` when ``process_raster`` finds no valid polygons —
    in that case we report zero objects instead of crashing on ``len(None)``.
    """
    import geopandas as gpd  # noqa: F401

    if gdf is None:
        return {
            "task": task,
            "description": DETECTION_MODELS[task]["description"],
            "num_objects": 0,
            "columns": [],
            "note": "No objects detected above the confidence threshold.",
        }

    num_objects = len(gdf)
    summary: Dict[str, Any] = {
        "task": task,
        "description": DETECTION_MODELS[task]["description"],
        "num_objects": int(num_objects),
        "columns": list(gdf.columns),
    }

    if num_objects == 0:
        return summary

    # Geometry types
    geom_types = set(gdf.geometry.geom_type.unique())
    summary["geometry_types"] = list(geom_types)

    # Confidence statistics (if present)
    if "confidence" in gdf.columns:
        conf = gdf["confidence"]
        summary["confidence"] = {
            "min": round(float(conf.min()), 4),
            "max": round(float(conf.max()), 4),
            "mean": round(float(conf.mean()), 4),
        }

    # Area statistics (if CRS is projected)
    try:
        areas = gdf.geometry.area
        summary["area_pixels"] = {
            "min": round(float(areas.min()), 2),
            "max": round(float(areas.max()), 2),
            "mean": round(float(areas.mean()), 2),
            "total": round(float(areas.sum()), 2),
        }
    except Exception:
        pass

    return summary


# ---------------------------------------------------------------------------
# Library-compat patches – work around geoai bugs that crash on real-world input
# ---------------------------------------------------------------------------

def _patch_dataset_channels(target_channels: int) -> None:
    """Patch geoai's ``CustomDataset`` to produce *target_channels* bands per
    chip instead of the hardcoded 3.

    The upstream ``__getitem__`` forces every chip to exactly 3 channels
    (duplicating the last band when fewer, slicing when more). That breaks
    4-band RGBN models whose ``image_mean``/``image_std`` have 4 entries —
    the model's internal Normalize raises ``tensor (3) vs (4)``. We wrap
    the original method to re-pad to the model's expected channel count.
    """
    import geoai
    import torch

    geoai.CustomDataset._target_channels = target_channels

    if getattr(geoai.CustomDataset, "_channels_patched", False):
        return

    _original_getitem = geoai.CustomDataset.__getitem__

    def _getitem_patched(self, idx):
        result = _original_getitem(self, idx)
        target = getattr(self, "_target_channels", 3)
        img = result["image"]
        if img.shape[0] != target:
            new_img = torch.zeros(
                (target, *img.shape[1:]), dtype=img.dtype, device=img.device
            )
            for c in range(target):
                new_img[c] = img[min(c, img.shape[0] - 1)]
            result["image"] = new_img
        return result

    geoai.CustomDataset.__getitem__ = _getitem_patched
    geoai.CustomDataset._channels_patched = True


@contextlib.contextmanager
def _fix_nodata_open():
    """Temporarily patch ``rasterio.open`` to strip NaN/Inf nodata from
    *write* calls.

    geoai's ``semantic_inference_on_geotiff`` copies the source profile
    (which may carry ``nodata=nan``) and only updates ``dtype`` to ``uint8``
    for the mask output. ``nan`` is not a valid nodata for integer dtypes,
    so ``rasterio.open(path, "w", **out_meta)`` raises
    ``ValueError: Given nodata value, nan, is beyond the valid range of its
    data type, uint8``. This context manager intercepts write-mode opens and
    drops the offending nodata value.
    """
    import math
    import rasterio

    _original_open = rasterio.open

    def _patched_open(*args, **kwargs):
        mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
        if mode == "w" and "nodata" in kwargs and kwargs["nodata"] is not None:
            nd = kwargs["nodata"]
            try:
                if isinstance(nd, float) and (math.isnan(nd) or math.isinf(nd)):
                    kwargs["nodata"] = None
            except TypeError:
                pass
        return _original_open(*args, **kwargs)

    rasterio.open = _patched_open
    try:
        yield
    finally:
        rasterio.open = _original_open


# ---------------------------------------------------------------------------
# Visualization helpers – produce PNG overlays the chat UI can render inline
# ---------------------------------------------------------------------------

def _read_rgb_for_display(raster_path: str):
    """Read up to 3 bands from *raster_path* and return a display-ready RGB
    uint8 array plus the raster bounds/crs. Bands are percentile-stretched."""
    import numpy as np
    import rasterio

    with rasterio.open(raster_path) as src:
        n = min(3, src.count)
        data = src.read(list(range(1, n + 1))).astype(np.float32)
        for i in range(data.shape[0]):
            p2, p98 = np.percentile(data[i], [2, 98])
            if p98 > p2:
                data[i] = np.clip((data[i] - p2) / (p98 - p2), 0, 1)
        if n == 1:
            data = np.repeat(data, 3, axis=0)
        elif n == 2:
            data = np.concatenate([data, data[:1]], axis=0)
        rgb = np.moveaxis(data, 0, -1)
        rgb = (rgb * 255).astype(np.uint8)
        return rgb, src.bounds, src.crs


def _save_detection_overlay(
    raster_path: str, geojson_path: str, output_png: str, task: str
) -> None:
    """Overlay detected objects as bounding boxes on the raster and save as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import geopandas as gpd
    from pathlib import Path

    rgb, bounds, crs = _read_rgb_for_display(raster_path)
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb, extent=extent, origin="upper")

    if Path(geojson_path).exists():
        try:
            gdf = gpd.read_file(geojson_path)
            if len(gdf) > 0:
                if gdf.crs is not None and crs is not None and gdf.crs != crs:
                    gdf = gdf.to_crs(crs)
                # Draw a bounding-box rectangle for each detected object
                bnd = gdf.geometry.bounds
                img_w = bounds.right - bounds.left
                lw = max(1.0, img_w / rgb.shape[1] * 1.5)
                fs = max(5, int(rgb.shape[1] / 80))
                for _, row in bnd.iterrows():
                    rect = Rectangle(
                        (row.minx, row.miny),
                        row.maxx - row.minx,
                        row.maxy - row.miny,
                        linewidth=lw,
                        edgecolor="#ff2020",
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                # Annotate confidence scores
                if "confidence" in gdf.columns:
                    for (_, br), conf in zip(bnd.iterrows(), gdf["confidence"]):
                        ax.text(
                            br.minx, br.maxy, f"{conf:.2f}",
                            fontsize=fs, color="#ffff00",
                            fontweight="bold",
                            va="bottom",
                            bbox=dict(
                                boxstyle="round,pad=0.1",
                                fc="#00000080",
                                ec="none",
                            ),
                        )
        except Exception:
            pass

    ax.set_axis_off()
    n_label = ""
    try:
        n_label = f" — {len(gdf)} objects" if Path(geojson_path).exists() else ""
    except Exception:
        pass
    ax.set_title(
        f"{task} detection{f' ({Path(geojson_path).stem})' if Path(geojson_path).exists() else ''}{n_label}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_png, dpi=120, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _save_segmentation_overlay(
    input_path: str, mask_path: str, output_png: str, model_name: str
) -> None:
    """Render a segmentation mask as a colourised overlay on the input image."""
    import numpy as np
    import rasterio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    try:
        rgb, _, _ = _read_rgb_for_display(input_path)
        if rgb.shape[:2] != mask.shape:
            pil = Image.fromarray(rgb)
            pil = pil.resize((mask.shape[1], mask.shape[0]), Image.LANCZOS)
            rgb = np.array(pil)
    except Exception:
        rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb)
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask > 0] = [0.0, 0.75, 1.0, 0.45]
    ax.imshow(overlay)
    ax.set_axis_off()
    ax.set_title(f"{model_name} segmentation", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_png, dpi=120, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _save_similarity_overlay(
    similarity_path: str, output_png: str, query_point=None
) -> None:
    """Render the DINOv3 similarity map as a heatmap PNG."""
    import numpy as np
    import rasterio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    if similarity_path.lower().endswith((".tif", ".tiff")):
        with rasterio.open(similarity_path) as src:
            sim = src.read(1)
            extent = [
                src.bounds.left, src.bounds.right,
                src.bounds.bottom, src.bounds.top,
            ]
            use_extent = src.crs is not None
    else:
        sim = np.array(Image.open(similarity_path).convert("L"), dtype=np.float32) / 255.0
        extent = None
        use_extent = False

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(
        sim, cmap="hot", origin="upper",
        extent=extent if use_extent else None,
        vmin=float(sim.min()), vmax=float(sim.max()),
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="similarity")
    if query_point is not None:
        ax.plot(query_point[0], query_point[1], "c+", markersize=18, mew=3)
    ax.set_axis_off()
    ax.set_title("DINOv3 patch similarity", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_png, dpi=120, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _save_features_preview(features_npy: str, output_png: str) -> None:
    """Render a PCA (top-3 components) preview of DINOv3 patch features."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(features_npy)  # (h_patches, w_patches, embed_dim)
    h, w, d = data.shape
    flat = data.reshape(-1, d).astype(np.float32)
    flat -= flat.mean(axis=0, keepdims=True)
    try:
        # Economy SVD → principal components
        U, S, Vt = np.linalg.svd(flat, full_matrices=False)
        pc = (flat @ Vt[:3].T).reshape(h, w, 3)
    except Exception:
        pc = flat.reshape(h, w, d)[..., :3]

    for i in range(pc.shape[-1]):
        lo, hi = pc[..., i].min(), pc[..., i].max()
        if hi > lo:
            pc[..., i] = (pc[..., i] - lo) / (hi - lo)
    pc = np.clip(pc, 0, 1)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(pc)
    ax.set_axis_off()
    ax.set_title("DINOv3 features (PCA preview)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_png, dpi=120, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _create_dinov3_processor(
    weights_path: str,
    device: Optional[str] = None,
):
    """Create a DINOv3GeoProcessor, working around the geoai library's
    failure to pass ``pretrained=False`` when a custom ``weights_path``
    is supplied (which causes an unwanted network download).

    The workaround monkey-patches ``_load_model`` so that the DINOv3 model
    is always created with ``pretrained=False`` before loading local weights.
    """
    import os
    import types
    import torch
    import geoai

    def _patched_load_model(self, wp=None):
        model = torch.hub.load(
            repo_or_dir=self.dinov3_location,
            model=self.model_name,
            source=self.dinov3_source,
            pretrained=False,
            weights=None,
            trust_repo=True,
            skip_validation=True,
        )
        if wp and os.path.exists(wp):
            state_dict = torch.load(wp, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        model.eval()
        return model

    original = geoai.DINOv3GeoProcessor._load_model
    geoai.DINOv3GeoProcessor._load_model = _patched_load_model
    try:
        processor = geoai.DINOv3GeoProcessor(
            model_name="dinov3_vitl16",
            weights_path=weights_path,
            device=device,
        )
    finally:
        geoai.DINOv3GeoProcessor._load_model = original

    # The upstream preprocess_image_for_dinov3 only handles 1-band and >3-band
    # inputs, but a 2-band GeoTIFF slips through and produces a 2-channel PIL
    # image that crashes the torchvision Normalize transform (expects 3). Wrap
    # it so any sub-3-band array is padded to 3 channels first.
    import numpy as _np
    _original_preprocess = processor.preprocess_image_for_dinov3

    def _safe_preprocess(data, target_size=896, normalize_percentile=True):
        if isinstance(data, _np.ndarray):
            if data.ndim == 2:
                data = _np.repeat(data[_np.newaxis, :, :], 3, axis=0)
            elif data.ndim == 3 and data.shape[0] < 3:
                if data.shape[0] == 1:
                    data = _np.repeat(data, 3, axis=0)
                else:  # 2 bands → append the first band as a third
                    data = _np.concatenate([data, data[:1]], axis=0)
        return _original_preprocess(data, target_size, normalize_percentile)

    processor.preprocess_image_for_dinov3 = _safe_preprocess
    return processor


# ---------------------------------------------------------------------------
# MCP tool: object detection
# ---------------------------------------------------------------------------

GEOAI_DETECTION_TOOL_DESCRIPTION = """
Detect and extract objects from remote sensing imagery using pre-trained MaskRCNN models.

Supported tasks:
  - "car"             : detect cars (3-band RGB)
  - "ship"            : detect ships (3-band RGB)
  - "solar_panel"     : detect solar panels (3-band RGB)
  - "building"        : extract building footprints (3-band RGB)
  - "building_rgbn"   : extract building footprints (4-band RGBN)
  - "wetland"         : detect wetlands (4-band RGBN)
  - "water"           : detect water bodies (4-band RGBN)

Parameters:
  - raster_path (str): Path to the input raster image (GeoTIFF, etc.).
  - task (str): Detection task name (see list above).
  - output_dir (str): Relative output directory under the tool temp dir.
  - confidence_threshold (float): Minimum detection confidence (0.0-1.0, default 0.5).
  - batch_size (int): Batch size for inference (default 4).
  - filter_edges (bool): Filter objects at image edges (default True).
  - band_indexes (list[int] | None): Band indexes to use (1-based). If None, uses all bands.
  - device (str | None): Device string, e.g. "cuda:0" or "cpu". Auto-detects if None.

Returns:
  - dict: Summary with object count, confidence/area stats, and paths to saved GeoJSON.
"""


def geoai_object_detection(
    raster_path: str,
    task: str = "building",
    output_dir: str = "geoai_detection",
    confidence_threshold: float = 0.5,
    batch_size: int = 4,
    filter_edges: bool = True,
    band_indexes: Optional[List[int]] = None,
    device: Optional[str] = None,
) -> dict:
    """Run object detection / instance segmentation on *raster_path*."""
    import os

    if not os.path.exists(raster_path):
        raise FileNotFoundError(
            f"Input raster file not found: {raster_path}. "
            f"Make sure to use the full path returned by get_filelist or the "
            f"upload directory path. The file must exist before calling "
            f"this tool."
        )

    out_dir = TEMP_DIR / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = _get_detector(task, device=device)

    geojson_path = out_dir / f"{task}_detections.geojson"

    # Ensure the dataset feeds the model the right number of bands (upstream
    # CustomDataset hardcodes 3, which crashes 4-band RGBN models).
    _patch_dataset_channels(DETECTION_MODELS[task]["channels"])

    gdf = detector.process_raster(
        raster_path=raster_path,
        output_path=str(geojson_path),
        batch_size=batch_size,
        filter_edges=filter_edges,
        band_indexes=band_indexes,
        confidence_threshold=confidence_threshold,
    )

    summary = _gdf_to_summary(gdf, task)
    summary["model_file"] = DETECTION_MODELS[task]["file"]
    summary["geojson_path"] = str(geojson_path)
    summary["raster_path"] = raster_path

    # Render a PNG overlay the chat UI can display inline
    overlay_path = out_dir / f"{task}_overlay.png"
    try:
        _save_detection_overlay(
            raster_path, str(geojson_path), str(overlay_path), task
        )
        summary["overlay_path"] = str(overlay_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Detection overlay rendering failed: %s", exc)

    # Save summary JSON
    summary_path = out_dir / f"{task}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    summary["summary_path"] = str(summary_path)

    # Persist to database
    try:
        from tools.utils import save_assessment_to_db
        save_assessment_to_db(task, summary, raster_path=raster_path)
    except Exception:
        pass

    return summary


# ---------------------------------------------------------------------------
# MCP tool: semantic segmentation
# ---------------------------------------------------------------------------

GEOAI_SEGMENTATION_TOOL_DESCRIPTION = """
Perform semantic segmentation on remote sensing imagery using trained UNet models.

Supported models:
  - "water_unet" : water body segmentation (UNet/ResNet34, 3-band RGB, 2 classes)

Parameters:
  - input_path (str): Path to input image (GeoTIFF, PNG, JPG, etc.).
  - model (str): Model key from the segmentation registry (default "water_unet").
  - output_dir (str): Relative output directory under the tool temp dir.
  - window_size (int): Sliding window size for large images (default 512).
  - overlap (int): Overlap between adjacent windows (default 256).
  - batch_size (int): Batch size for inference (default 4).
  - device (str | None): Device string. Auto-detects if None.
  - probability_threshold (float | None): Threshold for binary classification (0-1).
      If None, uses argmax.

Returns:
  - dict: Summary with output mask path and class meaning.
"""


def geoai_semantic_segmentation(
    input_path: str,
    model: str = "water_unet",
    output_dir: str = "geoai_segmentation",
    window_size: int = 512,
    overlap: int = 256,
    batch_size: int = 4,
    device: Optional[str] = None,
    probability_threshold: Optional[float] = None,
) -> dict:
    """Run semantic segmentation on *input_path* using a UNet model."""
    import os
    import geoai

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input image file not found: {input_path}. "
            f"Make sure to use the full path returned by get_filelist or the "
            f"upload directory path. The file must exist before calling "
            f"this tool."
        )

    if model not in SEGMENTATION_MODELS:
        raise ValueError(
            f"Unknown segmentation model '{model}'. "
            f"Available: {list(SEGMENTATION_MODELS.keys())}"
        )

    cfg = SEGMENTATION_MODELS[model]
    model_path = _resolve_model_path(cfg["file"])

    out_dir = TEMP_DIR / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = out_dir / f"{model}_mask.tif"
    prob_path = out_dir / f"{model}_probability.tif" if probability_threshold is not None else None

    # geoai copies the source profile (which may carry nodata=nan) into the
    # uint8 mask output; nan is invalid for uint8 and crashes rasterio.open.
    # The context manager strips NaN/Inf nodata from write calls.
    with _fix_nodata_open():
        geoai.semantic_segmentation(
            input_path=input_path,
            output_path=str(output_path),
            model_path=model_path,
            architecture=cfg["architecture"],
            encoder_name=cfg["encoder_name"],
            num_channels=cfg["num_channels"],
            num_classes=cfg["num_classes"],
            window_size=window_size,
            overlap=overlap,
            batch_size=batch_size,
            device=device,
            probability_path=str(prob_path) if prob_path else None,
            probability_threshold=probability_threshold,
        )

    summary: Dict[str, Any] = {
        "model": model,
        "model_file": cfg["file"],
        "description": cfg["description"],
        "input_path": input_path,
        "output_mask_path": str(output_path),
        "class_meaning": {"0": "background", "1": cfg["description"].split("(")[0].strip()},
        "legend": [
            {"value": 1, "label": cfg["description"].split("(")[0].strip(), "color": "#00bfff"},
        ],
    }
    if prob_path:
        summary["probability_path"] = str(prob_path)

    # Render a PNG overlay the chat UI can display inline
    overlay_path = out_dir / f"{model}_overlay.png"
    try:
        _save_segmentation_overlay(
            input_path, str(output_path), str(overlay_path), model
        )
        summary["overlay_path"] = str(overlay_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Segmentation overlay rendering failed: %s", exc)

    summary_path = out_dir / f"{model}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    summary["summary_path"] = str(summary_path)

    # Persist to database
    try:
        from tools.utils import save_assessment_to_db
        save_assessment_to_db(model, summary, raster_path=input_path)
    except Exception:
        pass

    return summary


# ---------------------------------------------------------------------------
# MCP tool: DINOv3 image similarity
# ---------------------------------------------------------------------------

GEOAI_SIMILARITY_TOOL_DESCRIPTION = """
Extract features and compute patch-level similarity from satellite/aerial imagery
using the DINOv3 ViT-L/16 foundation model (SAT-493M weights).

This tool loads a GeoTIFF/image, extracts DINOv3 embeddings for each patch, and
computes a similarity map relative to a user-specified query point. The result
highlights regions that are visually similar to the query location.

Parameters:
  - image_path (str): Path to the input image (GeoTIFF, PNG, JPG, etc.).
  - query_x (float): X coordinate (pixel column) of the query point.
  - query_y (float): Y coordinate (pixel row) of the query point.
  - output_dir (str): Relative output directory under the tool temp dir.
  - bands (list[int] | None): Band indexes to use (1-based). If None, uses first 3.
  - target_size (int): Target size for DINOv3 processing (default 896).
  - device (str | None): Device string. Auto-detects if None.

Returns:
  - dict: Summary with paths to similarity map and feature files.
"""


def geoai_image_similarity(
    image_path: str,
    query_x: float,
    query_y: float,
    output_dir: str = "geoai_similarity",
    bands: Optional[List[int]] = None,
    target_size: int = 896,
    device: Optional[str] = None,
) -> dict:
    """Compute DINOv3 patch similarity for *image_path* around a query point."""
    import os
    import geoai

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Input image file not found: {image_path}. "
            f"Make sure to use the full path returned by get_filelist or the "
            f"upload directory path."
        )

    weights_path = _resolve_model_path("dinov3_vitl16_sat493m.pth")

    out_dir = TEMP_DIR / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = _create_dinov3_processor(weights_path, device=device)

    results = processor.compute_similarity(
        source=image_path,
        query_coords=(float(query_x), float(query_y)),
        output_dir=str(out_dir),
        bands=bands,
        target_size=target_size,
        save_features=True,
    )

    # Build summary from results dict
    import numpy as np

    patch_coords = results.get("patch_coords", (0, 0))
    patch_grid = results.get("patch_grid_size", (0, 0))
    image_size = results.get("image_size", (0, 0))

    summary: Dict[str, Any] = {
        "model": "dinov3_vitl16_sat493m",
        "image_path": image_path,
        "query_point": [float(query_x), float(query_y)],
        "patch_coords": list(patch_coords),
        "patch_grid": list(patch_grid),
        "image_size": list(image_size),
        "output_dir": str(out_dir),
    }

    # Similarity statistics
    sim_arr = results.get("similarities")
    if isinstance(sim_arr, np.ndarray):
        summary["similarity_stats"] = {
            "max": round(float(sim_arr.max()), 4),
            "min": round(float(sim_arr.min()), 4),
            "mean": round(float(sim_arr.mean()), 4),
            "std": round(float(sim_arr.std()), 4),
        }

    # Output file paths (compute_similarity saves them internally)
    output_paths = results.get("output_paths", {})
    if output_paths:
        summary["output_paths"] = output_paths
    else:
        # Fallback: construct expected paths
        px, py = patch_coords
        summary["output_paths"] = {
            "similarity": str(out_dir / f"similarity_patch_{px}_{py}.tif"),
            "features": str(out_dir / f"features_patch_{px}_{py}.npy"),
            "metadata": str(out_dir / f"metadata_patch_{px}_{py}.json"),
        }

    # Save the raw similarities array for downstream use
    if isinstance(sim_arr, np.ndarray):
        sim_npy_path = out_dir / "similarities.npy"
        np.save(str(sim_npy_path), sim_arr)
        summary["similarities_npy_path"] = str(sim_npy_path)

    # Render a PNG heatmap the chat UI can display inline
    sim_source = summary.get("output_paths", {}).get("similarity") or ""
    overlay_path = out_dir / "similarity_overlay.png"
    try:
        if sim_source:
            _save_similarity_overlay(
                sim_source, str(overlay_path), query_point=(query_x, query_y)
            )
            summary["overlay_path"] = str(overlay_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Similarity overlay rendering failed: %s", exc)

    summary_path = out_dir / "similarity_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    summary["summary_path"] = str(summary_path)

    return summary


# ---------------------------------------------------------------------------
# MCP tool: DINOv3 feature extraction only
# ---------------------------------------------------------------------------

GEOAI_FEATURE_EXTRACTION_TOOL_DESCRIPTION = """
Extract DINOv3 patch embeddings from satellite/aerial imagery without computing
similarity. Useful for downstream clustering, classification, or change detection.

Parameters:
  - image_path (str): Path to the input image (GeoTIFF, PNG, JPG, etc.).
  - output_dir (str): Relative output directory under the tool temp dir.
  - bands (list[int] | None): Band indexes to use (1-based). If None, uses first 3.
  - target_size (int): Target size for DINOv3 processing (default 896).
  - device (str | None): Device string. Auto-detects if None.

Returns:
  - dict: Summary with path to saved features (.npy) and patch dimensions.
"""


def geoai_extract_features(
    image_path: str,
    output_dir: str = "geoai_features",
    bands: Optional[List[int]] = None,
    target_size: int = 896,
    device: Optional[str] = None,
) -> dict:
    """Extract DINOv3 patch embeddings from *image_path*."""
    import os
    import numpy as np
    import geoai

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Input image file not found: {image_path}. "
            f"Make sure to use the full path returned by get_filelist or the "
            f"upload directory path."
        )

    weights_path = _resolve_model_path("dinov3_vitl16_sat493m.pth")

    out_dir = TEMP_DIR / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = _create_dinov3_processor(weights_path, device=device)

    # Load and preprocess image
    data, metadata = processor.load_image(image_path, bands=bands)
    image = processor.preprocess_image_for_dinov3(data, target_size)

    # Extract features
    features, h_patches, w_patches = processor.extract_features(image)

    # Convert to numpy (may be a CUDA tensor)
    import torch
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()

    # Save features
    features_path = out_dir / "dinov3_features.npy"
    np.save(str(features_path), features)

    summary: Dict[str, Any] = {
        "model": "dinov3_vitl16_sat493m",
        "image_path": image_path,
        "features_path": str(features_path),
        "feature_shape": list(features.shape),
        "patch_grid": [int(h_patches), int(w_patches)],
        "embed_dim": int(features.shape[-1]) if features.ndim == 3 else None,
    }

    # Render a PCA preview PNG the chat UI can display inline
    overlay_path = out_dir / "features_overlay.png"
    try:
        _save_features_preview(str(features_path), str(overlay_path))
        summary["overlay_path"] = str(overlay_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Features overlay rendering failed: %s", exc)

    summary_path = out_dir / "features_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    summary["summary_path"] = str(summary_path)

    return summary


# ---------------------------------------------------------------------------
# MCP registration
# ---------------------------------------------------------------------------

if mcp is not None:
    mcp.tool(description=GEOAI_DETECTION_TOOL_DESCRIPTION)(geoai_object_detection)
    mcp.tool(description=GEOAI_SEGMENTATION_TOOL_DESCRIPTION)(geoai_semantic_segmentation)
    mcp.tool(description=GEOAI_SIMILARITY_TOOL_DESCRIPTION)(geoai_image_similarity)
    mcp.tool(description=GEOAI_FEATURE_EXTRACTION_TOOL_DESCRIPTION)(geoai_extract_features)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(description="GeoAI tools CLI")
    cli_parser.add_argument("--temp_dir", type=str, default=str(TEMP_DIR))
    sub = cli_parser.add_subparsers(dest="command")

    # detect
    p_det = sub.add_parser("detect", help="Object detection")
    p_det.add_argument("--raster", type=str, required=True)
    p_det.add_argument("--task", type=str, default="building",
                       choices=list(DETECTION_MODELS.keys()))
    p_det.add_argument("--output", type=str, default="geoai_detection")
    p_det.add_argument("--confidence", type=float, default=0.5)
    p_det.add_argument("--batch_size", type=int, default=4)
    p_det.add_argument("--device", type=str, default=None)

    # segment
    p_seg = sub.add_parser("segment", help="Semantic segmentation")
    p_seg.add_argument("--input", type=str, required=True)
    p_seg.add_argument("--model", type=str, default="water_unet",
                       choices=list(SEGMENTATION_MODELS.keys()))
    p_seg.add_argument("--output", type=str, default="geoai_segmentation")
    p_seg.add_argument("--device", type=str, default=None)

    # similarity
    p_sim = sub.add_parser("similarity", help="DINOv3 image similarity")
    p_sim.add_argument("--image", type=str, required=True)
    p_sim.add_argument("--x", type=float, required=True)
    p_sim.add_argument("--y", type=float, required=True)
    p_sim.add_argument("--output", type=str, default="geoai_similarity")
    p_sim.add_argument("--device", type=str, default=None)

    # features
    p_feat = sub.add_parser("features", help="DINOv3 feature extraction")
    p_feat.add_argument("--image", type=str, required=True)
    p_feat.add_argument("--output", type=str, default="geoai_features")
    p_feat.add_argument("--device", type=str, default=None)

    cli_args = cli_parser.parse_args()

    if cli_args.command is None:
        if mcp is None:
            raise FASTMCP_IMPORT_ERROR
        mcp.run()
    elif cli_args.command == "detect":
        print(json.dumps(geoai_object_detection(
            raster_path=cli_args.raster,
            task=cli_args.task,
            output_dir=cli_args.output,
            confidence_threshold=cli_args.confidence,
            batch_size=cli_args.batch_size,
            device=cli_args.device,
        ), indent=2, default=str))
    elif cli_args.command == "segment":
        print(json.dumps(geoai_semantic_segmentation(
            input_path=cli_args.input,
            model=cli_args.model,
            output_dir=cli_args.output,
            device=cli_args.device,
        ), indent=2, default=str))
    elif cli_args.command == "similarity":
        print(json.dumps(geoai_image_similarity(
            image_path=cli_args.image,
            query_x=cli_args.x,
            query_y=cli_args.y,
            output_dir=cli_args.output,
            device=cli_args.device,
        ), indent=2, default=str))
    elif cli_args.command == "features":
        print(json.dumps(geoai_extract_features(
            image_path=cli_args.image,
            output_dir=cli_args.output,
            device=cli_args.device,
        ), indent=2, default=str))
