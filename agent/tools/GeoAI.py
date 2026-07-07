"""GeoAI-based tools for remote sensing object detection, semantic segmentation,
and image similarity analysis using the geoai library.

Supported models (stored in GeoAIModels/):
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
MODELS_DIR = PROJECT_ROOT / "GeoAIModels"

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
    """Return the absolute path to a model file in GeoAIModels/."""
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
    """Summarise a GeoDataFrame of detections into a JSON-serialisable dict."""
    import geopandas as gpd  # noqa: F401

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

    # Save summary JSON
    summary_path = out_dir / f"{task}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    summary["summary_path"] = str(summary_path)

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
    }
    if prob_path:
        summary["probability_path"] = str(prob_path)

    summary_path = out_dir / f"{model}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    summary["summary_path"] = str(summary_path)

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
