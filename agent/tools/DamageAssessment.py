import argparse
import json
import sys
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

DAMAGE_TOOL_DESCRIPTION = """
Assess building damage from one pre-disaster image and one post-disaster image using the xView2 strong baseline model.

Parameters:
- pre_image_path (str): Path to the pre-disaster RGB image.
- post_image_path (str): Path to the post-disaster RGB image.
- output_dir (str): Relative output directory under the tool temp directory.
- checkpoint_path (str | None): Optional model checkpoint path. If omitted, the bundled xView2 checkpoint is used.
- input_size (int): Model input size. Default is 608.
- threshold (float): Building localization threshold. Default is 0.38.

Returns:
- dict: Damage pixel counts, damage ratio, damage level, and saved output paths.

Answer guidance:
- Generated images and download links are displayed at the bottom of the answer.
- Do not repeat or list output file paths in the final natural-language answer.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--temp_dir", type=str)
parser.add_argument("--pre", type=str, default=None)
parser.add_argument("--post", type=str, default=None)
parser.add_argument("--output", type=str, default="damage_assessment")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--input_size", type=int, default=608)
parser.add_argument("--threshold", type=float, default=0.38)
args, unknown = parser.parse_known_args()

TEMP_DIR = Path(args.temp_dir or "tmp/tmp/out")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
XVIEW2_ROOT = PROJECT_ROOT / "model" / "Xview2_Strong_Baseline-master" / "Xview2_Strong_Baseline-master"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "model"
    / "Xview2_Strong_Baseline-master"
    / "outputs"
    / "checkpoints_long"
    / "res34_unet_double_iter_018000.pt"
)

_MODEL = None
_DEVICE = None
_CHECKPOINT = None


def _ensure_xview2_on_path() -> None:
    xview2_path = str(XVIEW2_ROOT)
    if xview2_path not in sys.path:
        sys.path.insert(0, xview2_path)


def _normalize_image(img):
    import numpy as np

    img = img.astype(np.float32)
    img /= 127.0
    img -= 1.0
    return img


def _compute_stats(damage_mask) -> Dict[str, float]:
    building_total = int((damage_mask > 0).sum())
    no_damage = int((damage_mask == 1).sum())
    minor_damage = int((damage_mask == 2).sum())
    major_damage = int((damage_mask == 3).sum())
    destroyed = int((damage_mask == 4).sum())
    damaged = minor_damage + major_damage + destroyed
    damage_ratio = float(damaged / building_total) if building_total > 0 else 0.0
    if damage_ratio < 0.10:
        damage_level = "minor"
    elif damage_ratio < 0.40:
        damage_level = "moderate"
    else:
        damage_level = "severe"
    return {
        "building_total": building_total,
        "no_damage": no_damage,
        "minor_damage": minor_damage,
        "major_damage": major_damage,
        "destroyed": destroyed,
        "damaged": damaged,
        "damage_ratio": damage_ratio,
        "damage_level": damage_level,
    }


def _load_checkpoint(model, checkpoint_path: Path) -> None:
    import torch

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key.replace("model.", "").replace("module.", "")] = value
    model.load_state_dict(cleaned, strict=False)


def _load_model(checkpoint: Optional[str] = None):
    global _MODEL, _DEVICE, _CHECKPOINT

    import torch

    checkpoint_path = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
    if _MODEL is not None and _CHECKPOINT == str(checkpoint_path):
        return _MODEL, _DEVICE

    _ensure_xview2_on_path()
    from legacy.zoo.models import Res34_Unet_Double

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Res34_Unet_Double(pretrained=False).to(device)
    _load_checkpoint(model, checkpoint_path)
    model.eval()

    _MODEL = model
    _DEVICE = device
    _CHECKPOINT = str(checkpoint_path)
    return _MODEL, _DEVICE


def _predict_damage(pre_image_path: str, post_image_path: str, checkpoint: Optional[str], input_size: int, threshold: float):
    import cv2
    import numpy as np
    import torch

    pre = cv2.imread(pre_image_path, cv2.IMREAD_COLOR)
    post = cv2.imread(post_image_path, cv2.IMREAD_COLOR)
    if pre is None:
        raise FileNotFoundError(f"Failed to read pre-disaster image: {pre_image_path}")
    if post is None:
        raise FileNotFoundError(f"Failed to read post-disaster image: {post_image_path}")

    original_hw = pre.shape[:2]
    if post.shape[:2] != original_hw:
        post = cv2.resize(post, (original_hw[1], original_hw[0]), interpolation=cv2.INTER_LINEAR)

    model_input_pre = pre
    model_input_post = post
    if input_size > 0:
        size = (input_size, input_size)
        model_input_pre = cv2.resize(pre, size, interpolation=cv2.INTER_LINEAR)
        model_input_post = cv2.resize(post, size, interpolation=cv2.INTER_LINEAR)

    img = _normalize_image(np.concatenate([model_input_pre, model_input_post], axis=2))
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)

    model, device = _load_model(checkpoint)
    with torch.no_grad():
        logits = model(tensor.to(device))[0]
        probs = torch.sigmoid(logits).cpu().numpy().transpose(1, 2, 0)

    loc_pred = probs[..., 0]
    building = loc_pred > threshold
    if building.sum() == 0:
        fallback_threshold = np.quantile(loc_pred, 0.8)
        building = loc_pred >= fallback_threshold

    damage = probs[..., 1:].argmax(axis=2).astype(np.uint8) + 1
    damage = damage * building.astype(np.uint8)

    if damage.shape != original_hw:
        damage = cv2.resize(damage, (original_hw[1], original_hw[0]), interpolation=cv2.INTER_NEAREST)
        building = cv2.resize(building.astype(np.uint8), (original_hw[1], original_hw[0]), interpolation=cv2.INTER_NEAREST) > 0

    return building.astype(np.uint8), damage.astype(np.uint8), post


def _save_outputs(building_mask, damage_mask, post_image, output_dir: Path) -> Dict[str, str]:
    import cv2
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    building_path = output_dir / "building_mask.png"
    damage_path = output_dir / "damage_mask.png"
    overlay_path = output_dir / "damage_overlay.png"

    cv2.imwrite(str(building_path), building_mask * 255)
    cv2.imwrite(str(damage_path), damage_mask)

    colors = np.zeros((*damage_mask.shape, 3), dtype=np.uint8)
    colors[damage_mask == 1] = (0, 180, 0)
    colors[damage_mask == 2] = (0, 220, 255)
    colors[damage_mask == 3] = (0, 140, 255)
    colors[damage_mask == 4] = (0, 0, 255)
    overlay = post_image.copy()
    mask = damage_mask > 0
    overlay[mask] = cv2.addWeighted(post_image, 0.45, colors, 0.55, 0)[mask]
    cv2.imwrite(str(overlay_path), overlay)

    return {
        "building_mask_path": str(building_path),
        "damage_mask_path": str(damage_path),
        "overlay_path": str(overlay_path),
    }


def assess_building_damage(
    pre_image_path: str,
    post_image_path: str,
    output_dir: str = "damage_assessment",
    checkpoint_path: str | None = None,
    input_size: int = 608,
    threshold: float = 0.38,
) -> dict:
    building_mask, damage_mask, post = _predict_damage(
        pre_image_path=pre_image_path,
        post_image_path=post_image_path,
        checkpoint=checkpoint_path,
        input_size=input_size,
        threshold=threshold,
    )
    out_dir = TEMP_DIR / output_dir
    stats = _compute_stats(damage_mask)
    paths = _save_outputs(building_mask, damage_mask, post, out_dir)
    result = {
        **stats,
        **paths,
        "class_meaning": {
            "0": "background",
            "1": "building/no-damage",
            "2": "minor-damage",
            "3": "major-damage",
            "4": "destroyed",
        },
        "legend": [
            {"value": 1, "label": "建筑物/无损坏", "color": "#00b400"},
            {"value": 2, "label": "轻微损坏", "color": "#ffdc00"},
            {"value": 3, "label": "严重损坏", "color": "#ff8c00"},
            {"value": 4, "label": "完全摧毁", "color": "#ff0000"},
        ],
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    result["summary_path"] = str(summary_path)

    # Persist to database
    try:
        from tools.utils import save_assessment_to_db
        save_assessment_to_db("damage", result, raster_path=post_image_path)
    except Exception:
        pass

    return result


if mcp is not None:
    mcp.tool(description=DAMAGE_TOOL_DESCRIPTION)(assess_building_damage)


if __name__ == "__main__":
    if args.pre and args.post:
        print(json.dumps(
            assess_building_damage(
                pre_image_path=args.pre,
                post_image_path=args.post,
                output_dir=args.output,
                checkpoint_path=args.checkpoint,
                input_size=args.input_size,
                threshold=args.threshold,
            ),
            indent=2,
        ))
    else:
        if mcp is None:
            raise FASTMCP_IMPORT_ERROR
        mcp.run()
