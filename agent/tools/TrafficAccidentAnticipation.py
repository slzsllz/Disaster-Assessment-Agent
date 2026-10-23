"""MCP wrapper for the packaged UString video and feature anticipation model."""
import argparse
import asyncio
from functools import wraps
import importlib.util
import json
from pathlib import Path
from typing import Optional

try:
    from fastmcp import FastMCP
except ImportError as exc:
    FastMCP = None
    FASTMCP_IMPORT_ERROR = exc

mcp = FastMCP() if FastMCP is not None else None
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / 'model' / 'UString'
TEMP_DIR = Path('tmp/tmp/out')

TOOL_DESCRIPTION = """
Anticipate traffic accidents from a dashcam video_path or precomputed features_path using UString.
Supply exactly one input. Raw video uses bundled KITTI Cascade R-CNN and VGG16 weights.
Input features_path must contain data=(T,20,4096) and det=(T,19,4 or 6), 1<=T<=100.
Standalone images and remote-sensing imagery are not supported.
For video, start_time selects the clip offset in seconds; max_frames (1..100, default 100)
limits sampled frames. Only the selected clip is analyzed, not necessarily the whole video.
Sampling fps is capped at source FPS. Returned timestamps refer to the original video.
feature_batch_size controls VGG crop batch size (1..20, default 4).
Optional detector_weights and vgg_weights override bundled files.
The torchvision detector port is not benchmark-validated against original MMDetection.
Recognized bundled legacy checkpoints are supported automatically after SHA256 verification:
their unsaved/unoptimized recurrent layers are rebuilt with upstream initialization seed 123.
This returns status=legacy_compatible and unvalidated reliability, not recovered trained weights
or verified original-runtime accuracy. No allow_legacy_checkpoint flag is needed for bundled files.
Unknown incomplete checkpoints still raise an error. Set allow_legacy_checkpoint=True only for
explicitly requested experiments; that mode returns status=experimental with random recurrence.
Complete weights_path returns status=success, still subject to the stated validation limits.
output_dir is relative to the tool temp directory. device supports cpu or cuda:0 (CPU fallback).
fps is feature sampling rate (DAD default 20), threshold is a score cutoff (default 0.5),
mc_samples is Bayesian sample count (2..100), seed controls repeatability.
Returns per-frame accident scores, aleatoric/epistemic uncertainty, threshold crossing time,
CSV curve, PNG visualization, summary path, and reliability metadata.
Scores are not calibrated probabilities; crossing time is not accident time or lead time.
Explain findings and limitations first; leave output paths in the tool result.
"""


def _load_predict():
    spec = importlib.util.spec_from_file_location('ustring_infer', MODEL_ROOT / 'infer.py')
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load {MODEL_ROOT / "infer.py"}')
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f'UString requires {exc.name}; install: pip install -r model/UString/requirements.txt'
        ) from exc
    return module.predict


def anticipate_traffic_accident(
    features_path: Optional[str] = None,
    output_dir: str = 'traffic_accident_anticipation',
    weights_path: Optional[str] = None,
    device: str = 'cuda:0',
    fps: float = 20.0,
    threshold: float = 0.5,
    mc_samples: int = 10,
    seed: int = 0,
    allow_legacy_checkpoint: bool = False,
    video_path: Optional[str] = None,
    start_time: float = 0.0,
    max_frames: int = 100,
    feature_batch_size: int = 4,
    detector_weights: Optional[str] = None,
    vgg_weights: Optional[str] = None,
) -> dict:
    if bool(features_path) == bool(video_path):
        raise ValueError('Supply exactly one of video_path or features_path.')
    root = TEMP_DIR.resolve()
    out_dir = (root / output_dir).resolve()
    if Path(output_dir).is_absolute() or not out_dir.is_relative_to(root):
        raise ValueError('output_dir must be a relative directory inside the tool temp directory.')
    result = _load_predict()(
        ({'video': video_path} if video_path else {'features': features_path}), weights_path=weights_path, output_dir=str(out_dir),
        device=device, fps=fps, threshold=threshold, mc_samples=mc_samples, seed=seed,
        allow_legacy_checkpoint=allow_legacy_checkpoint, start_time=start_time,
        max_frames=max_frames, feature_batch_size=feature_batch_size,
        detector_weights=detector_weights, vgg_weights=vgg_weights,
    )
    result['task'] = 'traffic_accident_anticipation'
    summary_path = out_dir / 'summary.json'
    result['summary_path'] = str(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return result


_inference_lock = asyncio.Lock()


@wraps(anticipate_traffic_accident)
async def _mcp_anticipate(*args, **kwargs):
    # Keep inference off the MCP event loop and serialize global RNG/device work.
    async with _inference_lock:
        return await asyncio.to_thread(anticipate_traffic_accident, *args, **kwargs)


if mcp is not None:
    mcp.tool(description=TOOL_DESCRIPTION)(_mcp_anticipate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--temp_dir', default=str(TEMP_DIR))
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--features')
    group.add_argument('--video')
    parser.add_argument('--start-time', type=float, default=0.)
    parser.add_argument('--max-frames', type=int, default=100)
    parser.add_argument('--feature-batch-size', type=int, default=4)
    parser.add_argument('--detector-weights')
    parser.add_argument('--vgg-weights')
    parser.add_argument('--output', default='traffic_accident_anticipation')
    parser.add_argument('--weights')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fps', type=float, default=20)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--mc-samples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--allow-legacy-checkpoint', action='store_true')
    args = parser.parse_args()
    TEMP_DIR = Path(args.temp_dir)
    if args.features or args.video:
        print(json.dumps(anticipate_traffic_accident(
            args.features, args.output, args.weights, args.device, args.fps,
            args.threshold, args.mc_samples, args.seed, args.allow_legacy_checkpoint,
            video_path=args.video, start_time=args.start_time, max_frames=args.max_frames,
            feature_batch_size=args.feature_batch_size, detector_weights=args.detector_weights,
            vgg_weights=args.vgg_weights,
        ), indent=2, ensure_ascii=False))
    else:
        if mcp is None:
            raise FASTMCP_IMPORT_ERROR
        mcp.run()
