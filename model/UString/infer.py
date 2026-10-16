"""Unified UString inference for raw video or precomputed DAD/VGG16 features."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def local_module(name, path, package=False):
    key = '_ustring_' + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / path,
            submodule_search_locations=[str(ROOT / 'model')] if package else None)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(key, None)
            raise
    return sys.modules[key]


def load_features(path):
    if Path(path).suffix.lower() != '.npz':
        raise ValueError('Features must be a .npz file, not raw video; use input_paths["video"].')
    with np.load(path, allow_pickle=False) as data:
        if not {'data', 'det'} <= set(data.files):
            raise ValueError('NPZ requires data and det arrays.')
        features = np.asarray(data['data'], dtype=np.float32)
        boxes = np.asarray(data['det'], dtype=np.float32)
    if features.ndim != 3 or features.shape[1:] != (20, 4096) or not 1 <= len(features) <= 100:
        raise ValueError('data must be (T,20,4096), 1 <= T <= 100.')
    if boxes.shape not in ((len(features), 19, 4), (len(features), 19, 6)):
        raise ValueError('det must be (T,19,4 or 6).')
    if not np.isfinite(features).all() or not np.isfinite(boxes).all():
        raise ValueError('Inputs must be finite.')
    if (boxes[..., 2:4] < boxes[..., :2]).any():
        raise ValueError('Boxes require x2>=x1 and y2>=y1.')
    edges = np.array([(i, j) for i in range(19) for j in range(i + 1, 19)]).T
    centers = (boxes[..., :2].astype(np.float64) + boxes[..., 2:4]) / 2
    weights = np.exp(-((centers[:, edges[0]] - centers[:, edges[1]]) ** 2).sum(-1)).astype(np.float32)
    totals = weights.sum(-1, keepdims=True)
    weights = np.divide(weights, totals, out=np.ones_like(weights), where=totals > 0)
    return features, edges, weights


def predict(input_paths: dict, weights_path: str = None, output_dir: str = './output',
            device: str = 'cuda:0', **kwargs) -> dict:
    fps = float(kwargs.get('fps', 20.))
    threshold = float(kwargs.get('threshold', .5))
    samples = kwargs.get('mc_samples', 10)
    seed = kwargs.get('seed', 0)
    legacy = kwargs.get('allow_legacy_checkpoint', False)
    start = float(kwargs.get('start_time', 0.))
    count = kwargs.get('max_frames', 100)
    batch = kwargs.get('feature_batch_size', 4)
    if not np.isfinite(fps) or fps <= 0 or not 0 <= threshold <= 1:
        raise ValueError('fps must be positive and finite; threshold must be within [0,1].')
    for name, value, low, high in (('mc_samples', samples, 2, 100), ('max_frames', count, 1, 100),
                                  ('seed', seed, 0, 2**32 - 1), ('feature_batch_size', batch, 1, 20)):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer within [{low},{high}].')
    if not np.isfinite(start) or start < 0 or not isinstance(legacy, bool):
        raise ValueError('start_time must be finite and nonnegative; allow_legacy_checkpoint must be boolean.')
    if len(input_paths) != 1 or not set(input_paths) <= {'video', 'features'}:
        raise ValueError('Supply exactly one of input_paths.video or input_paths.features.')
    source = Path(next(iter(input_paths.values())))
    if not source.is_file():
        raise FileNotFoundError(source)
    requested_device = device
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    target = torch.device(device)
    weights = Path(weights_path) if weights_path else ROOT / 'weights/ustring/final_model_dad_vgg16-new.pth'
    network = local_module('network', 'model/__init__.py', package=True)
    devices = [target.index if target.index is not None else torch.cuda.current_device()] if target.type == 'cuda' else []
    out_dir = Path(output_dir).resolve()
    video_meta = {}
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            with torch.cuda.device(target):
                torch.cuda.manual_seed(seed)
        model = network.UString(4096, 256, 256, with_saa=False)
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
        state = checkpoint.get('model', checkpoint)
        state = {k: v for k, v in state.items() if not k.startswith(('predictor_aux.', 'self_aggregation.'))}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith('rnn.') for k in missing):
            raise ValueError(f'Incompatible checkpoint: missing={missing}, unexpected={unexpected}')
        if missing and not legacy:
            raise ValueError('Checkpoint is missing recurrent rnn parameters. Supply complete weights or set '
                             'allow_legacy_checkpoint=True for experimental inference with random recurrence.')
        feature_path = source
        if 'video' in input_paths:
            detector_path = Path(kwargs.get('detector_weights') or ROOT / 'weights/cascade_kitti.pth')
            vgg_path = Path(kwargs.get('vgg_weights') or ROOT / 'weights/vgg16-397923af.pth')
            for path in (detector_path, vgg_path):
                if not path.is_file():
                    raise FileNotFoundError(f'Missing video model weights: {path}')
            detector_module = local_module('detector', 'model/detector.py')
            video_module = local_module('video', 'video_features.py')
            detector = detector_module.CascadeDetector(detector_path).to(target).eval()
            extractor = video_module.VGGFeatures(vgg_path).to(target).eval()
            feature_path, video_meta = video_module.extract_video(
                source, detector, extractor, target, out_dir, fps, start, count, seed, batch)
            video_meta.update(detector_weights=str(detector_path), vgg_weights=str(vgg_path))
            del detector, extractor
            if target.type == 'cuda':
                torch.cuda.empty_cache()
        features, edges, edge_weights = load_features(feature_path)
        with np.load(feature_path, allow_pickle=False) as data:
            effective_fps = float(data['fps']) if 'fps' in data else fps
            times = np.asarray(data['timestamps_s'], dtype=np.float64) if 'timestamps_s' in data else np.arange(len(features)) / effective_fps
        if not np.isfinite(effective_fps) or effective_fps <= 0:
            raise ValueError('Cached fps must be positive and finite.')
        if times.shape != (len(features),) or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError('timestamps_s must be finite, strictly increasing, and match frame count.')
        # Reset sampling so cached features and video paths use the same Bayesian draws.
        torch.random.default_generator.manual_seed(seed)
        if devices:
            with torch.cuda.device(target):
                torch.cuda.manual_seed(seed)
        model.to(target).eval()
        with torch.inference_mode():
            values = model(torch.from_numpy(features).to(target), torch.from_numpy(edges).to(target),
                           torch.from_numpy(edge_weights).to(target), samples).cpu().numpy()
    if not np.isfinite(values).all():
        raise RuntimeError('Model produced non-finite outputs.')
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_path = out_dir / (source.stem + '_accident_curve.csv')
    np.savetxt(curve_path, np.column_stack((times, values)), delimiter=',',
               header='time_s,accident_score,aleatoric_uncertainty,epistemic_uncertainty', comments='')
    os.environ.setdefault('MPLCONFIGDIR', str(out_dir / '.matplotlib'))
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(9, 5), tight_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 1, sharex=True)
    axes[0].plot(times, values[:, 0], label='Accident score')
    axes[0].axhline(threshold, linestyle='--', color='gray')
    axes[0].set_ylim(0, 1)
    axes[0].set_title('EXPERIMENTAL: missing recurrent weights' if missing else 'UString (unvalidated)')
    axes[0].legend()
    axes[1].plot(times, values[:, 1], label='Aleatoric')
    axes[1].plot(times, values[:, 2], label='Epistemic')
    axes[1].set_xlabel('Time in source video (s)')
    axes[1].legend()
    vis_path = out_dir / (source.stem + '_accident_vis.png')
    figure.savefig(vis_path)
    crossing = np.flatnonzero(values[:, 0] >= threshold)
    return {
        'model_name': 'UString', 'disaster_type': 'traffic_accident', 'task_type': 'anticipation',
        'status': 'experimental' if missing else 'success',
        'outputs': {'frame_count': len(values), 'timestamps_s': times.tolist(),
                    'accident_scores': values[:, 0].tolist(), 'aleatoric_uncertainty': values[:, 1].tolist(),
                    'epistemic_uncertainty': values[:, 2].tolist(), 'max_accident_score': float(values[:, 0].max()),
                    'threshold': threshold, 'threshold_exceeded': bool(len(crossing)),
                    'first_threshold_time_s': float(times[crossing[0]]) if len(crossing) else None,
                    'curve_path': str(curve_path), 'vis_path': str(vis_path),
                    **{k: video_meta[k] for k in ('feature_path', 'preview_path') if k in video_meta}},
        'meta': {'weights': str(weights), 'device': str(target), 'requested_device': requested_device,
                 'fps': effective_fps, 'mc_samples': samples, 'seed': seed,
                 'missing_checkpoint_keys': missing, 'video': video_meta, 'resolution_m': None,
                 'input_spec': 'Dashcam video or DAD/VGG16 NPZ: data=(T,20,4096), det=(T,19,4 or 6).'},
        'reliability': {'metric': {}, 'level': 'unvalidated', 'note':
            ('Missing recurrent weights; seeded random recurrence. Experimental only. ' if missing else '') +
            'Video detector port has not been benchmarked against the original runtime. '
            'Scores are not calibrated probabilities; threshold time is not accident time or lead time.'},
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video')
    group.add_argument('--features')
    parser.add_argument('--weights')
    parser.add_argument('--output', default=str(ROOT / 'output'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fps', type=float, default=20.)
    parser.add_argument('--start-time', type=float, default=0.)
    parser.add_argument('--max-frames', type=int, default=100)
    parser.add_argument('--mc-samples', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--feature-batch-size', type=int, default=4)
    parser.add_argument('--detector-weights')
    parser.add_argument('--vgg-weights')
    parser.add_argument('--allow-legacy-checkpoint', action='store_true')
    options = vars(parser.parse_args())
    inputs = {'video': options.pop('video')} if options['video'] else {'features': options['features']}
    options.pop('video', None)
    options.pop('features', None)
    options['weights_path'] = options.pop('weights')
    options['output_dir'] = options.pop('output')
    print(json.dumps(predict(inputs, **options), indent=2, ensure_ascii=False))
