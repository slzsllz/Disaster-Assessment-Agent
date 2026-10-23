"""Unified UString inference for raw video or precomputed DAD/VGG16 features."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent

# Released files already bundled with this project. Match content, never just a filename.
BUNDLED_LEGACY_CHECKPOINTS = {
    '85158182a55add46fe47cf7a4d0bcd7a316f8a6aa6f18b908d720a0e88af4b53': 'dad_vgg16-new',
    '88f82c3ec9f82945b1fdb5101928e96a517534c2e27b0e0932988fda2756ac41': 'dad_vgg16',
    '1afbf5b95eb604cc78d7a81f286d2112b9feb7445b2b48ced22c2d0001fb16d8': 'a3d_vgg16',
    'd1a5c8954f4012bf036d8efb49718bb04baaa5dc102d2730a35d0c1e7342a776': 'crash_vgg16',
}
UPSTREAM_RECURRENT_SEED = 123
MODEL_PROFILES = {
    'ccd': {'checkpoint': 'final_model_crash_vgg16.pth', 'fps': 10., 'max_frames': 50},
    'dad': {'checkpoint': 'final_model_dad_vgg16-new.pth', 'fps': 20., 'max_frames': 100},
    'a3d': {'checkpoint': 'final_model_a3d_vgg16.pth', 'fps': 20., 'max_frames': 100},
}
RELEASE_PROFILES = {'crash_vgg16': 'ccd', 'dad_vgg16-new': 'dad', 'dad_vgg16': 'dad', 'a3d_vgg16': 'a3d'}


def checkpoint_fingerprint(weights):
    digest = hashlib.sha256()
    with Path(weights).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_settings(input_paths, weights_path=None, model_profile='auto'):
    """Choose the training domain from the pipeline/metadata, never from prediction scores."""
    if model_profile not in ('auto', *MODEL_PROFILES):
        raise ValueError('model_profile must be auto, ccd, dad or a3d.')
    feature_profile = 'ccd' if 'video' in input_paths else None
    if 'features' in input_paths:
        if Path(input_paths['features']).suffix.lower() != '.npz':
            raise ValueError('Features must be a .npz file, not raw video; use input_paths["video"].')
        with np.load(input_paths['features'], allow_pickle=False) as data:
            if 'model_profile' in data:
                value = np.asarray(data['model_profile'])
                if value.shape != () or str(value.item()) not in MODEL_PROFILES:
                    raise ValueError('NPZ model_profile must be a scalar ccd, dad or a3d string.')
                feature_profile = str(value.item())
    known_weights_profile = None
    if weights_path:
        release = BUNDLED_LEGACY_CHECKPOINTS.get(checkpoint_fingerprint(weights_path))
        known_weights_profile = RELEASE_PROFILES.get(release)
    if model_profile != 'auto':
        profile, reason = model_profile, 'explicit_model_profile'
    elif known_weights_profile:
        profile, reason = known_weights_profile, 'explicit_known_weights'
    elif feature_profile:
        profile, reason = feature_profile, 'video_ccd_demo_pipeline' if 'video' in input_paths else 'feature_metadata'
    else:
        profile, reason = 'dad', 'untagged_features_legacy_dad_default'
    weights = Path(weights_path) if weights_path else ROOT / 'weights/ustring' / MODEL_PROFILES[profile]['checkpoint']
    return weights, {'requested_profile': model_profile, 'profile': profile, 'reason': reason,
                     'feature_profile': feature_profile, 'weights_overridden': bool(weights_path)}


def load_checkpoint(model, weights, network, allow_legacy=False):
    """Restore released upstream initialization without accepting arbitrary incomplete weights.

    Upstream Graph_GRU_GCN uses ordinary lists, so its parameters are neither saved
    nor optimized. Upstream main.py/demo.py initialize the model with seed 123.
    Replaying the full constructor (including layers before rnn) recreates that
    initialization in the current runtime; it does not recover trained parameters
    or establish equivalence to the original PyTorch 1.x runtime.
    """
    fingerprint = checkpoint_fingerprint(weights)
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    state = checkpoint.get('model', checkpoint)
    state = {k: v for k, v in state.items() if not k.startswith(('predictor_aux.', 'self_aggregation.'))}
    renamed = {}
    if BUNDLED_LEGACY_CHECKPOINTS.get(fingerprint) == 'dad_vgg16':
        # Earlier released DAD file predates the encoder attribute rename.
        for old, new in [('enc_layer1.weight', 'enc_gcn1.weight'), ('enc_layer2.weight', 'enc_gcn2.weight')]:
            if old in state and new not in state:
                state[new] = state.pop(old)
                renamed[old] = new
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith('rnn.') for k in missing):
        raise ValueError(f'Incompatible checkpoint: missing={missing}, unexpected={unexpected}')
    info = {'mode': 'complete', 'missing_keys': missing, 'recurrent_seed': None,
            'recurrent_parameters_source': 'checkpoint', 'checkpoint_sha256': fingerprint,
            'renamed_keys': renamed}
    if not missing:
        return info
    if allow_legacy:
        return {**info, 'mode': 'experimental', 'recurrent_parameters_source': 'inference_seed_initialization'}
    recurrent_keys = {key for key in model.state_dict() if key.startswith('rnn.')}
    if fingerprint not in BUNDLED_LEGACY_CHECKPOINTS or set(missing) != recurrent_keys:
        raise ValueError('Checkpoint is missing recurrent rnn parameters and is not a recognized bundled release. '
                         'Supply complete weights or set allow_legacy_checkpoint=True only for experimental '
                         'inference with random recurrence.')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(UPSTREAM_RECURRENT_SEED)
        reference = network.UString(4096, 256, 256, with_saa=False)
        model.rnn.load_state_dict(reference.rnn.state_dict(), strict=True)
    model.rnn.requires_grad_(False)
    return {**info, 'mode': 'upstream_legacy', 'recurrent_seed': UPSTREAM_RECURRENT_SEED,
            'recurrent_parameters_source': 'upstream_seed_initialization', 'checkpoint_sha256': fingerprint,
            'release': BUNDLED_LEGACY_CHECKPOINTS[fingerprint]}


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
    if len(input_paths) != 1 or not set(input_paths) <= {'video', 'features'}:
        raise ValueError('Supply exactly one of input_paths.video or input_paths.features.')
    source = Path(next(iter(input_paths.values())))
    if not source.is_file():
        raise FileNotFoundError(source)
    weights, selection = resolve_model_settings(input_paths, weights_path, kwargs.get('model_profile', 'auto'))
    defaults = MODEL_PROFILES[selection['profile']]
    fps = float(defaults['fps'] if kwargs.get('fps') is None else kwargs['fps'])
    threshold = float(kwargs.get('threshold', .5))
    samples = kwargs.get('mc_samples', 10)
    seed = kwargs.get('seed', 0)
    legacy = kwargs.get('allow_legacy_checkpoint', False)
    start = float(kwargs.get('start_time', 0.))
    count = defaults['max_frames'] if kwargs.get('max_frames') is None else kwargs['max_frames']
    batch = kwargs.get('feature_batch_size', 4)
    if not np.isfinite(fps) or fps <= 0 or not 0 <= threshold <= 1:
        raise ValueError('fps must be positive and finite; threshold must be within [0,1].')
    for name, value, low, high in (('mc_samples', samples, 2, 100), ('max_frames', count, 1, 100),
                                  ('seed', seed, 0, 2**32 - 1), ('feature_batch_size', batch, 1, 20)):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer within [{low},{high}].')
    if not np.isfinite(start) or start < 0 or not isinstance(legacy, bool):
        raise ValueError('start_time must be finite and nonnegative; allow_legacy_checkpoint must be boolean.')
    requested_device = device
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    target = torch.device(device)
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
        checkpoint_info = load_checkpoint(model, weights, network, legacy)
        selection['checkpoint_profile'] = RELEASE_PROFILES.get(
            BUNDLED_LEGACY_CHECKPOINTS.get(checkpoint_info['checkpoint_sha256']), 'custom')
        selection['warnings'] = []
        if selection['feature_profile'] and selection['checkpoint_profile'] not in (selection['feature_profile'], 'custom'):
            selection['warnings'].append('Feature pipeline and checkpoint training domain differ; scores may be unreliable.')
        if 'features' in input_paths and not selection['feature_profile']:
            selection['warnings'].append('Untagged features: verify their training-domain compatibility explicitly.')
        missing = checkpoint_info['missing_keys']
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
    mode = checkpoint_info['mode']
    axes[0].set_title({'experimental': 'EXPERIMENTAL: missing recurrent weights',
                       'upstream_legacy': 'UString (legacy compatibility, unvalidated)'}.get(mode, 'UString (unvalidated)'))
    axes[0].legend()
    axes[1].plot(times, values[:, 1], label='Aleatoric')
    axes[1].plot(times, values[:, 2], label='Epistemic')
    axes[1].set_xlabel('Time in source video (s)')
    axes[1].legend()
    vis_path = out_dir / (source.stem + '_accident_vis.png')
    figure.savefig(vis_path)
    video_preview = {'status': 'not_available', 'reason': 'Features-only input has no source video to render.'}
    if 'video' in input_paths:
        renderer = local_module('video_preview', 'video_preview.py')
        with np.load(feature_path, allow_pickle=False) as cached:
            boxes = np.asarray(cached['det'])
        try:
            video_preview = renderer.render_video_preview(
                video_module.iter_frames(source, effective_fps, start, len(values)), boxes, times, values,
                effective_fps, threshold, mode, out_dir / (source.stem + '_accident_preview.mp4'))
        except (RuntimeError, ValueError, OSError) as exc:
            # Keep computed scores and charts when visualization fails, with an explicit reason.
            video_preview = {'status': 'error', 'reason': str(exc)}
    crossing = np.flatnonzero(values[:, 0] >= threshold)
    return {
        'model_name': 'UString', 'disaster_type': 'traffic_accident', 'task_type': 'anticipation',
        'status': {'experimental': 'experimental', 'upstream_legacy': 'legacy_compatible'}.get(mode, 'success'),
        'assessment': {'kind': 'model_score_threshold', 'accident_presence': 'undetermined',
                       'accident_ruled_out': False,
                       'statement': '模型分数达到阈值，需结合画面复核。' if len(crossing) else '模型分数未达到阈值，不能据此排除事故。'},
        'outputs': {'frame_count': len(values), 'timestamps_s': times.tolist(),
                    'accident_scores': values[:, 0].tolist(), 'aleatoric_uncertainty': values[:, 1].tolist(),
                    'epistemic_uncertainty': values[:, 2].tolist(), 'max_accident_score': float(values[:, 0].max()),
                    'threshold': threshold, 'threshold_exceeded': bool(len(crossing)),
                    'first_threshold_time_s': float(times[crossing[0]]) if len(crossing) else None,
                    'curve_path': str(curve_path), 'vis_path': str(vis_path),
                    **({'video_path': video_preview['path']} if video_preview['status'] == 'success' else {}),
                    **{k: video_meta[k] for k in ('feature_path', 'preview_path') if k in video_meta}},
        'meta': {'weights': str(weights), 'device': str(target), 'requested_device': requested_device,
                 'model_selection': selection,
                 'fps': effective_fps, 'mc_samples': samples, 'seed': seed,
                 'missing_checkpoint_keys': missing, 'checkpoint_compatibility': checkpoint_info,
                 'video': video_meta, 'video_preview': video_preview, 'resolution_m': None,
                 'input_spec': 'Dashcam video or DAD/VGG16 NPZ: data=(T,20,4096), det=(T,19,4 or 6).'},
        'reliability': {'metric': {}, 'level': 'unvalidated', 'note':
            ({'experimental': 'Missing recurrent weights; seeded random recurrence. Experimental only. ',
              'upstream_legacy': 'Recognized upstream checkpoint: unsaved, unoptimized recurrent layers rebuilt '
                                 'with upstream seed 123. This is legacy compatibility, not recovered trained weights '
                                 'or a verified reproduction of the original PyTorch runtime. '}.get(mode, '')) +
            'Video detector port has not been benchmarked against the original runtime. '
            'Scores are not calibrated probabilities; threshold time is not accident time or lead time. '
            'A below-threshold score does not mean no accident occurred.'},
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video')
    group.add_argument('--features')
    parser.add_argument('--weights')
    parser.add_argument('--model-profile', choices=('auto', 'ccd', 'dad', 'a3d'), default='auto')
    parser.add_argument('--output', default=str(ROOT / 'output'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fps', type=float)
    parser.add_argument('--start-time', type=float, default=0.)
    parser.add_argument('--max-frames', type=int)
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
