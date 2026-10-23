"""Numerical and integration checks; synthetic features are not accuracy tests."""
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


infer = load('ustring_infer_test', 'model/UString/infer.py')
network = infer.local_module('network', 'model/__init__.py', package=True)
tool = load('ustring_tool_test', 'agent/tools/TrafficAccidentAnticipation.py')


class UStringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.features = self.root / 'clip.npz'
        np.savez(self.features, data=np.zeros((3, 20, 4096), np.float32),
                 det=np.zeros((3, 19, 6), np.float32))

    def test_graph_matches_dense_upstream_equation(self):
        layer = network.GCNConv(4, 3, act=lambda x: x, bias=True)
        x = torch.randn(3, 4)
        edges = torch.tensor([[0, 0, 1], [1, 2, 2]])
        weights = torch.tensor([0.2, 0.3, 0.5])
        adjacency = torch.eye(3)
        adjacency[edges[0], edges[1]] = weights
        inv = adjacency.sum(1).rsqrt()
        normalized = inv[:, None] * adjacency * inv[None, :]
        expected = normalized @ (x @ layer.weight) + layer.bias
        torch.testing.assert_close(layer(x.unsqueeze(0), edges.unsqueeze(0), weights.unsqueeze(0))[0], expected)

    def test_input_validation_and_graph(self):
        features, edges, weights = infer.load_features(self.features)
        self.assertEqual(edges.shape, (2, 171))
        np.testing.assert_allclose(weights.sum(-1, dtype=np.float64), 1, rtol=1e-6)
        with self.assertRaisesRegex(ValueError, 'not raw video'):
            infer.load_features('clip.mp4')
        np.savez(self.features, data=features, det=np.zeros((2, 19, 6)))
        with self.assertRaisesRegex(ValueError, 'det must'):
            infer.load_features(self.features)
        features[0, 0, 0] = np.nan
        np.savez(self.features, data=features, det=np.zeros((3, 19, 6)))
        with self.assertRaisesRegex(ValueError, 'finite'):
            infer.load_features(self.features)

    def test_bundled_checkpoint_works_without_experimental_opt_in(self):
        result = infer.predict({'features': str(self.features)}, device='cpu', output_dir=str(self.root), mc_samples=2)
        self.assertEqual(result['status'], 'legacy_compatible')
        info = result['meta']['checkpoint_compatibility']
        self.assertEqual(info['mode'], 'upstream_legacy')
        self.assertEqual(info['recurrent_seed'], 123)
        self.assertIn(info['checkpoint_sha256'], infer.BUNDLED_LEGACY_CHECKPOINTS)
        self.assertEqual(result['reliability']['level'], 'unvalidated')
        self.assertTrue(result['meta']['missing_checkpoint_keys'])
        self.assertEqual(result['meta']['video_preview']['status'],'not_available')
        self.assertNotIn('video_path',result['outputs'])

    def test_raw_video_selection_uses_upstream_ccd_pipeline(self):
        weights, selection = infer.resolve_model_settings({'video':'dad_named_but_raw.mp4'})
        self.assertEqual(weights.name,'final_model_crash_vgg16.pth')
        self.assertEqual(selection['profile'],'ccd')
        self.assertEqual(selection['reason'],'video_ccd_demo_pipeline')
        self.assertEqual(selection['feature_profile'],'ccd')

    def test_tagged_features_reuse_ccd_while_untagged_features_remain_dad(self):
        options=dict(device='cpu',output_dir=str(self.root),mc_samples=2)
        old=infer.predict({'features':str(self.features)},**options)
        self.assertEqual(old['meta']['model_selection']['checkpoint_profile'],'dad')
        np.savez(self.features,data=np.zeros((3,20,4096),np.float32),det=np.zeros((3,19,6),np.float32),
                 model_profile=np.asarray('ccd'),fps=10.)
        tagged=infer.predict({'features':str(self.features)},**options)
        explicit=infer.predict({'features':str(self.features)},model_profile='ccd',**options)
        self.assertEqual(tagged['meta']['model_selection']['reason'],'feature_metadata')
        self.assertEqual(tagged['meta']['model_selection']['checkpoint_profile'],'ccd')
        self.assertEqual(tagged['outputs']['timestamps_s'],[0,.1,.2])
        self.assertEqual(tagged['outputs']['accident_scores'],explicit['outputs']['accident_scores'])

    def test_explicit_weights_and_profiles_are_honored(self):
        dad=infer.ROOT/'weights/ustring/final_model_dad_vgg16-new.pth'
        weights, selection=infer.resolve_model_settings({'video':'clip.mp4'},str(dad))
        self.assertEqual(weights,dad)
        self.assertEqual(selection['profile'],'dad')
        self.assertEqual(selection['reason'],'explicit_known_weights')
        weights, selection=infer.resolve_model_settings({'features':str(self.features)},model_profile='a3d')
        self.assertEqual(weights.name,'final_model_a3d_vgg16.pth')
        self.assertEqual(selection['reason'],'explicit_model_profile')
        custom=self.root/'custom.pth';custom.write_bytes(b'custom model')
        weights, selection=infer.resolve_model_settings({'video':'clip.mp4'},str(custom))
        self.assertEqual(weights,custom)
        self.assertTrue(selection['weights_overridden'])
        with self.assertRaisesRegex(ValueError,'model_profile'):
            infer.resolve_model_settings({'video':'clip.mp4'},model_profile='pick_highest_score')

    def test_score_threshold_never_becomes_an_accident_presence_verdict(self):
        for score in (.01,.99):
            values=torch.tensor([[score,.1,.001]]*3)
            with patch.object(network.UString,'forward',return_value=values):
                result=infer.predict({'features':str(self.features)},device='cpu',output_dir=str(self.root),mc_samples=2)
            self.assertEqual(result['outputs']['threshold_exceeded'],score>=.5)
            self.assertEqual(result['assessment']['accident_presence'],'undetermined')
            self.assertFalse(result['assessment']['accident_ruled_out'])

    def test_unknown_incomplete_checkpoint_is_still_rejected(self):
        model = network.UString(4096, 256, 256, with_saa=False)
        weights = self.root / 'final_model_dad_vgg16-new.pth'
        torch.save({'model': {k: v for k, v in model.state_dict().items() if not k.startswith('rnn.')}}, weights)
        with self.assertRaisesRegex(ValueError, 'missing recurrent'):
            infer.predict({'features': str(self.features)}, weights_path=str(weights), device='cpu')

    def test_upstream_recurrence_matches_independent_initialization_and_ignores_sampling_seed(self):
        # Reference: official Models.py UString.__init__, Graph_GRU_GCN.__init__, utils.glorot.
        # Plain lists in the reference deliberately keep these layers out of state_dict/optimizer.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            torch.nn.Linear(4096, 256)  # phi_x is constructed before the two encoders and GRU.
            for _ in range(2):
                bound = math.sqrt(6 / (512 + 256))
                torch.empty(512, 256).uniform_(-bound, bound)
            reference = {}
            for name, inputs in [('xz',768), ('hz',256), ('xr',768), ('hr',256), ('xh',768), ('hh',256)]:
                bound = math.sqrt(6 / (inputs + 256))
                reference[f'weight_{name}.0.weight'] = torch.empty(inputs, 256).uniform_(-bound, bound)
                reference[f'weight_{name}.0.bias'] = torch.zeros(256)
            for seed in (0, 42):
                torch.manual_seed(seed)
                model = network.UString(4096,256,256,with_saa=False)
                before = torch.random.get_rng_state().clone()
                info = infer.load_checkpoint(model, infer.ROOT/'weights/ustring/final_model_dad_vgg16-new.pth', network)
                torch.testing.assert_close(torch.random.get_rng_state(), before)
                self.assertEqual(info['recurrent_seed'],123)
                for key, tensor in model.rnn.state_dict().items():
                    torch.testing.assert_close(tensor, reference[key], rtol=0, atol=0)

    def test_all_bundled_releases_are_recognized_by_content(self):
        for weights in sorted((infer.ROOT/'weights/ustring').glob('*.pth')):
            with self.subTest(weights=weights.name):
                model = network.UString(4096,256,256,with_saa=False)
                info = infer.load_checkpoint(model, weights, network)
                self.assertEqual(info['mode'],'upstream_legacy')
                if weights.name == 'final_model_dad_vgg16.pth':
                    saved = torch.load(weights,map_location='cpu',weights_only=False)['model']
                    torch.testing.assert_close(model.enc_gcn1.weight,saved['enc_layer1.weight'])
                    self.assertEqual(info['renamed_keys']['enc_layer1.weight'],'enc_gcn1.weight')

    def test_experimental_tool_outputs_and_repeatability(self):
        original = tool.TEMP_DIR
        self.addCleanup(setattr, tool, 'TEMP_DIR', original)
        tool.TEMP_DIR = self.root
        options = dict(features_path=str(self.features), device='cpu', mc_samples=2,
                       allow_legacy_checkpoint=True)
        state = torch.random.get_rng_state().clone()
        result = tool.anticipate_traffic_accident(**options)
        torch.testing.assert_close(torch.random.get_rng_state(), state)
        again = tool.anticipate_traffic_accident(**options)
        self.assertEqual(result['outputs']['accident_scores'], again['outputs']['accident_scores'])
        self.assertEqual(result['status'], 'experimental')
        self.assertTrue(result['meta']['missing_checkpoint_keys'])
        self.assertEqual(json.loads(Path(result['summary_path']).read_text()), result)
        for key in ('curve_path', 'vis_path'):
            self.assertTrue(Path(result['outputs'][key]).is_file())
        self.assertEqual(len(result['outputs']['accident_scores']), 3)
        self.assertEqual(result['outputs']['timestamps_s'], [0, 0.05, 0.1])
        for key in ('accident_scores', 'aleatoric_uncertainty', 'epistemic_uncertainty'):
            self.assertTrue(np.isfinite(result['outputs'][key]).all())
        with self.assertRaisesRegex(ValueError, 'relative directory'):
            tool.anticipate_traffic_accident(**options, output_dir='../escape')

    def test_complete_checkpoint_and_incompatible_checkpoint(self):
        # Random full checkpoint validates loading only, not model accuracy.
        weights = self.root / 'complete.pth'
        model = network.UString(4096, 256, 256, with_saa=False)
        torch.save({'model': model.state_dict()}, weights)
        result = infer.predict({'features': str(self.features)}, weights_path=str(weights),
                               device='cpu', output_dir=str(self.root), mc_samples=2)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['meta']['missing_checkpoint_keys'], [])
        self.assertEqual(result['meta']['checkpoint_compatibility']['mode'], 'complete')
        state = model.state_dict()
        del state['phi_x.0.bias']
        torch.save({'model': state}, weights)
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            infer.predict({'features': str(self.features)}, weights_path=str(weights),
                          device='cpu', allow_legacy_checkpoint=True)

    def test_configs(self):
        for name in ('config.json', 'config_deepseek.json', 'config_qwen3.json'):
            config = json.loads((ROOT / 'agent' / name).read_text())
            entry = config['mcpServers']['TrafficAccidentAnticipation']
            self.assertEqual(entry['args'][0], 'tools/TrafficAccidentAnticipation.py')



class VideoTests(unittest.TestCase):
    def test_sampling_and_empty_video(self):
        import cv2
        video = infer.local_module('video', 'video_features.py')
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'clip.avi')
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
            self.assertTrue(writer.isOpened())
            for i in range(8):
                writer.write(np.full((48, 64, 3), i * 20, np.uint8))
            writer.release()
            frames = list(video.iter_frames(path, 5, start_time=.2, max_frames=3))
            np.testing.assert_allclose([f[0] for f in frames], [.2, .4, .6])
            self.assertEqual(len(list(video.iter_frames(path, 20, max_frames=100))), 8)
            self.assertEqual(list(video.iter_frames(path, 10, start_time=5)), [])

    def test_box_sampling_and_confidence(self):
        video = infer.local_module('video', 'video_features.py')
        rng = np.random.RandomState(0)
        boxes = video.sample_boxes(np.array([[1, 2, 20, 30, .85, 0]]), (40, 50, 3), rng)
        self.assertEqual(boxes.shape, (19, 6))
        np.testing.assert_allclose(boxes[:, 4], .85)
        boxes = video.sample_boxes(np.empty((0, 6)), (40, 50, 3), rng)
        np.testing.assert_allclose(boxes[0, :4], [0, 0, 50, 40])

    def test_legacy_box_decode_and_nms(self):
        detector = infer.local_module('detector', 'model/detector.py')
        boxes = torch.tensor([[2., 3., 12., 13.], [12., 13., 20., 20.]])
        torch.testing.assert_close(detector.decode(boxes, torch.zeros_like(boxes), (1, 1, 1, 1), (100,100)), boxes)
        # Inclusive pixel boxes share the boundary pixel; at zero IoU threshold one is suppressed.
        keep = detector.inclusive_nms(boxes, torch.tensor([.9, .8]), 0.)
        self.assertEqual(keep.tolist(), [0])

    def test_mutually_exclusive_tool_inputs(self):
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            tool.anticipate_traffic_accident()
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            tool.anticipate_traffic_accident(features_path='clip.npz', video_path='clip.mp4')


class ProcessedPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.renderer = infer.local_module('video_preview','video_preview.py')
        self.video = infer.local_module('video','video_features.py')
        self.times = np.array([.2,.4,.6])
        self.values = np.array([[.1,.2,.001],[.8,.3,.002],[.4,.25,.001]])
        self.boxes = np.array([[[2,3,30,35,.8,0]]]*3)
        self.frame = np.zeros((48,64,3),np.uint8)

    def frames(self):
        return ((t,self.frame.copy(),5.) for t in self.times)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'FFmpeg required')
    def test_processed_clip_is_browser_playable_and_matches_analyzed_frames(self):
        import cv2
        path = self.root/'clip_accident_preview.mp4'
        meta = self.renderer.render_video_preview(self.frames(),self.boxes,self.times,self.values,
                                                  5.,.5,'upstream_legacy',path)
        self.assertEqual(meta['status'],'success')
        self.assertEqual(meta['start_time_s'],.2)
        self.assertEqual(meta['last_sample_time_s'],.6)
        self.assertEqual(meta['frame_count'],3)
        probe = subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                                '-show_entries','stream=codec_name,pix_fmt,nb_read_frames,r_frame_rate',
                                '-of','json',str(path)],check=True,capture_output=True)
        stream=json.loads(probe.stdout)['streams'][0]
        self.assertEqual(stream['codec_name'],'h264')
        self.assertEqual(stream['pix_fmt'],'yuv420p')
        self.assertEqual(stream['nb_read_frames'],'3')
        self.assertEqual(stream['r_frame_rate'],'5/1')
        cap=cv2.VideoCapture(str(path))
        try:
            ok,frame=cap.read()
            self.assertTrue(ok)
            self.assertGreater(frame.shape[0],self.frame.shape[0])
            self.assertGreater(np.std(frame[:76]),5)  # Rendered text, rather than an unmodified copy.
        finally: cap.release()
        self.assertFalse(list(self.root.glob('*.avi')))
        self.assertFalse(list(self.root.glob('*.partial.mp4')))

    def test_overlay_uses_only_observed_scores(self):
        first = self.renderer.annotate_frame(self.frame,self.boxes[0],self.times,self.values,0,.5,'upstream_legacy')
        future_changed = self.values.copy(); future_changed[1:] = [[.99,.1,.01],[.99,.1,.01]]
        same = self.renderer.annotate_frame(self.frame,self.boxes[0],self.times,future_changed,0,.5,'upstream_legacy')
        np.testing.assert_array_equal(first,same)
        next_frame = self.renderer.annotate_frame(self.frame,self.boxes[1],self.times,self.values,1,.5,'upstream_legacy')
        self.assertFalse(np.array_equal(first,next_frame))

    def test_preview_failure_is_explicit_and_partial_files_are_removed(self):
        path=self.root/'clip_accident_preview.mp4'
        with patch.object(self.renderer.shutil,'which',return_value=None):
            with self.assertRaisesRegex(RuntimeError,'FFmpeg'):
                self.renderer.render_video_preview(self.frames(),self.boxes,self.times,self.values,5,.5,'complete',path)
        with patch.object(self.renderer.shutil,'which',return_value='/usr/bin/ffmpeg'), \
             patch.object(self.renderer.subprocess,'run',side_effect=subprocess.CalledProcessError(1,'ffmpeg')):
            with self.assertRaisesRegex(RuntimeError,'encode'):
                self.renderer.render_video_preview(self.frames(),self.boxes,self.times,self.values,5,.5,'complete',path)
        self.assertFalse(path.exists())
        self.assertFalse(list(self.root.glob('*.avi')))


if __name__ == '__main__':
    unittest.main()
