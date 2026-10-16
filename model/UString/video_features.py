"""Video decoding, KITTI region proposals and original VGG16 feature pipeline."""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms


def sample_boxes(detections, shape, rng):
    height, width = shape[:2]
    boxes = []
    for box in detections:
        x1, y1 = np.clip(box[:2].astype(int), [0, 0], [width - 1, height - 1])
        x2, y2 = np.clip(box[2:4].astype(int), [x1 + 1, y1 + 1], [width, height])
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            boxes.append([x1, y1, x2, y2, float(box[4]), float(box[5])])
    if not boxes:
        boxes = [[0, 0, width, height, 1., -1]]
    boxes = np.asarray(boxes, dtype=np.float32)
    if len(boxes) < 19:
        count = min(5, len(boxes))
        extra = rng.choice(count, 19 - len(boxes), replace=len(boxes) <= 19 - count)
        boxes = np.concatenate((boxes, boxes[extra]))
    return boxes[:19]


def iter_frames(path, sample_fps, start_time=0., max_frames=100):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(source_fps) or source_fps <= 0:
        cap.release()
        raise ValueError('Video has no valid FPS metadata.')
    effective_fps = min(sample_fps, source_fps)
    first = int(np.ceil(start_time * source_fps))
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    index, emitted = first, 0
    try:
        while emitted < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            target = first + int(round(emitted * source_fps / effective_fps))
            if index >= target:
                yield index / source_fps, frame, effective_fps
                emitted += 1
            index += 1
    finally:
        cap.release()


class VGGFeatures(torch.nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        net = models.vgg16(weights=None)
        net.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True), strict=True)
        self.features = net.features
        self.classifier = torch.nn.Sequential(*list(net.classifier.children())[:-3])
        self.transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        ])  # Original demo deliberately does not apply ImageNet Normalize.
        self.eval()

    def forward(self, images):
        return self.classifier(self.features(images).flatten(1))


@torch.inference_mode()
def extract_video(path, detector, extractor, device, output_dir, fps=20.,
                  start_time=0., max_frames=100, seed=0, feature_batch_size=4):
    rng = np.random.RandomState(seed)
    features, detections, timestamps = [], [], []
    first_frame = None
    effective_fps = fps
    for timestamp, frame, effective_fps in iter_frames(path, fps, start_time, max_frames):
        if first_frame is None:
            first_frame = frame.copy()
        boxes = sample_boxes(detector(frame), frame.shape, rng)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crops = [rgb] + [rgb[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in boxes]
        values = []
        for offset in range(0, len(crops), feature_batch_size):
            batch = torch.stack([extractor.transform(Image.fromarray(c))
                                 for c in crops[offset:offset + feature_batch_size]]).to(device)
            values.append(extractor(batch).cpu().numpy())
        features.append(np.concatenate(values))
        detections.append(boxes)
        timestamps.append(timestamp)
    if not features:
        raise ValueError('No decodable frames at the requested start_time.')
    path_out = Path(output_dir) / (Path(path).stem + '_features.npz')
    path_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path_out, data=np.asarray(features), det=np.asarray(detections),
                        timestamps_s=np.asarray(timestamps), fps=effective_fps)
    preview = Path(output_dir) / (Path(path).stem + '_preview.jpg')
    for box in detections[0]:
        cv2.rectangle(first_frame, tuple(box[:2].astype(int)), tuple(box[2:4].astype(int)), (0, 255, 0), 1)
    if not cv2.imwrite(str(preview), first_frame):
        raise RuntimeError(f'Failed to write {preview}')
    return path_out, {'source_video': str(path), 'feature_path': str(path_out),
                      'preview_path': str(preview), 'effective_fps': effective_fps,
                      'start_time_s': timestamps[0], 'end_time_s': timestamps[-1],
                      'max_frames': max_frames, 'scope': 'Requested clip only; not necessarily the full video.',
                      'detector_backend': 'torchvision port of legacy KITTI Cascade R-CNN; accuracy unvalidated'}
