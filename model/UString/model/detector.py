"""Inference-only KITTI Cascade R-CNN adapter using torchvision operators.

Implements the bundled legacy config, including inclusive box coordinates,
background-first logits, legacy ROIAlign coordinates, and class-wise zero-IoU NMS.
This port requires accuracy validation against the original MMDetection runtime.
"""
import math

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import ResNet, Bottleneck
from torchvision.ops import nms, roi_align


def decode(boxes, deltas, stds, image_shape):
    delta = deltas * deltas.new_tensor(stds)
    size = boxes[:, 2:] - boxes[:, :2] + 1
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    new_center = center + delta[:, :2] * size
    new_size = size * delta[:, 2:].clamp(-math.log(1000 / 16), math.log(1000 / 16)).exp()
    result = torch.cat((new_center - (new_size - 1) / 2, new_center + (new_size - 1) / 2), 1)
    result[:, 0::2].clamp_(0, image_shape[1] - 1)
    result[:, 1::2].clamp_(0, image_shape[0] - 1)
    return result


def inclusive_nms(boxes, scores, threshold):
    continuous = boxes.clone()
    continuous[:, 2:] += 1
    return nms(continuous, scores, threshold)


def anchors(height, width, stride, device):
    ratios = torch.tensor([0.33, 0.5, 1., 2., 3.], device=device).sqrt()
    sizes = torch.stack((stride * 8 / ratios, stride * 8 * ratios), 1)
    center = (stride - 1) / 2
    base = torch.cat((center - (sizes - 1) / 2, center + (sizes - 1) / 2), 1).round()
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    shifts = torch.stack((x, y, x, y), -1).reshape(-1, 4) * stride
    return (shifts[:, None] + base[None]).reshape(-1, 4)


class CascadeDetector(nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        self.backbone = ResNet(Bottleneck, [3, 4, 23, 3], groups=64, width_per_group=4)
        self.backbone.fc = nn.Identity()
        self.laterals = nn.ModuleList([nn.Conv2d(c, 256, 1) for c in (256, 512, 1024, 2048)])
        self.fpn = nn.ModuleList([nn.Conv2d(256, 256, 3, padding=1) for _ in range(4)])
        self.rpn_conv = nn.Conv2d(256, 256, 3, padding=1)
        self.rpn_cls = nn.Conv2d(256, 5, 1)
        self.rpn_reg = nn.Conv2d(256, 20, 1)
        self.heads = nn.ModuleList([nn.ModuleDict({
            'fc1': nn.Linear(256 * 7 * 7, 1024), 'fc2': nn.Linear(1024, 1024),
            'cls': nn.Linear(1024, 4), 'reg': nn.Linear(1024, 4),
        }) for _ in range(3)])
        state = torch.load(weights_path, map_location='cpu', weights_only=True)
        state = state.get('state_dict', state)
        mapped = {}
        for name, value in state.items():
            key = name
            for i in range(4):
                key = key.replace(f'neck.lateral_convs.{i}.conv.', f'laterals.{i}.')
                key = key.replace(f'neck.fpn_convs.{i}.conv.', f'fpn.{i}.')
            key = key.replace('rpn_head.', '')
            for i in range(3):
                for old, new in (('shared_fcs.0', 'fc1'), ('shared_fcs.1', 'fc2'), ('fc_cls', 'cls'), ('fc_reg', 'reg')):
                    key = key.replace(f'bbox_head.{i}.{old}.', f'heads.{i}.{new}.')
            mapped[key] = value
        self.load_state_dict(mapped, strict=True)
        self.eval()

    def pyramid(self, x):
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        features = []
        for layer in (b.layer1, b.layer2, b.layer3, b.layer4):
            x = layer(x)
            features.append(x)
        lateral = [layer(value) for layer, value in zip(self.laterals, features)]
        for i in (3, 2, 1):
            lateral[i - 1] = lateral[i - 1] + F.interpolate(lateral[i], size=lateral[i - 1].shape[-2:], mode='nearest')
        result = [layer(value) for layer, value in zip(self.fpn, lateral)]
        return result + [F.max_pool2d(result[-1], 1, stride=2)]

    def proposals(self, pyramid, shape):
        all_boxes, all_scores = [], []
        for feature, stride in zip(pyramid, (4, 8, 16, 32, 64)):
            hidden = F.relu(self.rpn_conv(feature))
            scores = self.rpn_cls(hidden).permute(0, 2, 3, 1).reshape(-1).sigmoid()
            deltas = self.rpn_reg(hidden).permute(0, 2, 3, 1).reshape(-1, 4)
            base = anchors(*feature.shape[-2:], stride, feature.device)
            top = scores.topk(min(1000, scores.numel())).indices
            boxes = decode(base[top], deltas[top], (1, 1, 1, 1), shape)
            scores = scores[top]
            kept = inclusive_nms(boxes, scores, 0.7)[:1000]
            all_boxes.append(boxes[kept])
            all_scores.append(scores[kept])
        boxes, scores = torch.cat(all_boxes), torch.cat(all_scores)
        return boxes[scores.topk(min(1000, len(scores))).indices]

    def pool(self, pyramid, boxes):
        size = (boxes[:, 2:] - boxes[:, :2] + 1).clamp(min=1)
        levels = ((size.prod(1).sqrt() / 56 + 1e-6).log2().floor()).clamp(0, 3).long()
        rois = torch.cat((boxes.new_zeros((len(boxes), 1)), boxes), 1)
        pooled = boxes.new_zeros((len(boxes), 256, 7, 7))
        for i in range(4):
            indices = torch.where(levels == i)[0]
            if len(indices):
                scale = 1 / (4 * 2**i)
                legacy_rois = rois[indices].clone()
                legacy_rois[:, 3:] += 1
                # Cancel torchvision's aligned half-pixel shift. Unlike aligned=False,
                # this preserves the legacy implementation's subpixel ROI widths.
                legacy_rois[:, 1:] += 0.5 / scale
                pooled[indices] = roi_align(pyramid[i], legacy_rois, 7,
                                            spatial_scale=scale, sampling_ratio=2, aligned=True)
        return pooled.flatten(1)

    @torch.inference_mode()
    def forward(self, frame):
        height, width = frame.shape[:2]
        scale = min(1696 / max(height, width), 512 / min(height, width))
        resized = cv2.resize(frame, (int(width * scale + 0.5), int(height * scale + 0.5)))
        shape = resized.shape[:2]
        rgb = resized[:, :, ::-1].astype(np.float32)
        rgb = (rgb - np.array([92.84, 97.80, 93.58], np.float32)) / np.array([58.395, 57.12, 57.375], np.float32)
        x = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0).to(next(self.parameters()).device)
        x = F.pad(x, (0, -shape[1] % 32, 0, -shape[0] % 32))
        pyramid = self.pyramid(x)
        boxes = self.proposals(pyramid, shape)
        logits = []
        for head, std in zip(self.heads, ((.1, .1, .2, .2), (.05, .05, .1, .1), (.033, .033, .067, .067))):
            # Chunk ROI features to bound CPU/GPU memory.
            cls, reg = [], []
            for batch in boxes.split(128):
                value = F.relu(head['fc1'](self.pool(pyramid, batch)))
                value = F.relu(head['fc2'](value))
                cls.append(head['cls'](value))
                reg.append(head['reg'](value))
            logits.append(torch.cat(cls))
            boxes = decode(boxes, torch.cat(reg), std, shape)
        scores = torch.stack(logits).mean(0).softmax(-1)
        boxes = boxes / scale
        detections = []
        for label in range(1, 4):
            valid = scores[:, label] > 1e-6
            candidates, confidence = boxes[valid], scores[valid, label]
            keep = inclusive_nms(candidates, confidence, 0.0)
            detections.append(torch.cat((candidates[keep], confidence[keep, None],
                                         confidence.new_full((len(keep), 1), label - 1)), 1))
        result = torch.cat(detections)
        # MMDetection limits globally then bbox2result groups by class.
        if len(result) > 100:
            result = result[result[:, 4].topk(100).indices]
        result = result[torch.argsort(result[:, 5], stable=True)]
        return result.cpu().numpy()
