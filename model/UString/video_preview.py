"""Render sampled video frames with UString scores, uncertainty and region proposals."""
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4

import cv2
import numpy as np


def annotate_frame(frame, boxes, times, values, index, threshold, mode):
    """Draw only observed scores up to this frame; ROI boxes are not accident locations."""
    height, width = frame.shape[:2]
    canvas_width = max(640, min(1280, width)) // 2 * 2
    scale = min(canvas_width / width, 720 / height)
    image_width, image_height = max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)
    # Use the actual rounded image scales for spatial overlays.
    sx, sy = image_width / width, image_height / height
    header, footer = 76, 178
    left = (canvas_width - image_width) // 2
    image = cv2.resize(frame, (image_width, image_height))
    canvas = np.full((header + image_height + footer, canvas_width, 3), (30, 24, 20), dtype=np.uint8)
    canvas[header:header + image_height, left:left + image_width] = image
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = min(1.3, canvas_width / 640)

    def text(value, x, y, color=(240, 240, 240), factor=.5):
        cv2.putText(canvas, value, (int(x), int(y)), font, factor * size, color, 1, cv2.LINE_AA)

    label = {'upstream_legacy': 'LEGACY COMPATIBILITY', 'experimental': 'EXPERIMENTAL'}.get(mode, 'MODEL OUTPUT')
    text(f'UString | {label} | UNVALIDATED', 12, 25, (90, 210, 255))
    text(f'Source t={times[index]:.3f}s | sampled clip {times[0]:.3f}-{times[-1]:.3f}s', 12, 49)
    text('ROI proposals only; not accident localization', 12, 68, (255, 220, 100), .4)

    unique = set()
    for box in boxes:
        if len(box) >= 6 and box[5] < 0:  # Full-frame fallback when no object was detected.
            continue
        xyxy = tuple(np.round(box[:4]).astype(int))
        if xyxy in unique:
            continue
        unique.add(xyxy)
        x1, y1, x2, y2 = xyxy
        x1, x2 = np.clip([x1, x2], 0, width - 1)
        y1, y2 = np.clip([y1, y2], 0, height - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(canvas, (left + int(x1*sx), header + int(y1*sy)),
                      (left + int(x2*sx), header + int(y2*sy)), (255, 220, 100), 2)

    bottom = header + image_height
    score, aleatoric, epistemic = values[index]
    color = (100, 170, 255) if score >= threshold else (150, 230, 150)
    text(f'Score {score:.4g} | cutoff {threshold:.3f}', 12, bottom + 25, color)
    text(f'Aleatoric {aleatoric:.3g} | Epistemic {epistemic:.3g}', 12, bottom + 49)
    x0, x1, y0, y1 = 30, canvas_width - 20, bottom + 66, bottom + 130
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (180, 180, 180), 1)
    cutoff_y = int(round(y1 - threshold * (y1-y0)))
    for x in range(x0, x1, 12):
        cv2.line(canvas, (x, cutoff_y), (min(x+6, x1), cutoff_y), (100, 100, 230), 1)
    span = max(float(times[-1] - times[0]), .001)
    points = np.array([(round(x0+(t-times[0])/span*(x1-x0)), round(y1-float(v[0])*(y1-y0)))
                       for t, v in zip(times[:index+1], values[:index+1])], dtype=np.int32)
    if len(points) > 1:
        cv2.polylines(canvas, [points], False, (255, 180, 80), 2, cv2.LINE_AA)
    cv2.circle(canvas, tuple(points[-1]), 4, (70, 220, 255), -1)
    text('Accident score is not a calibrated probability | sampled clip only', 12, bottom + 156,
         (200, 200, 200), .4)
    return canvas


def render_video_preview(frames, boxes, times, values, fps, threshold, mode, destination):
    executable = shutil.which('ffmpeg')
    if not executable:
        raise RuntimeError('FFmpeg is required to generate the processed preview video.')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temporary = destination.with_name(f'.{destination.stem}.{token}.avi')
    encoded = destination.with_name(f'.{destination.stem}.{token}.partial.mp4')
    writer, count = None, 0
    try:
        for index, (timestamp, frame, _) in enumerate(frames):
            if index >= len(times) or not np.isclose(timestamp, times[index], rtol=0, atol=1e-7):
                raise RuntimeError('Video frames no longer match the analyzed timestamps.')
            rendered = annotate_frame(frame, boxes[index], times, values, index, threshold, mode)
            if writer is None:
                writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'MJPG'), float(fps),
                                         (rendered.shape[1], rendered.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError('Cannot create the intermediate preview video.')
            writer.write(rendered)
            count += 1
        if count != len(times):
            raise RuntimeError('Video ended before all analyzed frames could be rendered.')
        writer.release()
        writer = None
        try:
            subprocess.run([executable, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                            '-threads', '2', '-filter_threads', '2', '-i', str(temporary), '-an',
                            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
                            '-movflags', '+faststart', '-threads', '2', str(encoded)],
                           check=True, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except (OSError, subprocess.SubprocessError):
            raise RuntimeError('Failed to encode the processed preview as H.264 MP4. Check FFmpeg/libx264.') from None
        if not encoded.is_file() or encoded.stat().st_size == 0:
            raise RuntimeError('Preview encoder produced no video.')
        encoded.replace(destination)
    finally:
        if writer is not None:
            writer.release()
        temporary.unlink(missing_ok=True)
        encoded.unlink(missing_ok=True)
    return {'status': 'success', 'path': str(destination), 'frame_count': count, 'fps': float(fps),
            'start_time_s': float(times[0]), 'last_sample_time_s': float(times[-1]),
            'audio': False, 'codec': 'h264', 'scope': 'Analyzed sampled frames only; not the entire source video.',
            'boxes': 'Feature region proposals, not accident localization.'}
