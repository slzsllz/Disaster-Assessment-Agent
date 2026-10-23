"""On-demand browser-compatible video previews; original analysis inputs stay unchanged."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
import threading
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

VIDEO_EXTENSIONS = {'.mp4', '.m4v', '.webm', '.ogv', '.ogg', '.mov', '.avi', '.mkv', '.mpeg', '.mpg', '.3gp', '.ts', '.mts', '.m2ts'}


class VideoPreviews:
    def __init__(self, directory):
        self.directory = Path(directory)
        # One encoder at a time, shared across files and requests in this process.
        self.lock = threading.Lock()

    def prepare(self, source):
        source = Path(source)
        stat = source.stat()
        key = hashlib.sha256(f'v1:{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}'.encode()).hexdigest()
        destination = self.directory / f'{key}.mp4'
        with self.lock:
            if destination.is_file():
                return destination
            executable = shutil.which('ffmpeg')
            if not executable:
                raise HTTPException(503, '未安装 FFmpeg，暂时无法生成兼容预览。请下载原视频。')
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = self.directory / f'{key}.{uuid4().hex}.partial.mp4'
            try:
                subprocess.run([
                    executable, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                    '-threads', '2', '-filter_threads', '2', '-i', str(source.resolve()), '-map', '0:v:0', '-map', '0:a:0?',
                    '-vf', "scale=w='min(1280,iw)':h='min(720,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac', '-movflags', '+faststart', '-threads', '2', str(temporary),
                ], check=True, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                if not temporary.is_file() or temporary.stat().st_size == 0:
                    raise HTTPException(422, '未能生成视频预览，请下载原视频。')
                temporary.replace(destination)
            except subprocess.TimeoutExpired:
                raise HTTPException(504, '视频预览准备超时，请下载原视频。') from None
            except (subprocess.CalledProcessError, OSError):
                raise HTTPException(422, '此视频无法生成兼容预览，请下载原视频。') from None
            finally:
                temporary.unlink(missing_ok=True)
        return destination


def create_video_router(files, directory):
    previews = VideoPreviews(directory)
    router = APIRouter(tags=['files'])

    @router.get('/api/files/{file_id}/video-preview')
    def video_preview(file_id: str):
        source = files.get(file_id)
        if source is None or not Path(source).is_file():
            raise HTTPException(404, '视频文件不存在。')
        if Path(source).suffix.lower() not in VIDEO_EXTENSIONS:
            raise HTTPException(415, '此文件不是支持的视频格式。')
        path = previews.prepare(source)
        # FileResponse handles byte-range requests so playback can seek.
        return FileResponse(path, media_type='video/mp4', filename=f'{Path(source).stem}_preview.mp4',
                            content_disposition_type='inline')

    return router
