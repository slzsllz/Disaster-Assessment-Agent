"""Same-origin AMap static basemap proxy; Web Service credentials stay on the server."""
from __future__ import annotations

import asyncio
import os
import time
from collections import OrderedDict

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response


class AMapImages:
    def __init__(self):
        self.cache = OrderedDict()
        self.lock = asyncio.Lock()
        self.last_request = 0.0

    async def get(self, longitude, latitude, zoom, width, height):
        key = os.getenv('AMAP_WEB_KEY', '').strip()
        if not key:
            raise HTTPException(503, '未配置 AMAP_WEB_KEY，无法加载高德地图。')
        params = {'location': f'{longitude:.6f},{latitude:.6f}', 'zoom': zoom,
                  'size': f'{width}*{height}', 'scale': 1}
        cache_key = (key, *params.values())
        async with self.lock:
            cached = self.cache.get(cache_key)
            if cached and time.monotonic() - cached[0] < 60:
                self.cache.move_to_end(cache_key)
                return cached[1:]
            await asyncio.sleep(max(0, .35 - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                async with httpx.AsyncClient(timeout=20, trust_env=False) as client:
                    result = await client.get('https://restapi.amap.com/v3/staticmap', params={**params, 'key': key})
            except httpx.HTTPError:
                raise HTTPException(502, '无法连接高德地图服务，请稍后重试。') from None
            content = result.content
            # Never forward upstream JSON, URLs or error descriptions containing credentials.
            if result.status_code != 200:
                raise HTTPException(502, '高德地图服务暂时不可用。')
            if content.startswith(b'\x89PNG\r\n\x1a\n'):
                media_type = 'image/png'
            elif content.startswith(b'\xff\xd8\xff'):
                media_type = 'image/jpeg'
            else:
                code = ''
                try:
                    code = str(result.json().get('infocode', ''))
                except (ValueError, AttributeError):
                    pass
                hint = {'10001': 'Key 无效', '10003': '调用次数已达上限',
                        '10004': '请求频率超限', '10005': 'IP 白名单限制',
                        '10009': 'Key 与服务平台不匹配'}.get(code, '请检查 Key 权限和服务配额')
                raise HTTPException(502, f'高德地图加载失败：{hint}。')
            if len(content) > 8 * 1024 * 1024:
                raise HTTPException(502, '高德地图返回的图片过大。')
            self.cache[cache_key] = (time.monotonic(), content, media_type)
            self.cache.move_to_end(cache_key)
            while len(self.cache) > 16:
                self.cache.popitem(last=False)
            return content, media_type


def create_amap_router(images=None):
    images = images or AMapImages()
    router = APIRouter(prefix='/api/maps/amap', tags=['maps'])

    @router.get('/static')
    async def static_map(
        longitude: float = Query(..., ge=-180, le=180),
        latitude: float = Query(..., ge=-85, le=85),
        zoom: int = Query(12, ge=1, le=17),
        width: int = Query(640, ge=64, le=1024),
        height: int = Query(640, ge=64, le=1024),
    ):
        content, media_type = await images.get(longitude, latitude, zoom, width, height)
        return Response(content, media_type=media_type, headers={
            'Cache-Control': 'private, max-age=60', 'X-Content-Type-Options': 'nosniff',
        })

    return router
