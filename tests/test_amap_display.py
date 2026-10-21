import os
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI

from agent.amap_display import AMapImages, create_amap_router

PNG = b'\x89PNG\r\n\x1a\n' + b'test-image'


class AMapDisplayTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'AMAP_WEB_KEY': 'private-test-key'})
        env.start()
        self.addCleanup(env.stop)

    def upstream(self, handler):
        original = httpx.AsyncClient
        transport = httpx.MockTransport(handler)
        return patch('agent.amap_display.httpx.AsyncClient', side_effect=lambda **kwargs: original(transport=transport, **kwargs))

    async def test_image_cache_and_private_key(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, content=PNG)
        images = AMapImages()
        with self.upstream(handler):
            first = await images.get(116.4, 39.9, 12, 640, 480)
            second = await images.get(116.4, 39.9, 12, 640, 480)
        self.assertEqual(first, (PNG, 'image/png'))
        self.assertEqual(first, second)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.host, 'restapi.amap.com')
        self.assertEqual(requests[0].url.params['key'], 'private-test-key')
        self.assertEqual(requests[0].url.params['size'], '640*480')

    async def test_proxy_validation_and_no_credential_leak(self):
        app = FastAPI()
        app.include_router(create_amap_router())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            for query in ('longitude=181&latitude=30', 'longitude=116&latitude=nan',
                          'longitude=116&latitude=39&width=2048', 'longitude=116&latitude=39&zoom=18'):
                self.assertEqual((await client.get('/api/maps/amap/static?' + query)).status_code, 422)
            with self.upstream(lambda _: httpx.Response(200, content=PNG)):
                response = await client.get('/api/maps/amap/static?longitude=116&latitude=39')
            self.assertEqual(response.content, PNG)
            self.assertEqual(response.headers['content-type'], 'image/png')
            self.assertNotIn('private-test-key', str(response.headers))
            self.assertIn('private', response.headers['cache-control'])
            with self.upstream(lambda _: httpx.Response(200, json={'info':'error private-test-key', 'infocode':'10009'})):
                response = await client.get('/api/maps/amap/static?longitude=117&latitude=39')
            self.assertEqual(response.status_code, 502)
            self.assertNotIn('private-test-key', response.text)
            self.assertIn('不匹配', response.json()['detail'])

    async def test_missing_key_and_network_failures(self):
        app = FastAPI()
        app.include_router(create_amap_router())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            with patch.dict(os.environ, {'AMAP_WEB_KEY': ''}):
                self.assertEqual((await client.get('/api/maps/amap/static?longitude=116&latitude=39')).status_code, 503)
            def failure(_):
                raise httpx.ConnectError('private-test-key')
            with self.upstream(failure):
                response = await client.get('/api/maps/amap/static?longitude=116&latitude=39')
            self.assertEqual(response.status_code, 502)
            self.assertNotIn('private-test-key', response.text)


if __name__ == '__main__':
    unittest.main()
