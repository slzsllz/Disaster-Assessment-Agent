"""Contract tests for Baidu web search without external API calls."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('web_search_test_tool', ROOT / 'agent/tools/WebSearch.py')
web = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web)

REFERENCE = {'id': 1, 'title': 'Official disaster report', 'url': 'https://www.mem.gov.cn/report',
             'content': 'Report excerpt', 'date': '2026-10-16', 'website': '应急管理部',
             'type': 'web', 'authority_score': .9}


class WebSearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original_temp = web.TEMP_DIR
        web.TEMP_DIR = self.root
        self.addCleanup(setattr, web, 'TEMP_DIR', self.original_temp)
        self.env = patch.dict(os.environ, {'QIANFAN_WEB_SEARCH_API_KEY': 'test-secret',
                                         'QIANFAN_WEB_SEARCH_URL': web.DEFAULT_URL})
        self.env.start()
        self.addCleanup(self.env.stop)

    def client_mock(self, handler):
        client_class = httpx.AsyncClient
        transport = httpx.MockTransport(handler)
        return patch.object(web.httpx, 'AsyncClient', side_effect=lambda **kw: client_class(transport=transport, **kw))

    async def test_request_response_and_unique_summaries(self):
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={'request_id': 'req-1', 'references': [REFERENCE]})
        with self.client_mock(handler):
            result = await web.search_web('  地震防范  ', top_k=3, domains=['MEM.GOV.CN'], recency='month')
            again = await web.search_web('地震防范')
        self.assertEqual(requests[0].headers['Authorization'], 'Bearer test-secret')
        body = json.loads(requests[0].content)
        self.assertEqual(body['messages'], [{'role': 'user', 'content': '地震防范'}])
        self.assertEqual(body['resource_type_filter'], [{'type': 'web', 'top_k': 3}])
        self.assertEqual(body['search_filter'], {'match': {'site': ['mem.gov.cn']}})
        self.assertEqual(body['search_recency_filter'], 'month')
        self.assertEqual(result['result_count'], 1)
        self.assertEqual(result['references'][0]['url'], REFERENCE['url'])
        saved = Path(result['summary_path']).read_text()
        self.assertNotIn('test-secret', saved)
        self.assertEqual(json.loads(saved), result)
        self.assertNotEqual(result['summary_path'], again['summary_path'])

    async def test_empty_results_are_success(self):
        with self.client_mock(lambda _: httpx.Response(200, json={'references': []})):
            result = await web.search_web('no matches')
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['result_count'], 0)

    async def test_validation_precedes_network(self):
        cases = [dict(query=' '), dict(query='q', top_k=0), dict(query='q', top_k=True),
                 dict(query='q', recency='day'), dict(query='q', domains=['https://mem.gov.cn/']),
                 dict(query='q', output_dir='../escape'), dict(query='q', output_dir='/tmp/escape')]
        with patch.object(web, '_request_search', new_callable=AsyncMock) as request:
            for options in cases:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    await web.search_web(**options)
            request.assert_not_awaited()

    async def test_provider_errors_redact_key(self):
        for status in (200, 401, 429):
            with self.client_mock(lambda _: httpx.Response(status, json={
                'code': 'InvalidAPIKey', 'message': 'test-secret was rejected'})):
                with self.assertRaises(RuntimeError) as ctx:
                    await web.search_web('q')
                self.assertNotIn('test-secret', str(ctx.exception))
                self.assertIn('InvalidAPIKey', str(ctx.exception))
        self.assertFalse(list(self.root.rglob('*.json')))

    async def test_missing_key_invalid_json_and_timeout(self):
        with patch.dict(os.environ, {'QIANFAN_WEB_SEARCH_API_KEY': ''}):
            with self.assertRaisesRegex(RuntimeError, 'Set QIANFAN'):
                await web.search_web('q')
        with self.client_mock(lambda _: httpx.Response(502, text='gateway html')):
            with self.assertRaisesRegex(RuntimeError, 'non-JSON'):
                await web.search_web('q')
        with self.client_mock(lambda _: httpx.Response(200, json={'request_id': 'r'})):
            with self.assertRaisesRegex(RuntimeError, 'references'):
                await web.search_web('q')
        def timeout(request):
            raise httpx.ReadTimeout('test-secret', request=request)
        with self.client_mock(timeout):
            with self.assertRaisesRegex(RuntimeError, 'timed out') as ctx:
                await web.search_web('q')
            self.assertNotIn('test-secret', str(ctx.exception))

    @unittest.skipIf(web.mcp is None, 'FastMCP unavailable')
    async def test_mcp_discovery_and_invocation(self):
        from fastmcp import Client
        with patch.object(web, '_request_search', new_callable=AsyncMock,
                          return_value={'references': [REFERENCE], 'request_id': 'mcp-1'}):
            async with Client(web.mcp, timeout=10) as client:
                tool = (await client.list_tools())[0]
                self.assertEqual(tool.name, 'search_web')
                self.assertEqual(tool.inputSchema['required'], ['query'])
                result = await client.call_tool('search_web', {'query': 'earthquake notice'})
                self.assertFalse(result.is_error)
                self.assertIn('Official disaster report', result.content[0].text)

    def test_configs(self):
        for name in ('config.json', 'config_deepseek.json', 'config_qwen3.json'):
            config = json.loads((ROOT / 'agent' / name).read_text())
            self.assertEqual(config['mcpServers']['WebSearch']['args'][0], 'tools/WebSearch.py')


if __name__ == '__main__':
    unittest.main()
