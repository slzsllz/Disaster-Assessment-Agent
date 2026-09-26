import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from agent.web_search import search_web, web_references
from agent.tool_router import ToolRouter


class WebSearchTests(unittest.TestCase):
    def test_news_search_parses_source_and_dates(self):
        feed = (b'<rss><channel><item><title>Flood update</title>'
                b'<link>https://example.org/flood</link>'
                b'<description>River level rising</description>'
                b'<pubDate>Fri, 25 Sep 2026 10:00:00 GMT</pubDate>'
                b'<source>News Desk</source></item></channel></rss>')
        with patch.dict("os.environ", {"WEB_SEARCH_PROVIDER": "news"}), patch("agent.web_search._get", return_value=feed):
            result = search_web("flood", 5)
        self.assertTrue(result["success"])
        self.assertEqual(result["scope"], "news")
        self.assertEqual(result["results"][0]["url"], "https://example.org/flood")
        self.assertEqual(result["results"][0]["published_at"], "2026-09-25T10:00:00+00:00")

    def test_brave_and_searxng_providers(self):
        brave_response = {"web": {"results": [{
            "title": "Official warning", "url": "https://example.org/warning",
            "description": "<b>Flood</b> warning",
        }]}}
        with patch.dict("os.environ", {"WEB_SEARCH_PROVIDER": "brave", "BRAVE_SEARCH_API_KEY": "test"}), patch(
            "agent.web_search._get", return_value=json.dumps(brave_response).encode()
        ) as request:
            brave = search_web("flood")
        self.assertEqual(brave["results"][0]["snippet"], "Flood warning")
        self.assertEqual(brave["scope"], "web")
        self.assertEqual(request.call_args.kwargs["headers"]["X-Subscription-Token"], "test")

        searx_response = {"results": [{
            "title": "River status", "url": "https://example.org/river",
            "content": "Current river status", "engine": "bing",
        }]}
        with patch.dict("os.environ", {"WEB_SEARCH_PROVIDER": "searxng", "SEARXNG_URL": "http://localhost:8080"}), patch(
            "agent.web_search._get", return_value=json.dumps(searx_response).encode()
        ) as request:
            searx = search_web("river")
        self.assertEqual(searx["results"][0]["source"], "bing")
        self.assertEqual(request.call_args.args[0], "http://localhost:8080/search")

    def test_references_only_from_successful_web_tool_and_public_urls(self):
        payload = {"success": True, "provider": "news", "retrieved_at": "now", "results": [
            {"title": "Flood", "url": "https://example.org/flood", "published_at": "then"},
            {"title": "Internal", "url": "http://127.0.0.1/admin"},
        ]}
        messages = [
            AIMessage(content="", tool_calls=[{"name": "web_search", "args": {}, "id": "call-1"}]),
            ToolMessage(content=json.dumps(payload), tool_call_id="call-1"),
            ToolMessage(content=json.dumps(payload), name="other_tool", tool_call_id="call-2"),
        ]
        refs = web_references(messages)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["url"], "https://example.org/flood")
        self.assertEqual(refs[0]["source_type"], "web")

    def test_router_detects_live_intent(self):
        class EmptyLLM:
            async def ainvoke(self, messages):
                return AIMessage(content='{"tools": []}')

        import asyncio
        from langchain_core.messages import HumanMessage
        from types import SimpleNamespace
        tools = [SimpleNamespace(name="web_search", description="Internet search")]
        router = ToolRouter(EmptyLLM(), tools)
        selected = asyncio.run(router.select([HumanMessage(content="搜索今天的洪水新闻")]))
        self.assertEqual([tool.name for tool in selected], ["web_search"])


if __name__ == "__main__":
    unittest.main()
