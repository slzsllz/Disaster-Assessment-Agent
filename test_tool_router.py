import asyncio
import json
import unittest
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from agent.tool_router import ToolRouter, parse_tool_names


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return SimpleNamespace(content=self.response)


class ToolRouterTests(unittest.TestCase):
    def test_rejects_hallucinated_and_duplicate_names(self):
        selected = parse_tool_names(
            json.dumps({"tools": ["calculate_ndvi", "invented", "calculate_ndvi"]}),
            {"calculate_ndvi"},
        )
        self.assertEqual(selected, ["calculate_ndvi"])

    def test_routes_and_caches_one_turn(self):
        llm = FakeLLM('{"tools": ["calculate_ndvi"]}')
        tools = [SimpleNamespace(name="calculate_ndvi", description="Vegetation index"),
                 SimpleNamespace(name="mean", description="Calculate mean")]
        router = ToolRouter(llm, tools)
        messages = [HumanMessage(content="计算 NDVI")]
        first = asyncio.run(router.select(messages))
        second = asyncio.run(router.select(messages))
        self.assertEqual([tool.name for tool in first], ["calculate_ndvi"])
        self.assertEqual(first, second)
        self.assertEqual(llm.calls, 1)

    def test_explicit_name_survives_router_failure(self):
        class BrokenLLM:
            async def ainvoke(self, messages):
                raise RuntimeError("unavailable")

        tool = SimpleNamespace(name="calculate_ndvi", description="Vegetation index")
        router = ToolRouter(BrokenLLM(), [tool])
        selected = asyncio.run(router.select([HumanMessage(content="请调用 calculate_ndvi")]))
        self.assertEqual(selected, [tool])


if __name__ == "__main__":
    unittest.main()
