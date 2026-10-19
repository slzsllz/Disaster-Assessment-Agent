import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from agent.error_memory import ErrorMemory
from agent.tool_input_guard import guard_tool_inputs, tool_input_scope
from agent.tool_recovery import recovery_run, wrap_tool


SCHEMA = {
    "type": "object",
    "properties": {"image_path": {"type": "string"}, "output_dir": {"type": "string"}},
    "required": ["image_path"],
}


class InputGuardTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.session = self.root / "session"
        (self.session / "uploads").mkdir(parents=True)
        (self.session / "out").mkdir()
        self.upload = self.session / "uploads" / "input.tif"
        self.upload.touch()
        self.server_sample = self.root / "benchmark" / "sample.tif"
        self.server_sample.parent.mkdir()
        self.server_sample.touch()
        self.callback = AsyncMock(return_value=('{"success":true}', None))
        tool = StructuredTool(
            name="detect_flood", description="Test", args_schema=SCHEMA,
            coroutine=self.callback, response_format="content_and_artifact",
        )
        self.tool = wrap_tool(guard_tool_inputs(tool), ErrorMemory())

    def tearDown(self):
        self.directory.cleanup()

    async def call(self, path, messages=None):
        with recovery_run(), tool_input_scope(self.session, messages or []):
            return await self.tool.ainvoke({
                "type": "tool_call", "id": "input-test", "name": self.tool.name,
                "args": {"image_path": str(path), "output_dir": "results"},
            })

    async def test_unprovided_server_sample_is_blocked_before_execution(self):
        result = await self.call(self.server_sample, [HumanMessage(content="评估这片淹没区域")])
        self.assertEqual(result.status, "error")
        self.assertIn("Ask the user to upload", result.content)
        self.callback.assert_not_awaited()

    async def test_invented_path_is_blocked_before_execution(self):
        result = await self.call(self.root / "benchmark" / "s1_bangkok.tif")
        self.assertEqual(result.status, "error")
        self.callback.assert_not_awaited()

    async def test_current_and_previous_uploads_remain_usable(self):
        self.assertEqual((await self.call(self.upload)).status, "success")
        self.callback.assert_awaited_once()

    async def test_same_conversation_outputs_remain_usable(self):
        output = self.session / "out" / "flood_mask.tif"
        output.touch()
        self.assertEqual((await self.call(output)).status, "success")

    async def test_explicit_user_path_is_allowed(self):
        message = HumanMessage(content=f"分析这个文件：`{self.server_sample}`")
        self.assertEqual((await self.call(self.server_sample, [message])).status, "success")

    async def test_system_and_assistant_paths_are_not_user_authorization(self):
        messages = [SystemMessage(content=str(self.server_sample)), AIMessage(content=str(self.server_sample))]
        self.assertEqual((await self.call(self.server_sample, messages)).status, "error")
        self.callback.assert_not_awaited()

    async def test_other_conversation_and_symlink_escape_are_blocked(self):
        link = self.session / "uploads" / "other.tif"
        link.symlink_to(self.server_sample)
        self.assertEqual((await self.call(link)).status, "error")
        self.callback.assert_not_awaited()

    async def test_batch_inputs_are_all_checked(self):
        callback = AsyncMock(return_value="[]")
        tool = StructuredTool(
            name="calc_batch_image_mean", description="Test",
            args_schema={"type": "object", "properties": {"file_list": {"type": "array", "items": {"type": "string"}}}},
            coroutine=callback,
        )
        guarded = wrap_tool(guard_tool_inputs(tool), ErrorMemory())
        with recovery_run(), tool_input_scope(self.session, []):
            result = await guarded.ainvoke({
                "type": "tool_call", "id": "batch-test", "name": guarded.name,
                "args": {"file_list": [str(self.upload), str(self.server_sample)]},
            })
        self.assertEqual(result.status, "error")
        callback.assert_not_awaited()

    async def test_default_fire_dataset_cannot_run_without_user_input(self):
        callback = AsyncMock(return_value="{}")
        tool = StructuredTool(
            name="detect_fire_burned_area_change", description="Test",
            args_schema={"type": "object", "properties": {}}, coroutine=callback,
        )
        guarded = wrap_tool(guard_tool_inputs(tool), ErrorMemory())
        with recovery_run(), tool_input_scope(self.session, []):
            result = await guarded.ainvoke({
                "type": "tool_call", "id": "fire-test", "name": guarded.name, "args": {},
            })
        self.assertEqual(result.status, "error")
        callback.assert_not_awaited()

    async def test_non_file_tools_are_not_disabled(self):
        callback = AsyncMock(return_value="{}")
        tool = StructuredTool(
            name="query_weather", description="Test",
            args_schema={"type": "object", "properties": {"location": {"type": "string"}}},
            coroutine=callback,
        )
        guarded = wrap_tool(guard_tool_inputs(tool), ErrorMemory())
        with recovery_run(), tool_input_scope(self.session, [HumanMessage(content="深圳天气")]):
            await guarded.ainvoke({"location": "Shenzhen"})
        callback.assert_awaited_once()

    async def test_parallel_requests_do_not_share_authorized_paths(self):
        async def request(authorize):
            messages = [HumanMessage(content=f"`{self.server_sample}`")] if authorize else []
            with recovery_run(), tool_input_scope(self.session, messages):
                await asyncio.sleep(0)
                return await self.tool.ainvoke({
                    "type": "tool_call", "id": str(authorize), "name": self.tool.name,
                    "args": {"image_path": str(self.server_sample)},
                })
        allowed, blocked = await asyncio.gather(request(True), request(False))
        self.assertEqual((allowed.status, blocked.status), ("success", "error"))
        self.callback.assert_awaited_once()


class BackendInputGuardTests(unittest.TestCase):
    def test_normal_and_stream_agent_reject_unprovided_files_and_ask_for_input(self):
        import backend_api

        class ToolCallingModel(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        for streaming in (False, True):
            with self.subTest(streaming=streaming), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                server_sample = root / "benchmark" / "sample.tif"
                server_sample.parent.mkdir()
                server_sample.touch()
                callback = AsyncMock(return_value=('{"success":true}', None))
                tool = StructuredTool(
                    name="detect_flood", description="Test", args_schema=SCHEMA,
                    coroutine=callback, response_format="content_and_artifact",
                )
                llm = ToolCallingModel(responses=[
                    AIMessage(content="", tool_calls=[{
                        "id": "invented-input", "name": "detect_flood", "args": {"image_path": str(server_sample)},
                    }]),
                    AIMessage(content="<Conclusion>请上传要分析的影像，并说明地点。</Conclusion>"),
                ])
                with (
                    patch.object(backend_api, "create_chat_model", return_value=llm),
                    patch.object(backend_api, "ERROR_MEMORY", ErrorMemory()),
                    patch("langchain_mcp_adapters.client.MultiServerMCPClient") as client,
                ):
                    client.return_value.get_tools = AsyncMock(return_value=[tool])
                    client.return_value.aclose = AsyncMock()
                    handle = backend_api.AgentHandle({"mcp_servers": {}}, root / "session")
                    try:
                        messages = [HumanMessage(content="评估这片淹没区域")]
                        if streaming:
                            events = list(handle.stream(messages))
                            response = next(event["response"] for event in events if event["type"] == "final")
                        else:
                            response = handle.invoke(messages)
                        self.assertIn("请上传", response["messages"][-1].content)
                        results = [message for message in response["messages"] if isinstance(message, ToolMessage)]
                        self.assertEqual(results[0].status, "error")
                        callback.assert_not_awaited()
                    finally:
                        handle.run(handle.loop.shutdown_default_executor())
                        handle.close()
                        handle.thread.join(timeout=5)
                        handle.loop.close()


if __name__ == "__main__":
    unittest.main()
