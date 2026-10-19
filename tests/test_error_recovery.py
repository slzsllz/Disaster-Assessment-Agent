import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool, ToolException
from langgraph.prebuilt import create_react_agent

from agent.error_memory import ErrorMemory, normalize_error
from agent.tool_recovery import MAX_FAILURES, recovery_run, wrap_tool


SCHEMA = {
    "type": "object",
    "properties": {
        "image_path": {"type": "string"},
        "device": {"type": "string"},
        "threshold": {"type": "number"},
    },
    "required": ["image_path"],
}


class MemoryDatabase:
    def __init__(self):
        self.entries = {}
        self.hits = {}
        self.available = True
        self.checked = False

    def health_check(self):
        self.checked = False
        return self.available

    def load_all_errors(self):
        if not self.available or self.checked:
            self.checked = True
            return {}
        return dict(self.entries)

    def save_error(self, pattern, fix):
        if not self.available or self.checked:
            self.checked = True
            return False
        self.entries[pattern] = fix
        return True

    def increment_error_hit(self, pattern):
        self.hits[pattern] = self.hits.get(pattern, 0) + 1
        return True


def make_tool(coroutine, memory):
    tool = StructuredTool(
        name="detect_flood", description="Test detector", args_schema=SCHEMA,
        coroutine=coroutine, response_format="content_and_artifact",
    )
    return wrap_tool(tool, memory)


async def invoke_tool(tool, arguments):
    return await tool.ainvoke({
        "type": "tool_call", "id": "call-test", "name": tool.name, "args": arguments,
    })


class ErrorMemoryTests(unittest.TestCase):
    def test_error_at_end_of_long_traceback_still_matches(self):
        traceback = "Traceback:\n" + "  irrelevant stack frame\n" * 200
        traceback += "ConnectionError: Connection refused"
        self.assertTrue(ErrorMemory().lookup_all(traceback))

    def test_legacy_json_memory_is_readable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "errors.json"
            path.write_text(json.dumps({"Example error": "Use a valid input"}), encoding="utf-8")
            self.assertEqual(ErrorMemory(path).lookup("Example error"), "Use a valid input")

    def test_database_scoped_retrieval_and_hits(self):
        database = MemoryDatabase()
        database.entries["detect_flood::CUDA out of memory"] = "Use device=cpu"
        memory = ErrorMemory(database=database)
        other = memory.lookup_all("CUDA out of memory", tool_name="other")
        self.assertFalse(any(pattern.startswith("detect_flood::") for pattern, _ in other))
        matches = memory.lookup_all("CUDA OUT OF MEMORY", tool_name="detect_flood")
        self.assertEqual(matches[0][1], "Use device=cpu")
        self.assertEqual(database.hits["detect_flood::CUDA out of memory"], 1)
        self.assertEqual(memory.lookup_all("MemoryError", limit=0), [])

    def test_persistence_survives_reload_and_database_outage(self):
        database = MemoryDatabase()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "errors.json"
            memory = ErrorMemory(path, database=database)
            memory.record("detect_flood::MemoryError", "Use device=cpu for this tool")
            self.assertIn("detect_flood::MemoryError", database.entries)
            self.assertIn("detect_flood::MemoryError", ErrorMemory(path).get_all())
            with patch.object(database, "save_error", return_value=False):
                memory.record("detect_flood::Example error", "Use a smaller batch size")
            memory._last_sync = float("-inf")
            memory.get_all()
            self.assertIn("detect_flood::Example error", database.entries)

    def test_patterns_do_not_store_local_paths_or_credentials(self):
        value = normalize_error("Error at /home/test/private/input.tif api_key=sk-abcdefg")
        self.assertNotIn("/home/test", value)
        self.assertNotIn("sk-abcdefg", value)
        value = normalize_error('Error at "/home/test/private data/input.tif" {"api_key": "private-key"}')
        self.assertNotIn("private data", value)
        self.assertNotIn("private-key", value)

    def test_pending_database_write_survives_restart_and_reconnect(self):
        database = MemoryDatabase()
        database.available = False
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "errors.json"
            original = ErrorMemory(path, database=database)
            original.record("detect_flood::MemoryError", "Use device=cpu")
            restored = ErrorMemory(path, database=database)
            restored.get_all()
            database.available = True
            restored._last_sync = float("-inf")
            restored.get_all()
            self.assertEqual(database.entries.get("detect_flood::MemoryError"), "Use device=cpu")
            self.assertFalse(restored._dirty)
            self.assertFalse(ErrorMemory(path)._dirty)


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_output_directory_change_is_not_learned_as_device_repair(self):
        memory = ErrorMemory()

        async def detect(**arguments):
            if arguments.get("output_dir") == "readonly":
                raise PermissionError("Output directory is not writable")
            return '{"success": true}', None

        tool = make_tool(detect, memory)
        before = memory.get_all()
        with recovery_run():
            await invoke_tool(tool, {"image_path": "scene.tif", "device": "cuda", "output_dir": "readonly"})
            await invoke_tool(tool, {"image_path": "scene.tif", "device": "cpu", "output_dir": "writable"})
        self.assertEqual(memory.get_all(), before)

    async def test_sampling_change_is_not_learned_as_execution_repair(self):
        memory = ErrorMemory()

        async def detect(**arguments):
            if arguments.get("sample_factor") == 1:
                raise MemoryError("Array is too large")
            return '{"success": true}', None

        tool = make_tool(detect, memory)
        before = memory.get_all()
        with recovery_run():
            await invoke_tool(tool, {"image_path": "scene.tif", "device": "cuda", "sample_factor": 1})
            await invoke_tool(tool, {"image_path": "scene.tif", "device": "cpu", "sample_factor": 8})
        self.assertEqual(memory.get_all(), before)

    async def test_explicit_default_does_not_reset_retry_budget(self):
        calls = []

        async def detect(**arguments):
            calls.append(arguments)
            raise ValueError("Invalid input")

        schema = {**SCHEMA, "properties": {**SCHEMA["properties"], "threshold": {"type": "number", "default": 0.5}}}
        tool = wrap_tool(StructuredTool(
            name="detect_flood", description="Test", args_schema=schema,
            coroutine=detect, response_format="content_and_artifact",
        ), ErrorMemory())
        with recovery_run():
            for _ in range(MAX_FAILURES):
                await invoke_tool(tool, {"image_path": "scene.tif"})
            await invoke_tool(tool, {"image_path": "scene.tif", "threshold": 0.5})
        self.assertEqual(len(calls), MAX_FAILURES)

    async def test_exception_reaches_model_then_success_is_learned(self):
        memory = ErrorMemory()
        artifact = [{"type": "image", "data": "test"}]

        async def detect(**arguments):
            if arguments.get("device") != "cpu":
                raise RuntimeError("CUDA out of memory: /private/input.tif")
            return json.dumps({"success": True, "flood_ratio": 0.2}), artifact

        tool = make_tool(detect, memory)
        with recovery_run():
            failed = await invoke_tool(tool, {"image_path": "scene.tif", "device": "cuda"})
            self.assertEqual(failed.status, "error")
            recovery = json.loads(failed.content)["error_recovery"]
            self.assertEqual(recovery["remaining_attempts"], MAX_FAILURES - 1)
            succeeded = await invoke_tool(tool, {"image_path": "scene.tif", "device": "cpu"})
            self.assertEqual(succeeded.status, "success")
            self.assertEqual(succeeded.artifact, artifact)
            self.assertEqual(json.loads(succeeded.content)["flood_ratio"], 0.2)
        learned = memory.lookup_all("RuntimeError: CUDA out of memory: /other/input.tif", tool_name=tool.name)
        self.assertTrue(learned)
        self.assertIn('"device": "cpu"', learned[0][1])
        self.assertNotIn("/private", learned[0][0])

    async def test_structured_failed_result_is_not_success(self):
        async def detect(**arguments):
            return json.dumps({"success": False, "error": "Invalid bands"}), None

        with recovery_run():
            result = await invoke_tool(make_tool(detect, ErrorMemory()), {"image_path": "scene.tif"})
        self.assertEqual(result.status, "error")
        self.assertIn("Invalid bands", result.content)

    async def test_plain_error_and_validation_failure_reach_model(self):
        from pydantic import BaseModel

        class Arguments(BaseModel):
            image_path: str

        calls = []

        async def detect(**arguments):
            calls.append(arguments)
            return "Error: image is unreadable", None

        tool = StructuredTool(
            name="detect_flood", description="Test", args_schema=Arguments,
            coroutine=detect, response_format="content_and_artifact",
        )
        wrapped = wrap_tool(tool, ErrorMemory())
        with recovery_run():
            invalid = await invoke_tool(wrapped, {})
            self.assertEqual(invalid.status, "error")
            self.assertIn("error_recovery", invalid.content)
            self.assertEqual(calls, [])
            failed = await invoke_tool(wrapped, {"image_path": "scene.tif"})
            self.assertEqual(failed.status, "error")
            self.assertIn("unreadable", failed.content)

    async def test_budget_blocks_execution_and_is_reset_per_request(self):
        calls = []

        async def detect(**arguments):
            calls.append(arguments)
            raise ToolException("Connection refused")

        tool = make_tool(detect, ErrorMemory())
        with recovery_run():
            for _ in range(MAX_FAILURES + 2):
                result = await invoke_tool(tool, {"image_path": "scene.tif"})
            self.assertEqual(len(calls), MAX_FAILURES)
            self.assertEqual(json.loads(result.content)["error_recovery"]["remaining_attempts"], 0)
            await invoke_tool(tool, {"image_path": "another.tif"})
            self.assertEqual(len(calls), MAX_FAILURES + 1)
        with recovery_run():
            await invoke_tool(tool, {"image_path": "scene.tif"})
            self.assertEqual(len(calls), MAX_FAILURES + 2)

    async def test_unrelated_input_and_scientific_changes_are_not_learned(self):
        memory = ErrorMemory()

        async def detect(**arguments):
            if arguments.get("device") == "cuda":
                raise ValueError("Invalid detection input")
            return json.dumps({"success": True}), None

        tool = make_tool(detect, memory)
        original = memory.get_all()
        with recovery_run():
            await invoke_tool(tool, {"image_path": "a.tif", "device": "cuda"})
            await invoke_tool(tool, {"image_path": "b.tif", "device": "cpu"})
            await invoke_tool(tool, {"image_path": "a.tif", "device": "cpu", "threshold": 0.9})
        self.assertEqual(memory.get_all(), original)

    async def test_concurrent_requests_do_not_share_failure_evidence(self):
        memory = ErrorMemory()

        async def detect(**arguments):
            await asyncio.sleep(0)
            if arguments.get("device") == "cuda":
                raise ValueError("An example failure")
            return json.dumps({"success": True}), None

        tool = make_tool(detect, memory)

        async def request(device):
            with recovery_run():
                return await invoke_tool(tool, {"image_path": "same.tif", "device": device})

        before = memory.get_all()
        await asyncio.gather(request("cuda"), request("cpu"))
        self.assertEqual(memory.get_all(), before)

    async def test_langgraph_agent_receives_failure_and_can_correct_call(self):
        class ToolCallingModel(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        async def detect(**arguments):
            if arguments.get("device") != "cpu":
                raise RuntimeError("MemoryError")
            return '{"success": true, "flood_ratio": 0.2}', None

        memory = ErrorMemory()
        llm = ToolCallingModel(responses=[
            AIMessage(content="", tool_calls=[{
                "id": "first", "name": "detect_flood",
                "args": {"image_path": "scene.tif", "device": "cuda"},
            }]),
            AIMessage(content="", tool_calls=[{
                "id": "second", "name": "detect_flood",
                "args": {"image_path": "scene.tif", "device": "cpu"},
            }]),
            AIMessage(content="<Conclusion>Flood ratio is 20%.</Conclusion>"),
        ])
        agent = create_react_agent(llm, [make_tool(detect, memory)])
        with recovery_run():
            response = await agent.ainvoke({"messages": [HumanMessage(content="Detect flood")]})
        results = [msg for msg in response["messages"] if isinstance(msg, ToolMessage)]
        self.assertEqual([msg.status for msg in results], ["error", "success"])
        self.assertTrue(json.loads(results[0].content)["error_recovery"]["matched_experiences"])
        self.assertEqual(response["messages"][-1].content, llm.responses[-1].content)

        # The production SSE path uses astream_events, which must carry the same context.
        with recovery_run():
            events = [event async for event in agent.astream_events(
                {"messages": [HumanMessage(content="Detect flood again")]}, version="v2",
            )]
        outputs = [event["data"]["output"] for event in events if event["event"] == "on_tool_end"]
        self.assertEqual([msg.status for msg in outputs], ["error", "success"])


class BackendRecoveryTests(unittest.TestCase):
    def test_failed_initialization_stops_background_loop(self):
        import backend_api

        handle = backend_api.AgentHandle.__new__(backend_api.AgentHandle)
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(backend_api, "create_chat_model"),
                patch("langchain_mcp_adapters.client.MultiServerMCPClient") as client_class,
            ):
                client_class.return_value.get_tools = AsyncMock(side_effect=RuntimeError("MCP unavailable"))
                try:
                    with self.assertRaisesRegex(RuntimeError, "MCP unavailable"):
                        handle.__init__({"mcp_servers": {}}, Path(temporary))
                    self.assertFalse(handle.thread.is_alive())
                finally:
                    if handle.thread.is_alive():
                        handle.loop.call_soon_threadsafe(handle.loop.stop)
                        handle.thread.join(timeout=5)
                    if not handle.loop.is_closed():
                        handle.loop.close()

    def test_stream_initialization_failure_returns_error_event_and_unlocks(self):
        import backend_api

        with tempfile.TemporaryDirectory() as temporary:
            session = backend_api.ChatSession(session_id="test", temp_dir=Path(temporary))
            with (
                patch.object(backend_api, "get_session", return_value=session),
                patch.object(backend_api, "AgentHandle", side_effect=RuntimeError("Connection refused")),
                patch.object(backend_api, "ERROR_MEMORY", ErrorMemory()),
                patch.object(backend_api.db, "create_session"),
                patch.object(backend_api.db, "save_chat_message"),
                patch.object(backend_api, "maybe_generate_session_title"),
            ):
                response = backend_api.chat_stream(
                    session_id="test", message="Test", system_prompt="Test",
                    recursion_limit=40, max_execution_time=600, show_trace=False, files=None,
                )
                self.assertFalse(session.lock.locked())

                async def consume():
                    return [chunk async for chunk in response.body_iterator]

                chunks = asyncio.run(consume())
                self.assertEqual(len(chunks), 1)
                self.assertIn("event: error", chunks[0])
                self.assertIn("错误经验建议", chunks[0])

    def test_backend_invoke_and_stream_use_same_recovery_path(self):
        import backend_api

        class ToolCallingModel(FakeMessagesListChatModel):
            def bind_tools(self, tools, **kwargs):
                return self

        async def detect(**arguments):
            if arguments.get("device") != "cpu":
                raise RuntimeError("MemoryError")
            return '{"success": true, "flood_ratio": 0.2}', None

        raw_tool = StructuredTool(
            name="detect_flood", description="Test", args_schema=SCHEMA,
            coroutine=detect, response_format="content_and_artifact",
        )
        model = ToolCallingModel(responses=[
            AIMessage(content="", tool_calls=[{
                "id": "failed", "name": "detect_flood",
                "args": {"image_path": "scene.tif", "device": "cuda"},
            }]),
            AIMessage(content="", tool_calls=[{
                "id": "repaired", "name": "detect_flood",
                "args": {"image_path": "scene.tif", "device": "cpu"},
            }]),
            AIMessage(content="<Conclusion>Flood ratio is 20%.</Conclusion>"),
        ])
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "uploads" / "scene.tif"
            image.parent.mkdir()
            image.touch()
            for message in model.responses:
                for call in message.tool_calls:
                    call["args"]["image_path"] = str(image)
            memory = ErrorMemory(Path(temporary) / "memory.json", database=MemoryDatabase())
            with (
                patch.object(backend_api, "ERROR_MEMORY", memory),
                patch.object(backend_api, "create_chat_model", return_value=model),
                patch("langchain_mcp_adapters.client.MultiServerMCPClient") as client_class,
            ):
                client_class.return_value.get_tools = AsyncMock(return_value=[raw_tool])
                client_class.return_value.aclose = AsyncMock()
                handle = backend_api.AgentHandle({"mcp_servers": {}}, Path(temporary))
                try:
                    result = handle.invoke([HumanMessage(content="Detect flood")])
                    statuses = [msg.status for msg in result["messages"] if isinstance(msg, ToolMessage)]
                    self.assertEqual(statuses, ["error", "success"])
                    events = list(handle.stream([HumanMessage(content="Detect flood again")]))
                    self.assertFalse(any(event["type"] == "error" for event in events))
                    self.assertEqual(sum(event["type"] == "final" for event in events), 1)
                    self.assertTrue(any(
                        event["type"] == "status" and "错误经验" in event["message"]
                        for event in events
                    ))
                    self.assertTrue(memory.lookup_all("RuntimeError: MemoryError", tool_name="detect_flood"))
                finally:
                    handle.run(handle.loop.shutdown_default_executor())
                    handle.close()
                    handle.thread.join(timeout=5)
                    handle.loop.close()

    def test_fatal_error_suggestions_are_visible_in_answer(self):
        import backend_api

        with patch.object(backend_api, "ERROR_MEMORY", ErrorMemory()):
            payload = backend_api.backend_error_payload(ConnectionError("Connection refused"))
        self.assertTrue(payload["memory_suggestions"])
        self.assertIn("错误经验建议", payload["answer"])
        self.assertIn(payload["memory_suggestions"][0]["fix"], payload["answer"])


if __name__ == "__main__":
    unittest.main()
