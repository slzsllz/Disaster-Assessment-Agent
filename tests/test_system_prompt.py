import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, SystemMessage

import backend_api
from agent.error_memory import ErrorMemory


class SystemPromptTests(unittest.TestCase):
    def test_default_prompt_is_loaded_from_markdown(self):
        self.assertEqual(
            backend_api.DEFAULT_SYSTEM_PROMPT,
            backend_api.SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip(),
        )
        self.assertIn("城市灾害与韧性评估助手", backend_api.DEFAULT_SYSTEM_PROMPT)
        self.assertIn("地理与气象信息辅助研判", backend_api.DEFAULT_SYSTEM_PROMPT)
        self.assertIn("对用户的完整最终回答", backend_api.DEFAULT_SYSTEM_PROMPT)

    def test_loader_reads_utf8_independently_of_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "system.md"
            prompt_path.write_text("  # 城市灾害\n\n测试提示词\n", encoding="utf-8")
            self.assertEqual(
                backend_api.load_system_prompt(prompt_path), "# 城市灾害\n\n测试提示词"
            )
            with patch("pathlib.Path.cwd", return_value=Path(directory)):
                self.assertEqual(
                    backend_api.load_system_prompt(), backend_api.DEFAULT_SYSTEM_PROMPT
                )

    def test_missing_prompt_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "Cannot read system prompt"):
                backend_api.load_system_prompt(Path(directory) / "missing.md")

    def test_empty_prompt_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "system.md"
            prompt_path.write_text(" \n\t", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "System prompt is empty"):
                backend_api.load_system_prompt(prompt_path)

    def test_dynamic_instructions_are_appended_to_markdown(self):
        with patch.object(backend_api.ERROR_MEMORY, "format_prompt_block", return_value="\nRecovery rules"):
            prompt = backend_api.build_system_prompt("", ["/test/uploads", "/test/out"])
        self.assertTrue(prompt.startswith(backend_api.DEFAULT_SYSTEM_PROMPT))
        self.assertIn("/test/uploads", prompt)
        self.assertIn("Recovery rules", prompt)
        self.assertIn(backend_api.RAG_INSTRUCTIONS, prompt)

    def test_explicit_api_prompt_override_remains_supported(self):
        with patch.object(backend_api.ERROR_MEMORY, "format_prompt_block", return_value=""):
            prompt = backend_api.build_system_prompt("Custom API prompt", [])
        self.assertTrue(prompt.startswith("Custom API prompt"))
        self.assertIn(backend_api.RAG_INSTRUCTIONS, prompt)

    def test_both_chat_endpoints_use_markdown_without_frontend_prompt(self):
        response = {"messages": [AIMessage(content="<Conclusion>Test result</Conclusion>")]}

        class Handle:
            def __init__(self):
                self.messages = None

            def invoke(self, messages, config=None):
                self.messages = messages
                return response

            def stream(self, messages, config=None):
                self.messages = messages
                yield {"type": "final", "response": response}

            def review_answer_and_artifacts(self, raw, uploaded, outputs, **kwargs):
                return raw

        for endpoint in ("/api/chat", "/api/chat/stream"):
            for override in (None, "Explicit API prompt"):
                with self.subTest(endpoint=endpoint, override=override), tempfile.TemporaryDirectory() as directory:
                    handle = Handle()
                    session = backend_api.ChatSession("prompt-test", Path(directory), handle=handle)
                    form = {"session_id": "prompt-test", "message": "Test"}
                    if override is not None:
                        form["system_prompt"] = override
                    with (
                        patch.object(backend_api, "get_session", return_value=session),
                        patch.object(backend_api, "ERROR_MEMORY", ErrorMemory()),
                        patch.object(backend_api.db, "create_session") as create_session,
                        patch.object(backend_api.db, "save_chat_message"),
                        patch.object(backend_api, "maybe_generate_session_title"),
                    ):
                        client = TestClient(backend_api.app)
                        try:
                            result = client.post(endpoint, data=form)
                        finally:
                            client.close()
                    self.assertEqual(result.status_code, 200)
                    if endpoint.endswith("/stream"):
                        self.assertIn("event: done", result.text)
                        self.assertNotIn("event: error", result.text)
                    else:
                        self.assertEqual(result.json()["answer"], "Test result")
                    expected = override or backend_api.DEFAULT_SYSTEM_PROMPT
                    self.assertIsInstance(handle.messages[0], SystemMessage)
                    self.assertTrue(handle.messages[0].content.startswith(expected))
                    self.assertIn(backend_api.RAG_INSTRUCTIONS, handle.messages[0].content)
                    self.assertEqual(create_session.call_args.kwargs["system_prompt"], expected)
                    self.assertFalse(session.lock.locked())


if __name__ == "__main__":
    unittest.main()
