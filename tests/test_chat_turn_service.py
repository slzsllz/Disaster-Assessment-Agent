"""Contract tests for the shared chat-turn lifecycle and its two HTTP adapters."""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import uuid

from langchain_core.messages import AIMessage

import backend_api as api


class FakeHandle:
    def __init__(self, answer: str = "测试结果", error: Exception | None = None):
        self.response = {"messages": [AIMessage(content=answer)]}
        self.error = error
        self.invocations = 0

    def invoke(self, messages, config):
        self.invocations += 1
        return self.response

    def stream(self, messages, config):
        self.invocations += 1
        yield {"type": "delta", "text": "草稿"}
        if self.error is not None:
            yield {"type": "error", "error": self.error}
        else:
            yield {"type": "final", "response": self.response}


def event_data(chunks: list[bytes], event: str) -> list[dict]:
    decoded = b"".join(chunks).decode("utf-8")
    return [
        json.loads(block.split("data: ", 1)[1])
        for block in decoded.split("\n\n")
        if block.startswith(f"event: {event}\n")
    ]


class ChatTurnServiceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.db = Mock()
        self.db.complete_turn.return_value = 7
        self.db.fail_turn.return_value = True
        self.stack.enter_context(patch.object(api, "db", self.db))
        self.stack.enter_context(
            patch.object(api, "get_session", side_effect=lambda _id: self.session)
        )
        self.stack.enter_context(
            patch.object(api, "retrieve_turn_context", return_value=("", []))
        )
        self.stack.enter_context(
            patch.object(api.ERROR_MEMORY, "lookup_all", return_value=[])
        )

        def start(session, message, system_prompt, files):
            session.messages.append({"role": "user", "content": message})
            return "turn-1", [], [], message

        self.stack.enter_context(patch.object(api, "start_chat_turn", side_effect=start))
        self.new_session()

    def new_session(self, handle: FakeHandle | None = None):
        self.session = api.ChatSession(
            session_id=str(uuid.uuid4()), temp_dir=Path("/tmp/chat-turn-test"),
            handle=handle or FakeHandle(),
        )

    def request(self, streaming: bool, message: str = "测试",
                required_web_search: bool = False):
        kwargs = dict(
            session_id=self.session.session_id, message=message,
            system_prompt=api.DEFAULT_SYSTEM_PROMPT,
            recursion_limit=40, max_execution_time=600,
            show_trace=True, required_web_search=required_web_search, files=None,
        )
        if streaming:
            with patch.object(api, "StreamingResponse", side_effect=lambda body, **_: body):
                return [event.encode("utf-8") for event in api.chat_stream(**kwargs)]
        return api.chat(**kwargs)

    def test_stream_and_json_publish_the_same_reviewed_result(self):
        normal = self.request(streaming=False)
        self.new_session()
        chunks = self.request(streaming=True)
        done = event_data(chunks, "done")

        self.assertEqual(len(done), 1)
        for key in ("answer", "tool_calls", "images", "files", "geometry",
                    "legend", "trace", "references", "review"):
            self.assertEqual(normal[key], done[0][key], key)
        self.assertEqual(normal["answer"], "测试结果")
        self.assertEqual(self.db.complete_turn.call_count, 2)
        self.assertEqual(self.session.messages[-1],
                         {"role": "assistant", "content": "测试结果"})

    def test_blocked_review_cannot_save_assessment_or_report(self):
        with patch.object(
            api, "execute_review",
            return_value=("证据不足", [], [], {"status": "blocked", "issues": []}),
        ), patch.object(api, "save_tool_assessments") as save_assessment, patch.object(
            api, "generate_and_store_pdf_report"
        ) as generate_report:
            result = self.request(streaming=False, message="请生成报告")

        self.assertEqual(result["review"]["status"], "blocked")
        self.assertIsNone(result["report"])
        save_assessment.assert_not_called()
        generate_report.assert_not_called()

    def test_requested_report_uses_the_same_outcome_in_both_modes(self):
        report = {"name": "report.pdf", "url": "/api/files/report"}
        with patch.object(api, "generate_and_store_pdf_report", return_value=report) as generate:
            normal = self.request(False, "请生成报告")
            self.new_session()
            chunks = self.request(True, "请生成报告")

        self.assertEqual(normal["report"], report)
        self.assertEqual(event_data(chunks, "report"), [report])
        self.assertEqual(generate.call_count, 2)

    def test_stream_error_fails_turn_once_and_releases_session_lock(self):
        self.new_session(FakeHandle(error=RuntimeError("tool failed")))
        chunks = self.request(streaming=True)
        errors = event_data(chunks, "error")

        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["turn_id"], "turn-1")
        self.db.fail_turn.assert_called_once()
        self.assertTrue(self.session.lock.acquire(blocking=False))
        self.session.lock.release()

    def test_closed_stream_marks_the_turn_cancelled(self):
        turn = api.ChatTurnService(
            self.session, "测试", api.DEFAULT_SYSTEM_PROMPT, 40, 600, False, False
        )
        self.session.lock.acquire()
        try:
            turn.start(None)
            events = api.iter_chat_events(turn)
            next(events)
            events.close()
            self.assertFalse(self.session.lock.locked())
        finally:
            if self.session.lock.locked():
                self.session.lock.release()

        self.db.fail_turn.assert_called_once_with(
            self.session.session_id, "turn-1", "请求中断，任务未完成。",
            code="cancelled",
        )

    def test_required_search_failure_blocks_both_response_modes(self):
        with patch.object(
            api, "run_required_web_search",
            side_effect=api.RequiredWebSearchError("没有可用来源"),
        ):
            normal = self.request(False, "最新洪水", required_web_search=True)
            self.assertIn("没有可用来源", normal["error"])
            self.assertEqual(self.session.handle.invocations, 0)

            self.new_session()
            chunks = self.request(True, "最新洪水", required_web_search=True)
            errors = event_data(chunks, "error")
            self.assertEqual(len(errors), 1)
            self.assertIn("没有可用来源", errors[0]["error"])
            self.assertEqual(self.session.handle.invocations, 0)

        self.db.complete_turn.assert_not_called()
        self.assertEqual(self.db.fail_turn.call_count, 2)


if __name__ == "__main__":
    unittest.main()
