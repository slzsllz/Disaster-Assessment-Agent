"""Contract tests for the shared chat-turn lifecycle and its two HTTP adapters."""

from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import unittest
import tempfile
from unittest.mock import Mock, patch
import uuid

from langchain_core.messages import AIMessage
from langchain_core.messages import ToolMessage

import backend_api as api
import agent.auth as auth
from agent.artifacts import ArtifactStore
from agent.web_fetch import fetch_web
from agent.web_search import search_result_references
from agent.disaster_feeds import query_disaster_feed


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
        self.stack.enter_context(patch.object(auth, "db", self.db))
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
            user={"id": str(uuid.uuid4())},
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

    def test_official_feed_citations_reach_review_without_creating_a_remote_sensing_assessment(self):
        payload = {"type": "FeatureCollection", "features": [{
            "id": "official-quake", "geometry": {"type": "Point", "coordinates": [104.65, 28.5, 5]},
            "properties": {"time": 1791367200000, "mag": 3.0, "title": "Official earthquake",
                           "url": "https://earthquake.usgs.gov/earthquakes/eventpage/official-quake"},
        }]}
        with patch("agent.disaster_feeds._get", return_value=json.dumps(payload).encode()):
            result = query_disaster_feed("usgs", start_time="2026-10-07", end_time="2026-10-09")
        self.session.handle.response = {"messages": [
            AIMessage(content="", tool_calls=[{"name": "disaster_feed", "args": {"provider": "usgs"}, "id": "feed-1"}]),
            ToolMessage(content=json.dumps(result), name="disaster_feed", tool_call_id="feed-1"),
            AIMessage(content="官方地震记录：震级 3.0，深度 5 公里。"),
        ]}

        def review(handle, question, response, answer, uploaded, output_paths, references, **kwargs):
            self.assertEqual(references[0]["structured_data"]["magnitude"], 3.0)
            self.assertEqual(references[0]["evidence_type"], "official_record")
            self.assertEqual(output_paths, [])
            return answer, [], [], {"status": "passed", "issues": [], "evidence": [], "version": api.REVIEW_VERSION}

        with patch.object(api, "execute_review", side_effect=review):
            outcome = self.request(False, "查询 USGS 官方地震目录")
        self.assertEqual(outcome["references"][0]["structured_data"]["depth_km"], 5)
        self.db.save_assessment.assert_not_called()

    def test_optional_feed_configuration_reaches_mcp_children(self):
        with patch.dict(os.environ, {"FIRMS_MAP_KEY": "test-map-key", "CHINA_ALERTS_CAP_URL": "https://www.12379.cn/cap.xml",
                                     "UNRELATED_TEST_SECRET": "must-not-inherit"}):
            env = api.build_mcp_child_env(self.session.session_id)
        self.assertEqual(env["FIRMS_MAP_KEY"], "test-map-key")
        self.assertEqual(env["CHINA_ALERTS_CAP_URL"], "https://www.12379.cn/cap.xml")
        self.assertNotIn("UNRELATED_TEST_SECRET", env)

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

    def test_fetched_evidence_reuses_artifact_and_upgrades_citation_without_creating_assessment(self):
        with tempfile.TemporaryDirectory() as directory:
            self.session.temp_dir = Path(directory) / "session"
            store = ArtifactStore(Path(directory) / "artifacts", self.db)
            saved = {}
            self.db.save_artifact.side_effect = lambda record: saved.update({record["id"]: record})
            self.db.get_artifact.side_effect = lambda artifact_id: saved.get(artifact_id)
            url = "https://example.com/alert"

            def get_page(_url, **kwargs):
                kwargs["response_info"].update(url=url, content_type="text/html; charset=utf-8")
                return b"<article>Public alert body with verified report data.</article>"

            with patch("agent.web_fetch._get", side_effect=get_page):
                fetched = fetch_web([url], output_dir=self.session.output_dir,
                                    session_id=self.session.session_id, artifact_store=store)
            path = fetched["results"][0]["output_path"]
            artifact_id = fetched["results"][0]["artifact"]["id"]
            self.session.handle.response = {"messages": [
                AIMessage(content="", tool_calls=[{"name": "web_fetch", "args": {"urls": [url]},
                                                   "id": "fetch-1", "type": "tool_call"}]),
                ToolMessage(content=json.dumps(fetched), tool_call_id="fetch-1", name="web_fetch"),
                AIMessage(content="已核实"),
            ]}
            search_refs = search_result_references({"success": True, "results": [{"url": url, "snippet": "thin snippet"}]})

            def review(*args, **kwargs):
                references = args[6]
                self.assertTrue(references[0]["body_verified"])
                self.assertIn("Public alert body", references[0]["body_excerpt"])
                return "已核实", [], [path], {"status": "passed", "issues": []}

            with patch.object(api, "ARTIFACT_STORE", store), patch.object(api, "execute_review", side_effect=review), patch.object(
                api, "retrieve_turn_context", return_value=("", search_refs)
            ):
                result = self.request(False, "打开这个链接核实")

            self.assertEqual(result["files"][0]["id"], artifact_id)
            self.assertEqual(len(result["references"]), 1)
            self.assertEqual(result["references"][0]["artifact_id"], artifact_id)
            self.assertNotIn("body_excerpt", result["references"][0])
            self.db.save_artifact.assert_called_once()
            self.db.save_assessment.assert_not_called()


if __name__ == "__main__":
    unittest.main()
