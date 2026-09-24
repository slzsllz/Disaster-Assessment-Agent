import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from agent.error_feedback import ToolErrorFeedback, tool_error
from agent.error_memory import ErrorMemory


class CandidateStore:
    def __init__(self):
        self.candidates = []

    def save_error_recovery_candidate(self, **candidate):
        self.candidates.append(candidate)
        return True


class ErrorFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.store = CandidateStore()
        self.feedback = ToolErrorFeedback(ErrorMemory(), db=self.store)

    def test_tool_error_recognizes_explicit_failures(self):
        self.assertIsNotNone(tool_error(ToolMessage(content='{"error": "bad bbox"}', tool_call_id="1")))
        self.assertIsNone(tool_error(ToolMessage(content='{"error_count": 0}', tool_call_id="2")))

    def test_relevant_hint_only_after_tool_failure(self):
        call = AIMessage(content="", tool_calls=[{"id": "a", "name": "read_raster", "args": {"path": "x"}}])
        result = ToolMessage(content="AttributeError: object has no attribute 'read'", tool_call_id="a", status="error")
        updated = self.feedback.before_model({"messages": [call, result]})
        self.assertIn("has no attribute 'read'", updated["llm_input_messages"][0].content)
        self.assertEqual(len(updated["llm_input_messages"]), 3)

    def test_changed_arguments_and_success_record_candidate_once(self):
        messages = [
            AIMessage(content="", tool_calls=[{"id": "a", "name": "raster", "args": {"path": "/secret/a"}}]),
            ToolMessage(content="FileNotFoundError: /secret/a", tool_call_id="a", status="error"),
            AIMessage(content="", tool_calls=[{"id": "b", "name": "raster", "args": {"path": "/secret/b"}}]),
            ToolMessage(content=json.dumps({"result": "ok"}), tool_call_id="b"),
        ]
        self.feedback.before_model({"messages": messages})
        self.feedback.before_model({"messages": messages})
        self.assertEqual(len(self.store.candidates), 1)
        candidate = self.store.candidates[0]
        self.assertEqual(candidate["changed_fields"], ["path"])
        self.assertNotIn("/secret", candidate["error_signature"])

    def test_same_arguments_do_not_record_recovery(self):
        messages = [
            AIMessage(content="", tool_calls=[{"id": "a", "name": "raster", "args": {"path": "a"}}]),
            ToolMessage(content="error: temporary", tool_call_id="a", status="error"),
            AIMessage(content="", tool_calls=[{"id": "b", "name": "raster", "args": {"path": "a"}}]),
            ToolMessage(content="ok", tool_call_id="b"),
        ]
        self.feedback.before_model({"messages": messages})
        self.assertFalse(self.store.candidates)

    def test_error_memory_uses_file_when_database_is_down(self):
        class OfflineStore:
            def load_all_errors(self):
                return {}

            def save_error(self, pattern, fix):
                return False

            def increment_error_hit(self, pattern):
                return False

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.json"
            memory = ErrorMemory(path=path, db=OfflineStore())
            self.assertTrue(memory.record("TestFailure", "Try a smaller input"))
            self.assertEqual(json.loads(path.read_text())["TestFailure"], "Try a smaller input")


if __name__ == "__main__":
    unittest.main()
