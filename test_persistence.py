import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from agent.artifacts import ArtifactStore
from agent.retrieval import RetrievalDocument, RetrievalHit
import backend_api


class FakeArtifactDb:
    def __init__(self):
        self.records = {}

    def save_artifact(self, record):
        self.records[record["id"]] = record

    def get_artifact(self, artifact_id):
        return self.records.get(artifact_id)


class PersistenceTests(unittest.TestCase):
    def test_artifact_survives_store_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.txt"
            source.write_text("a durable result", encoding="utf-8")
            database = FakeArtifactDb()
            session_id = str(uuid.uuid4())
            store = ArtifactStore(root / "artifacts", database)
            record = store.put_file(source, session_id, "tool_output")
            source.unlink()

            restored = ArtifactStore(root / "artifacts", database)
            self.assertEqual(
                restored.resolve(database.get_artifact(record["id"])).read_text(),
                "a durable result",
            )
            self.assertEqual(len(record["sha256"]), 64)
            restored.delete_session_files(session_id)
            with self.assertRaises(FileNotFoundError):
                restored.resolve(record)

    def test_tool_result_is_saved_with_artifact_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            result_file = Path(directory) / "result.txt"
            result_file.write_text("metric=2", encoding="utf-8")
            tool_message = ToolMessage(
                content=json.dumps({"metric": 2, "summary_path": str(result_file)}),
                name="calculate_batch_ndvi", tool_call_id="call-1",
            )
            fake_db = SimpleNamespace(save_assessment=lambda **kwargs: kwargs)
            saved = []
            fake_db.save_assessment = lambda **kwargs: saved.append(kwargs) or 1
            with patch.object(backend_api, "db", fake_db):
                backend_api.save_tool_assessments(
                    {"messages": [tool_message]}, str(uuid.uuid4()),
                    str(uuid.uuid4()), {str(result_file): {"id": "artifact-1"}},
                )
            self.assertEqual(saved[0]["task"], "calculate_batch_ndvi")
            self.assertEqual(saved[0]["summary"]["metric"], 2)
            self.assertEqual(saved[0]["artifact_ids"], ["artifact-1"])


@unittest.skipUnless(os.getenv("RUN_DB_INTEGRATION") == "1", "requires local PostgreSQL")
class DatabaseIntegrationTests(unittest.TestCase):
    def test_chat_stream_restart_artifact_and_assessment(self):
        from dotenv import load_dotenv
        from fastapi.testclient import TestClient
        from agent.db import db

        load_dotenv(".env")
        self.assertTrue(db.migrate())
        session_id = str(uuid.uuid4())

        class FakeHandle:
            def __init__(self, config, temp_dir, session_id=""):
                self.output_dir = Path(temp_dir) / "out"
                self.output_dir.mkdir(parents=True, exist_ok=True)

            def invoke(self, messages, config=None):
                result_file = self.output_dir / "result.txt"
                result_file.write_text("metric=2", encoding="utf-8")
                return {"messages": [
                    AIMessage(content="", tool_calls=[{
                        "id": "call-1", "name": "calculate_batch_ndvi", "args": {},
                    }]),
                    ToolMessage(
                        content=json.dumps({"metric": 2, "summary_path": str(result_file)}),
                        name="calculate_batch_ndvi", tool_call_id="call-1",
                    ),
                    AIMessage(content="<Conclusion>完成</Conclusion>"),
                ]}

            def stream(self, messages, config=None):
                yield {"type": "final", "response": self.invoke(messages, config)}

            def review_answer_and_artifacts(self, raw, uploaded, outputs):
                return raw

            def close(self):
                pass

        try:
            tool_reference = RetrievalHit(
                RetrievalDocument(
                    "tool", "calculate_batch_ndvi", None, "calculate_batch_ndvi",
                    "NDVI tool guidance", {"description": "NDVI tool guidance"},
                ), 0.9, "embedding+keyword",
            )
            with (patch.object(backend_api, "AgentHandle", FakeHandle),
                  patch.object(backend_api.RETRIEVAL, "search_tools", return_value=[tool_reference]),
                  TestClient(backend_api.app) as client):
                first = client.post(
                    "/api/chat",
                    data={"session_id": session_id, "message": "分析测试"},
                    files={"files": ("input.txt", b"sample", "text/plain")},
                )
                self.assertEqual(first.status_code, 200)
                self.assertIn("完成", first.json()["answer"])
                self.assertEqual(len(first.json()["files"]), 1)
                self.assertEqual(first.json()["references"][0]["source_id"], "calculate_batch_ndvi")
                self.assertEqual(client.get(first.json()["files"][0]["url"]).content, b"metric=2")
                history = client.get(f"/api/sessions/{session_id}/messages").json()["messages"]
                self.assertEqual(client.get(history[0]["attachments"][0]["url"]).content, b"sample")
                self.assertEqual(history[1]["references"][0]["source_id"], "calculate_batch_ndvi")

                backend_api.SESSIONS.pop(session_id, None)
                self.assertEqual(len(backend_api.get_session(session_id).messages), 2)
                second = client.post(
                    "/api/chat/stream",
                    data={"session_id": session_id, "message": "继续分析"},
                )
                self.assertIn("event: done", second.text)
                self.assertIn('"references":', second.text)
                self.assertIn("calculate_batch_ndvi", second.text)
                turns = db.list_turns(session_id)
                self.assertEqual([row["status"] for row in turns], ["completed", "completed"])
                self.assertEqual(len(db.query_assessments(session_id=session_id)), 2)
        finally:
            db.delete_session(session_id)
            backend_api.ARTIFACT_STORE.delete_session_files(session_id)
            backend_api.SESSIONS.pop(session_id, None)


if __name__ == "__main__":
    unittest.main()
