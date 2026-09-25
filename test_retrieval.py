import os
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx
from langchain_core.messages import HumanMessage

from agent.retrieval import EmbeddingClient, RetrievalService
from agent.tool_router import ToolRouter
from backend_api import build_messages


class FakeDb:
    def __init__(self):
        self.vectors = {}
        self.artifacts = []

    def load_retrieval_embeddings(self, source_type, source_ids, session_id=None):
        return {
            source_id: self.vectors[(source_type, source_id)]
            for source_id in source_ids
            if (source_type, source_id) in self.vectors
            and self.vectors[(source_type, source_id)]["session_id"] == session_id
        }

    def upsert_retrieval_embeddings(self, rows):
        for row in rows:
            self.vectors[(row["source_type"], row["source_id"])] = row

    def list_session_artifacts(self, session_id):
        return [row for row in self.artifacts if row["session_id"] == session_id]


class FakeEmbedder:
    model_name = "test-model"

    def __init__(self, failure=False):
        self.failure = failure
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        if self.failure:
            raise httpx.ConnectError("unavailable")
        return [[1.0, 0.0] if "洪水" in text else [0.0, 1.0] for text in texts]


class RetrievalTests(unittest.TestCase):
    def test_embed_client_accepts_server_contract_and_checks_dimension(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={
                "count": 2, "dim": 2, "vectors": [[3, 4], [0, 5]],
            })

        client = EmbeddingClient("http://example.test:8000")
        original_client = httpx.Client

        def mocked_client(*args, **kwargs):
            return original_client(transport=httpx.MockTransport(handler), **kwargs)

        from unittest.mock import patch
        with patch("agent.retrieval.httpx.Client", side_effect=mocked_client):
            vectors = client.embed(["洪水", "火灾"])
        self.assertEqual(len(vectors), 2)
        self.assertAlmostEqual(vectors[0][0], 0.6)
        self.assertEqual(requests[0].url.path, "/embed")
        self.assertIn(b'"normalize":true', requests[0].content)

    def test_tool_vectors_are_reused_and_keyword_fallback_works(self):
        database = FakeDb()
        embedder = FakeEmbedder()
        service = RetrievalService(database, None, embedder)
        tools = [
            SimpleNamespace(name="extract_flood_inundation", description="Extract flood area"),
            SimpleNamespace(name="detect_fire_burned_area_change", description="Detect fire damage"),
        ]
        first = service.search_tools("洪水淹没分析", tools)
        self.assertEqual(first[0].document.source_id, "extract_flood_inundation")
        self.assertEqual(len(database.vectors), 2)
        self.assertEqual(first[0].method, "embedding+keyword")
        calls_after_first = embedder.calls
        service.search_tools("洪水淹没分析", tools)
        self.assertEqual(embedder.calls, calls_after_first + 1)

        unavailable = RetrievalService(database, None, FakeEmbedder(failure=True))
        fallback = unavailable.search_tools("洪水淹没分析", tools)
        self.assertEqual(fallback[0].document.source_id, "extract_flood_inundation")
        self.assertEqual(fallback[0].method, "keyword")

    def test_artifact_search_is_session_scoped_and_excludes_new_upload(self):
        first_session = str(uuid.uuid4())
        other_session = str(uuid.uuid4())
        database = FakeDb()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, session_id in enumerate((first_session, first_session, other_session)):
                file = root / f"file-{index}"
                file.write_text("data")
                database.artifacts.append({
                    "id": str(uuid.uuid4()), "session_id": session_id,
                    "kind": "upload", "original_name": f"洪水影像{index}.tif",
                    "mime_type": "image/tiff", "path": file,
                })

            class Store:
                @staticmethod
                def resolve(row):
                    return row["path"]

            service = RetrievalService(database, Store(), FakeEmbedder())
            excluded = {database.artifacts[0]["id"]}
            hits = service.search_artifacts("使用上次的洪水影像", first_session, excluded)
            self.assertEqual([hit.document.source_id for hit in hits], [database.artifacts[1]["id"]])
            self.assertEqual(hits[0].public()["url"], f"/api/files/{database.artifacts[1]['id']}")

    def test_retrieval_context_does_not_replace_router_question(self):
        messages = build_messages(
            [{"role": "user", "content": "请分析洪水"}], "System prompt",
            "Retrieved reference data: fire tool",
        )
        self.assertEqual(messages[-1].content, "请分析洪水")
        self.assertEqual(messages[-2].name, "retrieval_context")

        class LLM:
            async def ainvoke(self, messages):
                self.context = messages[-1].content
                return SimpleNamespace(content='{"tools": []}')

        import asyncio
        llm = LLM()
        router = ToolRouter(llm, [SimpleNamespace(name="extract_flood_inundation", description="Flood")])
        asyncio.run(router.select(messages))
        self.assertNotIn("fire tool", llm.context)


@unittest.skipUnless(os.getenv("RUN_DB_INTEGRATION") == "1", "requires local PostgreSQL")
class RetrievalDatabaseTests(unittest.TestCase):
    def test_vectors_and_references_survive_restart_and_respect_session(self):
        from dotenv import load_dotenv
        from agent.db import db

        load_dotenv(".env")
        self.assertTrue(db.migrate())
        session_id = str(uuid.uuid4())
        another_session = str(uuid.uuid4())
        source_id = str(uuid.uuid4())
        try:
            db.create_session(session_id)
            db.create_session(another_session)
            db.upsert_retrieval_embeddings([{
                "source_type": "artifact", "source_id": source_id,
                "session_id": session_id, "content_sha256": "abc",
                "model_name": "test-model", "dimensions": 2,
                "embedding": [0.6, 0.8],
            }])
            found = db.load_retrieval_embeddings("artifact", [source_id], session_id)
            self.assertAlmostEqual(found[source_id]["embedding"][0], 0.6, places=5)
            self.assertEqual(
                db.load_retrieval_embeddings("artifact", [source_id], another_session), {},
            )
            turn_id = str(uuid.uuid4())
            db.start_turn(session_id, turn_id, "test", "test", [])
            references = [{"source_type": "artifact", "source_id": source_id,
                           "title": "test.tif", "url": f"/api/files/{source_id}"}]
            db.complete_turn(session_id, turn_id, "answer", "answer", [], [],
                             [], 0.1, [], references)
            messages = db.load_chat_messages_strict(session_id)
            self.assertEqual(messages[-1]["retrieval_refs"], references)
        finally:
            db.delete_session(session_id)
            db.delete_session(another_session)


if __name__ == "__main__":
    unittest.main()
