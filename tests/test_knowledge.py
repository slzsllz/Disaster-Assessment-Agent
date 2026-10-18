import asyncio
import io
import json
import os
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import httpx
from docx import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langgraph.prebuilt import create_react_agent
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from agent.db import psycopg, sql, _build_dsn
from agent.knowledge import (
    EmbeddingClient, EmbeddingSettings, KnowledgeBase, KnowledgeError, NeedsOCR,
    chunk_sections, extract_rule_references, make_search_tool, parse_document,
)


def word_bytes(text="Flood response: protect hospitals and evacuate flooded roads."):
    doc = Document()
    doc.add_heading("Emergency response rules", 0)
    doc.add_paragraph(text)
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Water depth (m)"
    table.cell(0, 1).text = "Required action"
    output = io.BytesIO()
    doc.save(output)
    return output.getvalue()


class ParserTests(unittest.TestCase):
    def test_docx_paragraphs_and_tables_keep_units(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "rule.docx"
            path.write_bytes(word_bytes())
            sections = parse_document(path)
            self.assertIn("Water depth (m)", "\n".join(text for _, text in sections))
            self.assertTrue(all("正文" in locator for locator, _ in sections))

    def test_pdf_preserves_real_page_numbers(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "rule.pdf"
            writer = PdfWriter()
            for text in ["First page flood rules", "Second page emergency routes"]:
                page = writer.add_blank_page(width=200, height=100)
                font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                         NameObject('/Subtype'): NameObject('/Type1'),
                                         NameObject('/BaseFont'): NameObject('/Helvetica')})
                page[NameObject('/Resources')] = DictionaryObject({
                    NameObject('/Font'): DictionaryObject({NameObject('/F1'): font}),
                })
                stream = DecodedStreamObject()
                stream.set_data(f'BT /F1 12 Tf 10 50 Td ({text}) Tj ET'.encode())
                page[NameObject('/Contents')] = stream
            writer.write(path)
            self.assertEqual([loc for loc, _ in parse_document(path)], ["第 1 页", "第 2 页"])

    def test_image_only_pdf_requires_ocr(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "scan.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            writer.write(path)
            with self.assertRaises(NeedsOCR):
                parse_document(path)

    def test_chunks_are_bounded_and_keep_source_locator(self):
        text = "洪水处置规则：道路受淹后先核实水深，不能只根据面积判断危险。" * 100
        chunks = chunk_sections([("第 7 页", text)])
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c["content"]) <= 900 and c["locator"] == "第 7 页" for c in chunks))


class EmbeddingTests(unittest.TestCase):
    def test_independent_endpoint_never_receives_chat_key(self):
        with patch.dict(os.environ, {"RAG_EMBEDDING_BASE_URL": "https://other.example/v1"}, clear=True):
            settings = EmbeddingSettings.from_env({
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key": "chat-secret",
            })
        self.assertEqual(settings.api_key, "")
        self.assertFalse(settings.configured)

    def test_native_endpoint_can_reuse_existing_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = EmbeddingSettings.from_env({
                "base_url": "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", "api_key": "chat-secret",
            })
        self.assertTrue(settings.configured)
        self.assertEqual(settings.dimensions, 1024)

    def test_embedding_order_normalization_and_query_type(self):
        client = EmbeddingClient(EmbeddingSettings("https://example.test/v1", "secret", dimensions=2))
        seen = []

        def handle(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"data": [
                {"index": 1, "embedding": [0, 5]}, {"index": 0, "embedding": [3, 4]},
            ]})

        original_client = httpx.Client
        with patch("agent.knowledge.httpx.Client", side_effect=lambda **kw: original_client(transport=httpx.MockTransport(handle), **kw)):
            vectors = client.embed(["a", "b"], query=True)
        self.assertEqual(vectors, [[.6, .8], [0, 1]])
        self.assertEqual(seen[0]["text_type"], "query")

    def test_quota_failure_is_not_reported_as_success(self):
        client = EmbeddingClient(EmbeddingSettings("https://example.test/v1", "secret", dimensions=2))
        original_client = httpx.Client
        with patch("agent.knowledge.httpx.Client", side_effect=lambda **kw: original_client(
            transport=httpx.MockTransport(lambda req: httpx.Response(403, json={"error": {"code": "insufficient_quota"}})), **kw,
        )):
            with self.assertRaisesRegex(KnowledgeError, "额度不足"):
                client.embed(["test"])

    def test_invalid_vector_rejected(self):
        client = EmbeddingClient(EmbeddingSettings("https://example.test/v1", "secret", dimensions=2))
        original_client = httpx.Client
        with patch("agent.knowledge.httpx.Client", side_effect=lambda **kw: original_client(
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"data": [{"index": 0, "embedding": [0, 0]}]})), **kw,
        )):
            with self.assertRaisesRegex(KnowledgeError, "无效"):
                client.embed(["test"])


class RuleToolTests(unittest.TestCase):
    def test_original_question_is_injected_without_exposing_config_to_model(self):
        class Probe:
            def search(self, *args, **kwargs):
                return {"results": [], "original": kwargs["question"], "query": args[0]}
        tool = make_search_tool(Probe())
        self.assertNotIn("config", tool.args)
        result = json.loads(tool.invoke({"query": "compressed keywords"}, config={
            "configurable": {"rag_question": "What does my test document actually say?"},
        }))
        self.assertEqual(result["query"], "compressed keywords")
        self.assertEqual(result["original"], "What does my test document actually say?")

    def test_only_cited_retrieved_evidence_becomes_reference(self):
        refs = [{"citation": "[规则-a]", "id": "a"}, {"citation": "[规则-b]", "id": "b"}]
        response = {"messages": [ToolMessage(
            content=json.dumps({"results": refs, "status": "sufficient"}), name="search_rule_documents", tool_call_id="search",
        )]}
        self.assertEqual(extract_rule_references(response, "依据[规则-a]。"), [refs[0]])

    def test_rejected_evidence_cannot_be_saved_as_a_reference(self):
        response = {"messages": [ToolMessage(
            content=json.dumps({"status": "insufficient", "results": [{"citation": "[规则-a]"}]}),
            name="search_rule_documents", tool_call_id="search",
        )]}
        self.assertEqual(extract_rule_references(response, "依据[规则-a]。"), [])

    def test_retrieval_failure_is_explicit_to_agent(self):
        class Missing:
            def search(self, *args, **kwargs):
                raise KnowledgeError("Embedding 额度不足")
        payload = json.loads(make_search_tool(Missing()).invoke({"query": "洪水规则"}))
        self.assertEqual(payload["results"], [])
        self.assertIn("额度不足", payload["error"])


class BackendCitationTests(unittest.TestCase):
    def test_normal_and_stream_chat_return_and_save_rule_references(self):
        import backend_api
        reference = {"id": "chunk-test", "document_id": str(uuid.uuid4()),
                     "filename": "防汛规则.docx", "citation": "[规则-test]", "locator": "第 1 页",
                     "content": "保障医院通行。", "url": "/api/knowledge/documents/test/file"}
        response = {"messages": [
            ToolMessage(content=json.dumps({"results": [reference], "status": "sufficient"}), name="search_rule_documents", tool_call_id="rules-test"),
            AIMessage(content="<Conclusion>应保障医院通行[规则-test]。</Conclusion>"),
        ]}

        class Handle:
            def invoke(self, *args, **kwargs):
                assert kwargs["config"]["configurable"]["rag_question"] == "防汛规则"
                return response

            def stream(self, *args, **kwargs):
                assert kwargs["config"]["configurable"]["rag_question"] == "防汛规则"
                yield {"type": "final", "response": response}

            def review_answer_and_artifacts(self, raw, uploaded, outputs, **kwargs):
                assert kwargs["rule_references"] == [reference]
                return raw

        for streaming in (False, True):
            with self.subTest(streaming=streaming), tempfile.TemporaryDirectory() as root:
                session = backend_api.ChatSession(session_id="test", temp_dir=Path(root), handle=Handle())
                with (
                    patch.object(backend_api, "get_session", return_value=session),
                    patch.object(backend_api.db, "create_session"),
                    patch.object(backend_api.db, "save_chat_message") as save,
                    patch.object(backend_api, "maybe_generate_session_title"),
                ):
                    fn = backend_api.chat_stream if streaming else backend_api.chat
                    result = fn(session_id="test", message="防汛规则", system_prompt="Test",
                                recursion_limit=40, max_execution_time=600, show_trace=False, files=None)
                    if streaming:
                        async def consume():
                            return [chunk async for chunk in result.body_iterator]
                        done = next(chunk for chunk in asyncio.run(consume()) if "event: done" in chunk)
                        payload = json.loads(done.split("data: ", 1)[1])
                    else:
                        payload = result
                    self.assertEqual(payload["retrieval_refs"], [reference])
                    self.assertEqual(save.call_args.kwargs["retrieval_refs"], [reference])
                    self.assertEqual(backend_api._serialize_message({
                        "id": 1, "role": "assistant", "content": payload["answer"], "retrieval_refs": [reference],
                    })["retrieval_refs"], [reference])


class FakeEmbedder:
    """Deterministic vectors for integration tests, never used in production."""
    def __init__(self):
        self.calls = 0

    def embed(self, texts, *, query=False):
        self.calls += 1
        return [[1., 0.] if "flood" in text.lower() else [0., 1.] for text in texts]


class FakeVerifier:
    """Isolate storage tests from both paid verification services."""
    def verify(self, query, candidates, top_k, region, disaster_type):
        return {"results": candidates[:top_k], "status": "sufficient", "message": "Test evidence accepted"}

    def status(self):
        return {"rerank_model": "fake", "judge_model": "fake", "rerank_configured": True, "judge_configured": True}


@unittest.skipUnless(os.getenv("RAG_TEST_DB") == "1", "Set RAG_TEST_DB=1 for isolated PostgreSQL tests")
class DatabaseIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dotenv import load_dotenv
        load_dotenv(".env", override=True)
        cls.schema = "test_knowledge_" + uuid.uuid4().hex
        cls.dsn = _build_dsn()
        with psycopg.connect(cls.dsn, connect_timeout=3) as conn:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(cls.schema)))

    @classmethod
    def tearDownClass(cls):
        with psycopg.connect(cls.dsn, connect_timeout=3) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(cls.schema)))

    def setUp(self):
        test = self

        class TestDatabase:
            def health_check(self):
                return True

            @contextmanager
            def _conn(self):
                with psycopg.connect(test.dsn, connect_timeout=3) as conn:
                    conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(test.schema)))
                    yield conn

        self.database = TestDatabase()
        with self.database._conn() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS chat_messages (id integer)")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.embedder = FakeEmbedder()
        self.knowledge = KnowledgeBase(Path(self.temporary.name), EmbeddingSettings(
            "https://test.invalid/v1", "test", dimensions=2,
        ), database=self.database, embedder=self.embedder, verifier=FakeVerifier())
        self.knowledge.ensure_schema()
        with self.database._conn() as conn:
            conn.execute("TRUNCATE knowledge_documents CASCADE")

    def add_document(self, filename="flood.docx", text="Flood response: protect hospitals.", **meta):
        result = self.knowledge.upload(filename, word_bytes(text), meta)
        self.assertTrue(self.knowledge.process_one())
        return result["document"]["id"]

    def test_ingest_duplicate_search_disable_delete(self):
        document_id = self.add_document(region="深圳", disaster_type="洪水")
        self.assertEqual(self.knowledge.get_document(document_id)["status"], "ready")
        duplicate = self.knowledge.upload("same.docx", self.knowledge.document_path(
            self.knowledge.get_document(document_id)).read_bytes(), {})
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(duplicate["document"]["id"], document_id)
        results = self.knowledge.search("flood emergency", region="深圳")["results"]
        self.assertTrue(results)
        self.assertEqual(results[0]["filename"], "flood.docx")
        self.assertIn("规则-", results[0]["citation"])
        self.assertEqual(self.knowledge.search("flood", region="北京")["results"], [])
        self.knowledge.update_document(document_id, enabled=False)
        self.assertEqual(self.knowledge.search("flood")["results"], [])
        self.knowledge.delete(document_id)
        self.assertFalse(list(Path(self.temporary.name).iterdir()))

    def test_no_library_and_no_match_do_not_force_results(self):
        self.assertEqual(self.knowledge.search("flood")["results"], [])
        self.assertEqual(self.embedder.calls, 0)
        self.add_document("earthquake.docx", "Earthquake shelter plan")
        self.assertEqual(self.knowledge.search("flood")["results"], [])

    def test_verification_rejection_and_outage_never_return_raw_candidates(self):
        from agent.rag_verification import RetrievalVerificationError
        self.add_document()
        with patch.object(self.knowledge.verifier, "verify", return_value={
            "results": [], "status": "insufficient", "message": "Incomplete evidence",
        }):
            result = self.knowledge.search("flood")
            self.assertEqual(result["results"], [])
            self.assertEqual(result["status"], "insufficient")
        with patch.object(self.knowledge.verifier, "verify", side_effect=RetrievalVerificationError("核验超时")):
            result = self.knowledge.search("flood")
            self.assertEqual(result["results"], [])
            self.assertEqual(result["status"], "unavailable")
            self.assertIn("核验超时", result["message"])

    def test_recall_pool_is_larger_than_final_result_limit(self):
        self.add_document("flood-1.docx", "Flood response: verify image dates.")
        self.add_document("flood-2.docx", "Flood response: verify coordinate reference systems.")
        with patch.object(self.knowledge.verifier, "verify", wraps=self.knowledge.verifier.verify) as verify:
            result = self.knowledge.search("flood", top_k=1)
        self.assertGreaterEqual(len(verify.call_args.args[1]), 2)
        self.assertEqual(len(result["results"]), 1)

    def test_full_question_not_keywords_controls_evidence_verification(self):
        self.add_document()
        original = "请读取测试文档中的 flood 报告核对要求，不是判断正式法规效力。"
        with patch.object(self.knowledge.verifier, "verify", wraps=self.knowledge.verifier.verify) as verify:
            self.knowledge.search("flood checklist", question=original)
        self.assertEqual(verify.call_args.args[0], original)

    def test_failure_atomicity_and_retry_after_restart(self):
        document = self.knowledge.upload("flood.docx", word_bytes(), {})["document"]
        with patch.object(self.embedder, "embed", side_effect=KnowledgeError("额度不足")):
            self.knowledge.process_one()
        self.assertEqual(self.knowledge.get_document(document["id"])["status"], "failed")
        self.assertEqual(self.knowledge.search("flood")["results"], [])
        self.knowledge.retry(document["id"])
        restarted = KnowledgeBase(Path(self.temporary.name), self.knowledge.settings,
                                  database=self.database, embedder=self.embedder)
        restarted.process_one()
        self.assertEqual(restarted.get_document(document["id"])["status"], "ready")

    def test_model_profile_change_requires_reindex(self):
        document_id = self.add_document()
        changed = KnowledgeBase(Path(self.temporary.name), EmbeddingSettings(
            "https://test.invalid/v1", "test", model="other-model", dimensions=2,
        ), database=self.database, embedder=self.embedder)
        self.assertTrue(changed.get_document(document_id)["needs_reindex"])
        self.assertEqual(changed.search("flood")["results"], [])

    def test_expired_lease_is_recovered_and_deleted_job_is_not_resurrected(self):
        document = self.knowledge.upload("flood.docx", word_bytes(), {})["document"]
        with self.database._conn() as conn:
            conn.execute("UPDATE knowledge_documents SET status='embedding',lease_until=now()-interval '1 minute' WHERE id=%s", (document["id"],))
        self.knowledge.process_one()
        self.assertEqual(self.knowledge.get_document(document["id"])["status"], "ready")
        self.knowledge.retry(document["id"])
        def delete_while_embedding(texts, **kwargs):
            self.knowledge.delete(document["id"])
            return [[1., 0.] for _ in texts]
        with patch.object(self.embedder, "embed", side_effect=delete_while_embedding):
            self.knowledge.process_one()
        self.assertEqual(self.knowledge.list_documents(), [])

    def test_scan_and_invalid_upload(self):
        with self.assertRaises(KnowledgeError):
            self.knowledge.upload("fake.pdf", b"not a pdf", {})
        with self.assertRaises(KnowledgeError):
            self.knowledge.upload("old.doc", b"old word", {})
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        output = io.BytesIO()
        writer.write(output)
        document = self.knowledge.upload("scan.pdf", output.getvalue(), {})["document"]
        self.knowledge.process_one()
        self.assertEqual(self.knowledge.get_document(document["id"])["status"], "needs_ocr")

    def test_fastapi_crud_preview_and_search(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from agent.knowledge_api import knowledge_router
        app = FastAPI()
        app.include_router(knowledge_router(self.knowledge))
        client = TestClient(app)
        response = client.post('/api/knowledge/documents', files={"file": ("flood.docx", word_bytes())})
        self.assertEqual(response.status_code, 202)
        document_id = response.json()["document"]["id"]
        self.knowledge.process_one()
        self.assertTrue(client.get(f'/api/knowledge/documents/{document_id}').json()["preview"])
        self.assertTrue(client.post('/api/knowledge/search', json={"query": "flood"}).json()["results"])
        self.assertEqual(client.get(f'/api/knowledge/documents/{document_id}/file').status_code, 200)
        self.assertEqual(client.patch(f'/api/knowledge/documents/{document_id}', json={"enabled": False}).status_code, 200)
        self.assertEqual(client.delete(f'/api/knowledge/documents/{document_id}').status_code, 200)
        self.assertEqual(client.get(f'/api/knowledge/documents/{document_id}').status_code, 404)
        self.assertEqual(client.get('/api/knowledge/documents/not-uuid').status_code, 422)

    def test_real_langgraph_calls_retrieval_and_preserves_citations(self):
        document_id = self.add_document()
        reference = self.knowledge.search("flood")["results"][0]
        llm = FakeMessagesListChatModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "search_rule_documents", "args": {"query": "flood"}, "id": "rules-call", "type": "tool_call"}]),
            AIMessage(content=f"<Conclusion>Protect hospitals {reference['citation']}</Conclusion>"),
        ])
        with patch.object(FakeMessagesListChatModel, "bind_tools", lambda model, tools, **kwargs: model, create=True):
            agent = create_react_agent(llm, [make_search_tool(self.knowledge)])
            with patch.object(self.knowledge.verifier, "verify", wraps=self.knowledge.verifier.verify) as verify:
                response = asyncio.run(agent.ainvoke({"messages": [HumanMessage(content="flood response rules?")]}, config={
                    "configurable": {"rag_question": "What does the uploaded document say about flood response?"},
                }))
                self.assertEqual(verify.call_args.args[0], "What does the uploaded document say about flood response?")
        refs = extract_rule_references(response, response["messages"][-1].content)
        self.assertEqual(refs[0]["document_id"], document_id)


if __name__ == '__main__':
    unittest.main()
