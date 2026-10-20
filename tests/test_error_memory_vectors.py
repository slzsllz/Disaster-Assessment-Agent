import asyncio
import math
import os
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.tools import StructuredTool

from agent.db import Database, _build_dsn, psycopg, sql
from agent.error_memory import ErrorMemory
from agent.error_memory_vectors import ErrorVectorIndex
from agent.tool_recovery import recovery_run, wrap_tool


PATTERN = "detect_flood::CUDA out of memory"
QUERY = "GPU memory allocation failed because the device capacity is exhausted"
FIX = "Use the exposed device=cpu parameter without changing the model"


class Embedder:
    def __init__(self, vectors=None, profile="test-profile"):
        self.settings = SimpleNamespace(profile=profile, dimensions=3)
        self.vectors = vectors or {}
        self.calls = []
        self.fail = False

    def embed(self, texts, *, query=False):
        self.calls.append((list(texts), query))
        if self.fail:
            raise RuntimeError("Provider offline")
        return [self.vectors.get(text, [1, 0, 0]) for text in texts]


class VectorDatabase:
    def __init__(self):
        self.available = True
        self.rows = {}

    def ensure_error_vector_schema(self):
        return self.available

    def load_error_vectors(self, profile):
        if not self.available:
            return None
        return [dict(row) for (entry_id, p), row in self.rows.items() if p == profile]

    def save_error_vector(self, entry_id, profile, content_hash, embedding):
        if not self.available:
            return False
        self.rows[entry_id, profile] = {
            "entry_id": entry_id, "content_hash": content_hash, "embedding": list(embedding),
        }
        return True


class SemanticMemoryTests(unittest.TestCase):
    def setUp(self):
        self.embedder = Embedder()
        self.index = ErrorVectorIndex(self.embedder)
        self.memory = ErrorMemory(semantic_index=self.index)
        self.memory._memory = {PATTERN: FIX}

    def test_paraphrase_and_exact_match_contract(self):
        self.assertEqual(self.memory.lookup_all(QUERY, tool_name="detect_flood"), [(PATTERN, FIX)])
        self.assertEqual(len(self.embedder.calls), 2)
        self.embedder.calls.clear()
        self.assertEqual(self.memory.lookup_all("CUDA out of memory", tool_name="detect_flood"), [(PATTERN, FIX)])
        self.assertFalse(self.embedder.calls)

    def test_other_tool_and_unknown_tool_cannot_use_scoped_experience(self):
        for tool_name in (None, "detect_damage"):
            self.assertEqual(self.memory.lookup_all(QUERY, tool_name=tool_name), [])
        self.assertFalse(self.embedder.calls)

    def test_global_experience_is_available_across_tools(self):
        self.memory._memory = {"CUDA out of memory": FIX}
        self.assertTrue(self.memory.lookup_all(QUERY, tool_name="detect_damage"))

    def test_known_error_types_do_not_cross_match_even_with_identical_vectors(self):
        for query in (
            "PermissionError: output directory access denied",
            "TimeoutError: request timed out",
            "FileNotFoundError: file absent",
            "CUDA error: invalid device ordinal",
            "CUDA error: an illegal memory access was encountered",
        ):
            self.assertEqual(self.memory.lookup_all(query, tool_name="detect_flood"), [])
        self.assertFalse(self.embedder.calls)

    def test_dependency_and_attribute_names_must_agree(self):
        for pattern, query in (
            ("No module named 'torch'", "Import failed: No module named 'tensorflow'"),
            ("No module named 'torch'", "ImportError: required module tensorflow is unavailable"),
            ("has no attribute 'read'", "Array missing attribute 'crs'"),
        ):
            self.assertEqual(self.index.lookup(query, {pattern: "Repair with matching dependency"}), [])
        self.assertFalse(self.embedder.calls)

    def test_same_dependency_with_different_wording_is_eligible(self):
        pattern = "No module named 'torch'"
        result = self.index.lookup(
            "ImportError: required module torch is unavailable", {pattern: FIX},
        )
        self.assertEqual(result, [(pattern, FIX)])

    def test_low_similarity_returns_nothing(self):
        self.embedder.vectors[QUERY] = [0, 1, 0]
        self.assertEqual(self.memory.lookup_all(QUERY, tool_name="detect_flood"), [])

    def test_similar_scores_with_conflicting_repairs_are_rejected(self):
        entries = {PATTERN: FIX, "CUDA allocation failed": "Reduce the batch_size"}
        self.assertEqual(self.index.lookup(QUERY, entries, tool_name="detect_flood"), [])

    def test_identical_repairs_are_not_false_ambiguity(self):
        entries = {PATTERN: FIX, "CUDA allocation failed": FIX}
        self.assertEqual(len(self.index.lookup(QUERY, entries, tool_name="detect_flood")), 1)

    def test_scoped_experience_shadows_same_global_error_pattern(self):
        entries = {PATTERN: FIX, "CUDA out of memory": "A different generic repair"}
        self.assertEqual(self.index.lookup(QUERY, entries, tool_name="detect_flood"), [(PATTERN, FIX)])

    def test_empty_scoped_pattern_does_not_break_other_candidates(self):
        entries = {PATTERN: FIX, "detect_flood::": "Malformed legacy entry"}
        self.assertEqual(self.index.lookup(QUERY, entries, tool_name="detect_flood"), [(PATTERN, FIX)])

    def test_repeated_queries_use_cache(self):
        first = self.memory.lookup_all(QUERY, tool_name="detect_flood")
        self.assertTrue(first)
        count = len(self.embedder.calls)
        self.assertEqual(self.memory.lookup_all(QUERY, tool_name="detect_flood"), first)
        self.assertEqual(len(self.embedder.calls), count)

    def test_parallel_duplicate_queries_are_single_flight(self):
        with ThreadPoolExecutor(max_workers=6) as executor:
            results = list(executor.map(
                lambda _: self.memory.lookup_all(QUERY, tool_name="detect_flood"), range(12),
            ))
        self.assertTrue(all(result == [(PATTERN, FIX)] for result in results))
        self.assertEqual(len(self.embedder.calls), 2)

    def test_query_cache_is_bounded(self):
        for number in range(260):
            self.index.lookup(f"GPU memory allocation exhausted at stage {number}", {PATTERN: FIX}, tool_name="detect_flood")
        self.assertEqual(len(self.index._queries), 256)

    def test_backfill_respects_provider_batch_limit(self):
        entries = {f"GPU memory allocation failed at stage {number}": FIX for number in range(25)}
        self.assertTrue(self.index.index_entries(entries))
        self.assertEqual([len(texts) for texts, _ in self.embedder.calls], [10, 10, 5])

    def test_embedding_outage_has_backoff_and_preserves_exact_lookup(self):
        self.embedder.fail = True
        for _ in range(3):
            self.assertEqual(self.memory.lookup_all(QUERY, tool_name="detect_flood"), [])
        self.assertEqual(len(self.embedder.calls), 1)
        self.assertTrue(self.memory.lookup_all("CUDA out of memory", tool_name="detect_flood"))
        self.embedder.fail = False
        self.index._next_embedding_try = 0
        self.assertTrue(self.memory.lookup_all(QUERY, tool_name="detect_flood"))

    def test_query_embedding_failure_is_not_replaced_by_highest_candidate(self):
        self.index.index_entries({PATTERN: FIX})
        self.embedder.fail = True
        self.assertEqual(self.index.lookup(QUERY, {PATTERN: FIX}, tool_name="detect_flood"), [])

    def test_invalid_vectors_fail_closed(self):
        for vector in ([0, 0, 0], [1, 2], [math.nan, 0, 0], [1e308, 0, 0]):
            with self.subTest(vector=vector):
                index = ErrorVectorIndex(Embedder({"CUDA out of memory": vector}))
                self.assertEqual(index.lookup(QUERY, {PATTERN: FIX}, tool_name="detect_flood"), [])

    def test_no_paths_credentials_or_stack_prefix_sent_to_embedding(self):
        query = "stack frame " * 500 + (
            "GPU memory allocation exhausted /home/private/scene.tif "
            "api_key=private-secret Bearer private-token "
            'password="first secret second" token=\'quoted\\\' secret tail\' '
            + "AAP" + "sensitive" * 12
        )
        self.memory.lookup_all(query, tool_name="detect_flood")
        payload = str(self.embedder.calls)
        for secret in ("/home/private", "private-secret", "private-token", "sensitive", "first", "second", "quoted", "tail"):
            self.assertNotIn(secret, payload)
        self.assertTrue(all(len(text) <= 2000 for texts, _ in self.embedder.calls for text in texts))

    def test_record_indexes_new_fix_and_never_resurrects_removed_entries(self):
        self.memory.record(PATTERN, FIX)
        self.assertEqual(len(self.embedder.calls), 1)
        self.memory.record(PATTERN, "Reduce the supported batch_size only")
        self.assertEqual(len(self.embedder.calls), 2)
        self.assertEqual(self.memory.lookup_all(QUERY, tool_name="detect_flood")[0][1], "Reduce the supported batch_size only")
        self.memory._memory.clear()
        self.assertEqual(self.memory.lookup_all(QUERY, tool_name="detect_flood"), [])

    def test_stale_fix_changed_during_vector_lookup_is_not_returned(self):
        def change(*args, **kwargs):
            self.memory._memory[PATTERN] = "A different current repair"
            return [(PATTERN, FIX)]
        with patch.object(self.index, "lookup", side_effect=change):
            self.assertEqual(self.memory.lookup_all(QUERY, tool_name="detect_flood"), [])

    def test_bad_thresholds_are_rejected(self):
        for threshold in (-0.1, 1.1, math.nan):
            with self.assertRaises(ValueError):
                ErrorVectorIndex(self.embedder, min_score=threshold)

    def test_tool_failure_contains_semantic_experience(self):
        async def detect(image_path: str):
            raise RuntimeError(QUERY)

        tool = wrap_tool(StructuredTool.from_function(
            coroutine=detect, name="detect_flood", description="Test flood tool",
        ), self.memory)

        async def run():
            with recovery_run():
                result = await tool.ainvoke({
                    "type": "tool_call", "name": "detect_flood", "id": "test-call",
                    "args": {"image_path": "explicit-upload.tif"},
                })
            self.assertIn(FIX, result.content)
            self.assertIn("remaining_attempts", result.content)

        asyncio.run(run())

    def test_execution_failure_embeds_off_the_event_loop(self):
        async def detect(image_path: str):
            raise RuntimeError(QUERY)

        tool = wrap_tool(StructuredTool.from_function(
            coroutine=detect, name="detect_flood", description="Test flood tool",
        ), self.memory)
        threads = []
        original_embed = self.embedder.embed

        def embed(*args, **kwargs):
            threads.append(threading.get_ident())
            return original_embed(*args, **kwargs)

        async def run():
            loop_thread = threading.get_ident()
            with patch.object(self.embedder, "embed", side_effect=embed), recovery_run():
                result = await tool.ainvoke({
                    "type": "tool_call", "name": tool.name, "id": "worker-test",
                    "args": {"image_path": "explicit-upload.tif"},
                })
            self.assertIn(FIX, result.content)
            self.assertEqual(len(threads), 2)
            self.assertTrue(all(thread != loop_thread for thread in threads))

        asyncio.run(run())


class PersistenceTests(unittest.TestCase):
    def test_restart_reuses_vectors_but_profile_change_reindexes(self):
        database = VectorDatabase()
        ErrorVectorIndex(Embedder(), database=database).index_entries({PATTERN: FIX})
        restored_embedder = Embedder()
        restored = ErrorVectorIndex(restored_embedder, database=database)
        self.assertTrue(restored.lookup(QUERY, {PATTERN: FIX}, tool_name="detect_flood"))
        self.assertEqual(restored_embedder.calls, [([QUERY], True)])
        changed = Embedder(profile="changed-model-or-dimensions")
        self.assertTrue(ErrorVectorIndex(changed, database=database).lookup(QUERY, {PATTERN: FIX}, tool_name="detect_flood"))
        self.assertEqual(len(changed.calls), 2)
        self.assertEqual(len(database.rows), 2)

    def test_database_outage_retains_vectors_and_retries_pending_writes(self):
        database = VectorDatabase()
        database.available = False
        embedder = Embedder()
        index = ErrorVectorIndex(embedder, database=database)
        self.assertTrue(index.lookup(QUERY, {PATTERN: FIX}, tool_name="detect_flood"))
        self.assertTrue(index._pending)
        database.available = True
        index._next_db_try = 0
        self.assertTrue(index.lookup(QUERY, {PATTERN: FIX}, tool_name="detect_flood"))
        self.assertFalse(index._pending)
        self.assertEqual(len(database.rows), 1)
        self.assertEqual(len(embedder.calls), 2)

    def test_wrong_dimension_persisted_vector_is_rebuilt(self):
        database = VectorDatabase()
        ErrorVectorIndex(Embedder(), database=database).index_entries({PATTERN: FIX})
        next(iter(database.rows.values()))["embedding"] = [1, 0]
        embedder = Embedder()
        self.assertTrue(ErrorVectorIndex(embedder, database=database).lookup(QUERY, {PATTERN: FIX}, tool_name="detect_flood"))
        self.assertEqual(len(embedder.calls), 2)


@unittest.skipUnless(os.getenv("ERROR_MEMORY_TEST_DB") == "1", "Enable isolated PostgreSQL integration test")
class DatabaseIntegrationTests(unittest.TestCase):
    def test_real_schema_roundtrip_and_restart(self):
        from dotenv import load_dotenv
        load_dotenv(".env", override=True)
        schema = "test_error_vectors_" + uuid.uuid4().hex
        dsn = _build_dsn()
        with psycopg.connect(dsn, connect_timeout=3) as conn:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))

        class TestDatabase:
            ensure_error_vector_schema = Database.ensure_error_vector_schema
            load_error_vectors = Database.load_error_vectors
            save_error_vector = Database.save_error_vector

            def health_check(self):
                return True

            def _ensure_pool(self):
                return True

            @contextmanager
            def _conn(self):
                with psycopg.connect(dsn, connect_timeout=3) as conn:
                    conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
                    yield conn

        try:
            database = TestDatabase()
            first = ErrorVectorIndex(Embedder(), database=database)
            self.assertTrue(first.index_entries({PATTERN: FIX}))
            self.assertFalse(first._pending)
            self.assertTrue(database.ensure_error_vector_schema())
            embedder = Embedder()
            restored = ErrorVectorIndex(embedder, database=database)
            self.assertTrue(restored.lookup(QUERY, {PATTERN: FIX}, tool_name="detect_flood"))
            self.assertEqual(embedder.calls, [([QUERY], True)])
            restored.index_entries({PATTERN: "Updated supported execution repair"})
            self.assertEqual(len(database.load_error_vectors("test-profile")), 1)
        finally:
            with psycopg.connect(dsn, connect_timeout=3) as conn:
                conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


if __name__ == "__main__":
    unittest.main()
