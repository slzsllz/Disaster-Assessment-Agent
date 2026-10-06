"""Opt-in PostgreSQL integration tests in a disposable, isolated schema.

Run with AUTH_TEST_DATABASE=1 against the configured development database.
Existing application tables and records are never changed.
"""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid

from dotenv import load_dotenv

from agent.db import Database, _build_dsn, psycopg, sql


@unittest.skipUnless(os.getenv("AUTH_TEST_DATABASE") == "1", "set AUTH_TEST_DATABASE=1 for PostgreSQL tests")
class AuthenticationDatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
        cls.schema = "auth_test_" + uuid.uuid4().hex
        cls.dsn = _build_dsn()
        with psycopg.connect(cls.dsn, connect_timeout=3) as conn:
            if not conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'postgis'").fetchone():
                raise unittest.SkipTest("Integration tests require a database with PostGIS already installed")
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(cls.schema)))
        cls.addClassCleanup(cls.drop_schema)
        cls.db = object.__new__(Database)
        cls.db._ensure_pool = lambda: True

        @contextmanager
        def isolated_connection():
            with psycopg.connect(cls.dsn, connect_timeout=3,
                                 options=f"-csearch_path={cls.schema},public") as conn:
                yield conn

        cls.db._conn = isolated_connection
        if not cls.db.migrate():
            raise RuntimeError("Isolated schema migration failed")

    @classmethod
    def drop_schema(cls):
        with psycopg.connect(cls.dsn, connect_timeout=3) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(cls.schema)))

    def setUp(self):
        self.user = self.db.create_user("user_" + uuid.uuid4().hex[:12], "测试", "hash-v1")

    def test_schema_migration_is_repeatable_and_usernames_unique(self):
        self.assertTrue(self.db.migrate())
        self.assertIsNone(self.db.create_user(self.user["username"], "duplicate", "hash-v2"))
        loaded = self.db.get_user_by_username(self.user["username"])
        self.assertEqual(loaded["id"], self.user["id"])

    def test_session_expiry_revocation_and_password_change(self):
        user_id = str(self.user["id"])
        expiry = datetime.now(timezone.utc) + timedelta(hours=1)
        token = uuid.uuid4().hex * 2
        self.assertTrue(self.db.create_auth_session(user_id, "hash-v1", token, expiry))
        self.assertEqual(str(self.db.get_authenticated_user(token)["id"]), user_id)
        self.assertFalse(self.db.change_user_password(user_id, "wrong-hash", "hash-v2"))
        self.assertTrue(self.db.change_user_password(user_id, "hash-v1", "hash-v2"))
        self.assertIsNone(self.db.get_authenticated_user(token))
        self.assertFalse(self.db.create_auth_session(user_id, "hash-v1", token, expiry))
        self.assertTrue(self.db.create_auth_session(user_id, "hash-v2", token, expiry))
        self.db.revoke_auth_session(token)
        self.assertIsNone(self.db.get_authenticated_user(token))
        self.assertTrue(self.db.create_auth_session(user_id, "hash-v2", token,
                                                  datetime.now(timezone.utc) - timedelta(seconds=1)))
        self.assertIsNone(self.db.get_authenticated_user(token))

    def test_session_ownership_and_lists_exclude_other_users_and_legacy_data(self):
        first, other, legacy = (str(uuid.uuid4()) for _ in range(3))
        second_user = self.db.create_user("other_" + uuid.uuid4().hex[:12], "其他用户", "hash")
        user_id = str(self.user["id"])
        other_id = str(second_user["id"])
        self.assertTrue(self.db.session_belongs_to_user(first, user_id, create=True))
        self.assertTrue(self.db.session_belongs_to_user(other, other_id, create=True))
        self.assertEqual(self.db.create_session(session_id=legacy), legacy)
        self.assertFalse(self.db.session_belongs_to_user(first, other_id, create=True))
        self.assertFalse(self.db.session_belongs_to_user(legacy, user_id, create=True))
        self.db.create_session(session_id=first, title="owned")
        self.assertTrue(self.db.session_belongs_to_user(first, user_id))
        self.assertEqual([str(row["id"]) for row in self.db.list_recent_sessions(user_id=user_id)], [first])
        with self.db._conn() as conn:
            for session_id in (first, other, legacy):
                conn.execute("INSERT INTO assessment_results(session_id, task) VALUES (%s, '洪水')", (session_id,))
        rows = self.db.query_assessments(user_id=user_id)
        self.assertEqual([str(row["session_id"]) for row in rows], [first])

    def test_rate_limit_is_durable_and_resets_after_expiry(self):
        key = "test:" + uuid.uuid4().hex
        self.assertTrue(self.db.consume_auth_attempt(key, 2, 900))
        self.assertTrue(self.db.consume_auth_attempt(key, 2, 900))
        self.assertFalse(self.db.consume_auth_attempt(key, 2, 900))
        with self.db._conn() as conn:
            conn.execute("UPDATE auth_rate_limits SET expires_at = now() - interval '1 second' WHERE key = %s", (key,))
        self.assertTrue(self.db.consume_auth_attempt(key, 2, 900))
        self.db.reset_auth_attempts(key)
        self.assertTrue(self.db.consume_auth_attempt(key, 1, 900))

    def test_http_account_lifecycle_with_real_database(self):
        from fastapi.testclient import TestClient
        import agent.auth as auth
        import backend_api as api

        username = "http_" + uuid.uuid4().hex[:12]
        password = "correct horse battery staple"
        new_password = "a different long password"
        with patch.object(auth, "db", self.db), patch.object(api, "db", self.db), TestClient(api.app) as client:
            registered = client.post("/api/auth/register", json={"username": username, "password": password})
            self.assertEqual(registered.status_code, 201, registered.text)
            user_id = registered.json()["user"]["id"]
            self.assertEqual(client.get("/api/auth/me").json()["user"]["id"], user_id)
            self.assertEqual(client.get("/api/sessions").json(), {"sessions": []})
            session_id = str(uuid.uuid4())
            self.assertTrue(self.db.session_belongs_to_user(session_id, user_id, create=True))
            self.assertEqual(client.get(f"/api/sessions/{session_id}/messages").status_code, 200)
            self.assertEqual(client.patch("/api/auth/me", json={"display_name": "新名字"}).json()["user"]["display_name"], "新名字")
            changed = client.post("/api/auth/password", json={"current_password": password, "new_password": new_password})
            self.assertEqual(changed.status_code, 200, changed.text)
            self.assertEqual(client.get("/api/auth/me").status_code, 401)
            login = client.post("/api/auth/login", json={"username": username, "password": new_password})
            self.assertEqual(login.status_code, 200, login.text)
            self.assertEqual(client.post("/api/auth/logout").status_code, 200)
            self.assertEqual(client.get("/api/auth/me").status_code, 401)


if __name__ == "__main__":
    unittest.main()
