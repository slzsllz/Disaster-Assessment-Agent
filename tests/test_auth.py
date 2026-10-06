"""HTTP regression tests for authentication, revocation and access control."""

from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

from fastapi.testclient import TestClient

import agent.auth as auth
import backend_api as api
from agent.db import DatabaseUnavailable

PASSWORD = "correct horse battery staple"
NEW_PASSWORD = "another long password for testing"


class MemoryDatabase:
    """Stateful storage substitute; hashing, cookies and HTTP routing are real."""

    def __init__(self):
        self.users = {}
        self.tokens = {}
        self.sessions = {}
        self.attempts = {}
        self.artifacts = {}
        self.messages = {}
        self.available = True

    def require_available(self):
        if not self.available:
            raise DatabaseUnavailable("offline")

    def migrate(self):
        return True

    def reconcile_stale_turns(self):
        return 0

    def create_user(self, username, display_name, password_hash):
        self.require_available()
        if username in self.users:
            return None
        user = dict(id=str(uuid.uuid4()), username=username, display_name=display_name,
                    password_hash=password_hash, is_active=True,
                    created_at=datetime.now(timezone.utc))
        self.users[username] = user
        return user.copy()

    def get_user_by_username(self, username):
        self.require_available()
        user = self.users.get(username)
        return user.copy() if user else None

    def create_auth_session(self, user_id, password_hash, token_hash, expires_at):
        user = next(user for user in self.users.values() if user["id"] == user_id)
        if user["password_hash"] != password_hash or not user["is_active"]:
            return False
        self.tokens[token_hash] = (user_id, expires_at)
        return True

    def get_authenticated_user(self, token_hash):
        self.require_available()
        session = self.tokens.get(token_hash)
        if not session or session[1] <= datetime.now(timezone.utc):
            return None
        user = next(user for user in self.users.values() if user["id"] == session[0])
        return user.copy() if user["is_active"] else None

    def revoke_auth_session(self, token_hash):
        self.require_available()
        self.tokens.pop(token_hash, None)

    def update_user_profile(self, user_id, display_name):
        user = next(user for user in self.users.values() if user["id"] == user_id)
        user["display_name"] = display_name
        return user.copy()

    def change_user_password(self, user_id, old_hash, new_hash):
        user = next(user for user in self.users.values() if user["id"] == user_id)
        if user["password_hash"] != old_hash:
            return False
        user["password_hash"] = new_hash
        self.tokens = {key: value for key, value in self.tokens.items() if value[0] != user_id}
        return True

    def consume_auth_attempt(self, key, limit, window_seconds):
        self.require_available()
        self.attempts[key] = self.attempts.get(key, 0) + 1
        return self.attempts[key] <= limit

    def reset_auth_attempts(self, key):
        self.attempts.pop(key, None)

    def session_belongs_to_user(self, session_id, user_id, create=False):
        self.require_available()
        if create:
            self.sessions.setdefault(session_id, user_id)
        return self.sessions.get(session_id) == user_id

    def list_recent_sessions(self, limit=30, user_id=None):
        return [dict(id=key, title="test", message_count=0)
                for key, owner in self.sessions.items() if owner == user_id][:limit]

    def query_assessments(self, task=None, limit=50, user_id=None, session_id=None):
        return [dict(id=1, session_id=key, task="洪水", summary={})
                for key, owner in self.sessions.items()
                if owner == user_id or (session_id is not None and key == session_id)]

    def load_chat_messages_strict(self, session_id):
        return self.messages.get(session_id, [])

    def get_artifact(self, artifact_id):
        return self.artifacts.get(artifact_id)


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.db = MemoryDatabase()
        self.stack.enter_context(patch.object(api, "db", self.db))
        self.stack.enter_context(patch.object(auth, "db", self.db))
        self.stack.enter_context(patch.dict("os.environ", {"AUTH_COOKIE_SECURE": "false"}))
        self.client = TestClient(api.app)
        self.addCleanup(self.client.close)

    def register(self, username="alice", client=None):
        response = (client or self.client).post("/api/auth/register", json={
            "username": username, "display_name": "测试用户", "password": PASSWORD,
        })
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["user"]

    def test_register_normalizes_username_and_sets_private_cookie(self):
        user = self.register(" Alice ")
        self.assertEqual(user["username"], "alice")
        self.assertNotIn("password_hash", user)
        self.assertNotIn(PASSWORD, self.db.users["alice"]["password_hash"])
        token = self.client.cookies.get(auth.COOKIE_NAME)
        self.assertIn(auth.token_hash(token), self.db.tokens)
        self.assertNotIn(token, self.db.tokens)
        self.assertEqual(self.client.get("/api/auth/me").json()["user"], user)
        duplicate = self.client.post("/api/auth/register", json={"username": "ALICE", "password": PASSWORD})
        self.assertEqual(duplicate.status_code, 409)

    def test_cookie_attributes_and_https_secure_cookie(self):
        with TestClient(api.app, base_url="https://testserver") as client:
            response = client.post("/api/auth/register", json={"username": "alice", "password": PASSWORD})
            cookie = response.headers["set-cookie"].lower()
            for attribute in ("httponly", "samesite=lax", "secure", "path=/api", "max-age=604800"):
                self.assertIn(attribute, cookie)
            self.assertEqual(response.headers["cache-control"], "no-store")

    def test_login_rotates_cookie_logout_revokes_it(self):
        self.register()
        old_token = self.client.cookies.get(auth.COOKIE_NAME)
        bad = self.client.post("/api/auth/login", json={"username": "alice", "password": "wrong"})
        missing = self.client.post("/api/auth/login", json={"username": "nobody", "password": "wrong"})
        self.assertEqual((bad.status_code, bad.json()), (missing.status_code, missing.json()))
        response = self.client.post("/api/auth/login", json={"username": "ALICE", "password": PASSWORD})
        self.assertEqual(response.status_code, 200)
        new_token = self.client.cookies.get(auth.COOKIE_NAME)
        self.assertNotEqual(old_token, new_token)
        self.assertNotIn(auth.token_hash(old_token), self.db.tokens)
        self.assertEqual(self.client.post("/api/auth/logout").status_code, 200)
        self.assertNotIn(auth.token_hash(new_token), self.db.tokens)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

    def test_profile_and_password_change_revokes_every_device(self):
        user = self.register()
        other = TestClient(api.app)
        self.addCleanup(other.close)
        other.post("/api/auth/login", json={"username": "alice", "password": PASSWORD})
        updated = self.client.patch("/api/auth/me", json={"display_name": " 新昵称 "})
        self.assertEqual(updated.json()["user"]["display_name"], "新昵称")
        invalid = self.client.post("/api/auth/password", json={"current_password": "wrong", "new_password": NEW_PASSWORD})
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(other.get("/api/auth/me").status_code, 200)
        response = self.client.post("/api/auth/password", json={"current_password": PASSWORD, "new_password": NEW_PASSWORD})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)
        self.assertEqual(other.get("/api/auth/me").status_code, 401)
        self.assertEqual(other.post("/api/auth/login", json={"username": "alice", "password": PASSWORD}).status_code, 401)
        login = other.post("/api/auth/login", json={"username": "alice", "password": NEW_PASSWORD})
        self.assertEqual(login.json()["user"]["id"], user["id"])

    def test_unauthenticated_business_routes_require_login(self):
        session = str(uuid.uuid4())
        routes = [
            ("GET", "/api/sessions"), ("GET", "/api/assessments"),
            ("GET", f"/api/files/{uuid.uuid4()}"),
            *[("GET", f"/api/sessions/{session}/{suffix}")
              for suffix in ("messages", "turns", "assessments", "latest-geometry")],
            ("POST", f"/api/sessions/{session}/clear"), ("DELETE", f"/api/sessions/{session}"),
        ]
        for method, path in routes:
            with self.subTest(path=path):
                self.assertEqual(self.client.request(method, path).status_code, 401)
        for path in ("/api/chat", "/api/chat/stream"):
            self.assertEqual(self.client.post(path, data={"session_id": session, "message": "hi"}).status_code, 401)

    def test_users_cannot_read_modify_or_claim_others_sessions(self):
        alice = self.register()
        alice_session = str(uuid.uuid4())
        anonymous_session = str(uuid.uuid4())
        self.db.sessions[alice_session] = alice["id"]
        self.db.sessions[anonymous_session] = None
        bob = TestClient(api.app)
        self.addCleanup(bob.close)
        user = self.register("bob", bob)
        bob_session = str(uuid.uuid4())
        self.db.sessions[bob_session] = user["id"]
        self.assertEqual([s["id"] for s in bob.get("/api/sessions").json()["sessions"]], [bob_session])
        self.assertEqual([s["session_id"] for s in bob.get("/api/assessments").json()["assessments"]], [bob_session])
        for session in (alice_session, anonymous_session):
            for suffix in ("messages", "turns", "assessments", "latest-geometry"):
                self.assertEqual(bob.get(f"/api/sessions/{session}/{suffix}").status_code, 404)
            self.assertEqual(bob.post(f"/api/sessions/{session}/clear").status_code, 404)
            self.assertEqual(bob.delete(f"/api/sessions/{session}").status_code, 404)
            for path in ("/api/chat", "/api/chat/stream"):
                self.assertEqual(bob.post(path, data={"session_id": session, "message": "hi"}).status_code, 404)
        self.assertEqual(self.db.sessions[alice_session], alice["id"])
        self.assertIsNone(self.db.sessions[anonymous_session])

    def test_new_chat_session_is_owned_before_agent_starts(self):
        user = self.register()
        session_id = str(uuid.uuid4())
        for path in ("/api/chat", "/api/chat/stream"):
            with patch.object(api, "get_session", side_effect=api.HTTPException(503, "test runtime unavailable")):
                response = self.client.post(path, data={"session_id": session_id, "message": "hi"})
                self.assertEqual(response.status_code, 503)
                self.assertEqual(self.db.sessions[session_id], user["id"])

    def test_artifact_and_legacy_file_downloads_check_owner(self):
        alice = self.register()
        session_id = str(uuid.uuid4())
        self.db.sessions[session_id] = alice["id"]
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "report.pdf"
            file.write_bytes(b"test report")
            legacy = api.file_payload(str(file), session_id)["url"]
            artifact_id = str(uuid.uuid4())
            self.db.artifacts[artifact_id] = dict(session_id=session_id, mime_type="application/pdf", original_name="report.pdf")
            bob = TestClient(api.app)
            self.addCleanup(bob.close)
            self.register("bob", bob)
            with patch.object(api.ARTIFACT_STORE, "resolve", return_value=file):
                for url in (legacy, f"/api/files/{artifact_id}"):
                    self.assertEqual(bob.get(url).status_code, 404)
                    response = self.client.get(url)
                    self.assertEqual(response.content, b"test report")
                    self.assertEqual(response.headers["cache-control"], "private, no-store")

    def test_cross_origin_writes_are_rejected_and_same_origin_allowed(self):
        payload = {"username": "alice", "password": PASSWORD}
        bad = self.client.post("/api/auth/register", json=payload, headers={"Origin": "https://evil.example"})
        self.assertEqual(bad.status_code, 403)
        self.assertEqual(self.db.users, {})
        good = self.client.post("/api/auth/register", json=payload, headers={"Origin": "http://testserver"})
        self.assertEqual(good.status_code, 201)
        self.assertEqual(self.client.post("/api/auth/logout", headers={"Origin": "https://evil.example"}).status_code, 403)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 200)
        self.assertEqual(self.client.post("/api/auth/logout", headers={"sec-fetch-site": "cross-site"}).status_code, 403)

    def test_expired_and_disabled_sessions_cannot_authenticate(self):
        self.register()
        key = auth.token_hash(self.client.cookies.get(auth.COOKIE_NAME))
        user_id, _ = self.db.tokens[key]
        self.db.tokens[key] = (user_id, datetime.now(timezone.utc) - timedelta(seconds=1))
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)
        self.client.post("/api/auth/login", json={"username": "alice", "password": PASSWORD})
        self.db.users["alice"]["is_active"] = False
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

    def test_rate_limits_and_database_failures(self):
        self.register()
        for _ in range(10):
            response = self.client.post("/api/auth/login", json={"username": "alice", "password": "wrong"})
            self.assertEqual(response.status_code, 401)
        limited = self.client.post("/api/auth/login", json={"username": "alice", "password": "wrong"})
        self.assertEqual(limited.status_code, 429)
        self.assertIn("retry-after", limited.headers)
        self.db.available = False
        self.assertEqual(self.client.get("/api/auth/me").status_code, 503)
        self.assertEqual(self.client.post("/api/auth/register", json={"username": "bob", "password": PASSWORD}).status_code, 503)

    def test_password_and_username_validation(self):
        malformed = self.client.post("/api/auth/register", json={"password": PASSWORD})
        self.assertEqual(malformed.status_code, 422)
        self.assertNotIn(PASSWORD, malformed.text)
        for payload in ({"username": "ab", "password": PASSWORD},
                        {"username": "alice", "password": "short"},
                        {"username": "alice", "password": "a" * 129}):
            self.assertEqual(self.client.post("/api/auth/register", json=payload).status_code, 400)
        self.assertEqual(self.db.users, {})
        first = auth.hash_password(PASSWORD)
        second = auth.hash_password(PASSWORD)
        self.assertNotEqual(first, second)
        self.assertTrue(auth.verify_password(PASSWORD, first))
        self.assertFalse(auth.verify_password("wrong", first))
        self.assertFalse(auth.verify_password(PASSWORD, "corrupt"))


if __name__ == "__main__":
    unittest.main()
