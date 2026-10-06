"""Password authentication with revocable, database-backed HttpOnly sessions."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from agent.db import db

COOKIE_NAME = "disaster_auth"
PASSWORD_MIN_LENGTH = 12
PASSWORD_MAX_LENGTH = 128
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**15, 8, 3
USERNAME_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{2,31}\Z")
TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{43}\Z")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P,
        maxmem=64 * 1024 * 1024, dklen=32,
    )
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    if len(password) > PASSWORD_MAX_LENGTH:
        return False
    try:
        algorithm, n, r, p, salt, expected = stored.split("$")
        # Only accept our supported cost, preventing corrupted hashes from
        # requesting unbounded memory or CPU.
        if algorithm != "scrypt" or (int(n), int(r), int(p)) != (SCRYPT_N, SCRYPT_R, SCRYPT_P):
            return False
        salt_bytes, expected_bytes = bytes.fromhex(salt), bytes.fromhex(expected)
        if len(salt_bytes) != 16 or len(expected_bytes) != 32:
            return False
        actual = hashlib.scrypt(
            password.encode("utf-8"), salt=salt_bytes, n=int(n), r=int(r), p=int(p),
            maxmem=64 * 1024 * 1024, dklen=32,
        )
        return hmac.compare_digest(actual, expected_bytes)
    except (ValueError, TypeError, AttributeError):
        return False


DUMMY_PASSWORD_HASH = hash_password(secrets.token_urlsafe(32))


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def allowed_origins() -> list[str]:
    defaults = "http://localhost:5173,http://127.0.0.1:5173"
    return [value.strip().rstrip("/") for value in os.getenv("AUTH_ALLOWED_ORIGINS", defaults).split(",")
            if value.strip()]


def check_request_origin(request: Request) -> None:
    """Reject cross-origin browser writes, including login CSRF.

    CLI clients without an Origin remain supported. Browsers supply Origin on
    writes; Sec-Fetch-Site also rejects cross-site requests without it.
    """
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return
    origin = request.headers.get("origin")
    same_origin = f"{request.url.scheme}://{request.url.netloc}"
    if origin:
        if origin.rstrip("/") not in {same_origin, *allowed_origins()}:
            raise HTTPException(403, "请求来源不受信任。")
    elif request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(403, "请求来源不受信任。")


def public_user(user: dict) -> dict:
    return {"id": str(user["id"]), "username": user["username"],
            "display_name": user["display_name"], "created_at": user["created_at"]}


def require_user(request: Request, response: Response) -> dict:
    response.headers["Cache-Control"] = "private, no-store"
    token = request.cookies.get(COOKIE_NAME, "")
    user = db.get_authenticated_user(token_hash(token)) if TOKEN_RE.fullmatch(token) else None
    if user is None:
        raise HTTPException(401, "请先登录，或登录已过期。", headers={"Cache-Control": "no-store"})
    return user


def require_session_owner(session_id: str, user: dict, *, create: bool = False) -> str:
    try:
        session_id = str(uuid.UUID(session_id))
    except (ValueError, AttributeError):
        raise HTTPException(400, "会话 ID 无效。") from None
    if not db.session_belongs_to_user(session_id, str(user["id"]), create=create):
        # Do not reveal whether another person's session exists.
        raise HTTPException(404, "会话不存在。")
    return session_id


def _secure_cookie(request: Request) -> bool:
    return request.url.scheme == "https" or os.getenv("AUTH_COOKIE_SECURE", "false").lower() == "true"


def _clear_cookie(request: Request, response: Response) -> None:
    response.delete_cookie(COOKIE_NAME, path="/api", httponly=True,
                           secure=_secure_cookie(request), samesite="lax")
    response.headers["Cache-Control"] = "no-store"


def _issue_session(user: dict, request: Request, response: Response) -> None:
    token = secrets.token_urlsafe(32)
    hours = max(1, min(int(os.getenv("AUTH_SESSION_HOURS", "168")), 720))
    expires_at = datetime.now(timezone.utc) + timedelta(hours=hours)
    if not db.create_auth_session(str(user["id"]), user["password_hash"], token_hash(token), expires_at):
        raise HTTPException(409, "账号状态已变更，请重新登录。")
    previous = request.cookies.get(COOKIE_NAME, "")
    if TOKEN_RE.fullmatch(previous):
        db.revoke_auth_session(token_hash(previous))
    response.set_cookie(COOKIE_NAME, token, max_age=hours * 3600, expires=expires_at,
                        path="/api", httponly=True, secure=_secure_cookie(request), samesite="lax")
    response.headers["Cache-Control"] = "no-store"


def _rate_key(scope: str, value: str) -> str:
    return f"{scope}:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _limit(scope: str, value: str, limit: int, seconds: int = 900) -> str:
    key = _rate_key(scope, value)
    if not db.consume_auth_attempt(key, limit, seconds):
        raise HTTPException(429, "尝试过于频繁，请稍后重试。", headers={"Retry-After": str(seconds)})
    return key


def _client_ip(request: Request) -> str:
    # Trust forwarded headers only when uvicorn is explicitly configured to.
    return request.client.host if request.client else "unknown"


def _username(value: str) -> str:
    normalized = value.strip().lower()
    if not USERNAME_RE.fullmatch(normalized):
        raise HTTPException(400, "用户名需为 3–32 位字母、数字、下划线、点或短横线，以字母或数字开头。")
    return normalized


def _display_name(value: str) -> str:
    value = value.strip()
    if not 1 <= len(value) <= 50 or any(ord(c) < 32 for c in value):
        raise HTTPException(400, "昵称需为 1–50 个字符。")
    return value


def _new_password(value: SecretStr) -> str:
    password = value.get_secret_value()
    if not PASSWORD_MIN_LENGTH <= len(password) <= PASSWORD_MAX_LENGTH:
        raise HTTPException(400, "密码需为 12–128 个字符。")
    return password


class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(max_length=100)
    password: SecretStr


class Registration(Credentials):
    display_name: str = Field(default="", max_length=50)


class ProfileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_name: str = Field(max_length=50)


class PasswordChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current_password: SecretStr
    new_password: SecretStr


router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", status_code=201)
def register(payload: Registration, request: Request, response: Response) -> dict:
    _limit("register-ip", _client_ip(request), 10, 3600)
    username = _username(payload.username)
    password = _new_password(payload.password)
    display_name = _display_name(payload.display_name or username)
    user = db.create_user(username, display_name, hash_password(password))
    if user is None:
        raise HTTPException(409, "该用户名已被注册。")
    _issue_session(user, request, response)
    return {"user": public_user(user)}


@router.post("/login")
def login(payload: Credentials, request: Request, response: Response) -> dict:
    _limit("login-ip", _client_ip(request), 60)
    username = _username(payload.username)
    account_key = _limit("login-account", username, 10)
    user = db.get_user_by_username(username)
    valid = verify_password(payload.password.get_secret_value(),
                            user["password_hash"] if user else DUMMY_PASSWORD_HASH)
    if not valid or not user or not user["is_active"]:
        raise HTTPException(401, "用户名或密码错误。")
    _issue_session(user, request, response)
    db.reset_auth_attempts(account_key)
    return {"user": public_user(user)}


@router.get("/me")
def me(user: dict = Depends(require_user)) -> dict:
    return {"user": public_user(user)}


@router.patch("/me")
def update_profile(payload: ProfileUpdate, user: dict = Depends(require_user)) -> dict:
    updated = db.update_user_profile(str(user["id"]), _display_name(payload.display_name))
    return {"user": public_user(updated)}


@router.post("/password")
def change_password(payload: PasswordChange, request: Request, response: Response,
                    user: dict = Depends(require_user)) -> dict:
    _limit("password-user", str(user["id"]), 10)
    password = _new_password(payload.new_password)
    if not verify_password(payload.current_password.get_secret_value(), user["password_hash"]):
        raise HTTPException(400, "当前密码不正确。")
    if hmac.compare_digest(password.encode("utf-8"), payload.current_password.get_secret_value().encode("utf-8")):
        raise HTTPException(400, "新密码应与当前密码不同。")
    if not db.change_user_password(str(user["id"]), user["password_hash"], hash_password(password)):
        raise HTTPException(409, "密码已变更，请重新登录。")
    _clear_cookie(request, response)
    return {"ok": True, "message": "密码已修改，请重新登录。"}


@router.post("/logout")
def logout(request: Request, response: Response) -> dict:
    token = request.cookies.get(COOKIE_NAME, "")
    if TOKEN_RE.fullmatch(token):
        db.revoke_auth_session(token_hash(token))
    _clear_cookie(request, response)
    return {"ok": True}
