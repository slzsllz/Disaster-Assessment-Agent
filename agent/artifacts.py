"""Durable local artifact storage with database metadata."""

from __future__ import annotations

import hashlib
import mimetypes
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from agent.db import Database


class ArtifactStore:
    def __init__(self, root: Path, db: Database):
        self.root = root.resolve()
        self.db = db
        self.root.mkdir(parents=True, exist_ok=True)

    def put_file(self, source: str | Path, session_id: str, kind: str, name: str | None = None) -> dict:
        source = Path(source).resolve(strict=True)
        if not source.is_file():
            raise ValueError("Artifact source must be a file")
        artifact_id = str(uuid.uuid4())
        safe_name = Path((name or source.name).replace("\\", "/")).name[:200] or "artifact"
        session_dir = self.root / str(uuid.UUID(session_id))
        session_dir.mkdir(parents=True, exist_ok=True)
        destination = session_dir / artifact_id
        digest = hashlib.sha256()
        size = 0
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=session_dir, delete=False) as output:
                temporary = Path(output.name)
                with source.open("rb") as input_file:
                    while chunk := input_file.read(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
            record = {
                "id": artifact_id,
                "session_id": session_id,
                "kind": kind,
                "original_name": safe_name,
                "mime_type": mimetypes.guess_type(safe_name)[0] or "application/octet-stream",
                "size_bytes": size,
                "sha256": digest.hexdigest(),
                "relative_path": str(destination.relative_to(self.root)),
            }
            self.db.save_artifact(record)
            return record
        except BaseException:
            destination.unlink(missing_ok=True)
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise

    def put_bytes(self, data: bytes, session_id: str, kind: str, name: str) -> dict:
        with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
        try:
            return self.put_file(temporary, session_id, kind, name)
        finally:
            temporary.unlink(missing_ok=True)

    def resolve(self, record: dict) -> Path:
        path = (self.root / record["relative_path"]).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise FileNotFoundError("Artifact content is unavailable")
        return path

    def delete_session_files(self, session_id: str) -> None:
        shutil.rmtree(self.root / str(uuid.UUID(session_id)), ignore_errors=True)


def artifact_payload(record: dict) -> dict[str, str]:
    return {"id": str(record["id"]), "name": record["original_name"],
            "url": f"/api/files/{record['id']}"}
