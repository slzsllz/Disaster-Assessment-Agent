"""Move historical inline chat files into the durable artifact store.

Run without arguments to preview, then pass --apply to migrate. Each message is
updated only after every available inline file has been copied and verified.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from agent.artifacts import ArtifactStore  # noqa: E402
from agent.db import db  # noqa: E402


def migrate_rows(apply: bool) -> tuple[int, int, int]:
    if apply and not db.migrate():
        raise RuntimeError("Database migration failed")
    db.require_available()
    store = ArtifactStore(PROJECT_ROOT / "data" / "artifacts", db)
    legacy_root = (PROJECT_ROOT / "tmp" / "fastapi_out").resolve()
    last_id = 0
    candidates = converted = missing = 0

    def copy_item(item: dict, session_id: str, kind: str) -> dict:
        nonlocal missing
        if item.get("artifact_id"):
            return item
        if item.get("unavailable"):
            return item
        name = str(item.get("name") or "artifact")
        encoded = item.get("data_base64")
        if encoded:
            data = base64.b64decode(encoded, validate=True)
            record = store.put_bytes(data, session_id, kind, name)
            if hashlib.sha256(store.resolve(record).read_bytes()).hexdigest() != record["sha256"]:
                raise RuntimeError("Artifact checksum verification failed")
        else:
            path_text = item.get("path")
            if not path_text:
                missing += 1
                return {"name": name, "unavailable": True}
            path = Path(path_text).resolve()
            if not path.is_relative_to(legacy_root) or not path.is_file():
                missing += 1
                return {"name": name, "unavailable": True}
            record = store.put_file(path, session_id, kind, name)
        result = {"name": record["original_name"], "artifact_id": record["id"]}
        if item.get("description"):
            result["description"] = item["description"]
        return result

    def convert(descriptors: list, inline: list, session_id: str, kind: str) -> list:
        result: list[dict] = []
        inline_names: set[str] = set()
        for item in inline:
            if isinstance(item, dict):
                result.append(copy_item(item, session_id, kind))
                inline_names.add(str(item.get("name") or "artifact"))
        for item in descriptors:
            if not isinstance(item, dict):
                continue
            if item.get("artifact_id") or item.get("unavailable"):
                result.append(item)
            elif str(item.get("name") or "artifact") not in inline_names:
                result.append(copy_item(item, session_id, kind))
        return result

    while True:
        with db._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT id, session_id, attachments, images,
                   attachment_files, image_files, report_files
                   FROM chat_messages WHERE id > %s ORDER BY id LIMIT 50""",
                (last_id,),
            )
            rows = cur.fetchall()
        if not rows:
            break
        for row in rows:
            (message_id, session_id, attachments, images,
             attachment_files, image_files, report_files) = row
            last_id = message_id
            old = [*(attachments or []), *(images or []), *(attachment_files or []),
                   *(image_files or []), *(report_files or [])]
            if not any(isinstance(item, dict) and not item.get("artifact_id")
                       and not item.get("unavailable") for item in old):
                continue
            candidates += 1
            if not apply:
                continue
            sid = str(session_id)
            new_attachments = convert(attachments or [], attachment_files or [], sid, "upload")
            new_images = convert(images or [], image_files or [], sid, "tool_output")
            new_reports = convert(report_files or [], [], sid, "report")
            db.update_message_artifact_refs(message_id, new_attachments, new_images, new_reports)
            converted += 1
    return candidates, converted, missing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Copy and update legacy records")
    args = parser.parse_args()
    candidates, converted, missing = migrate_rows(args.apply)
    print(f"legacy messages: {candidates}; converted: {converted}; unavailable files: {missing}")
