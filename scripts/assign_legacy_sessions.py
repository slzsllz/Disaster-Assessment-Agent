"""Assign confirmed legacy anonymous sessions to an existing account.

Defaults to preview. Run locally as the database operator, after migrating.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from agent.db import db, dict_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True, help="Existing account receiving the sessions")
    parser.add_argument("--session-id", action="append", type=uuid.UUID,
                        help="Restrict to this session; repeat for multiple sessions")
    parser.add_argument("--apply", action="store_true", help="Apply the assignment; defaults to preview")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=True)
    user = db.get_user_by_username(args.username.strip().lower())
    if not user or not user["is_active"]:
        parser.error("The target account does not exist or is disabled")
    with db._auth_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        query = "SELECT id, title FROM sessions WHERE user_id IS NULL"
        params = []
        if args.session_id:
            query += " AND id = ANY(%s::uuid[])"
            params.append([str(value) for value in args.session_id])
        query += " ORDER BY created_at"
        if args.apply:
            query += " FOR UPDATE"
        cur.execute(query, params)
        rows = cur.fetchall()
        print(f"Target account: {user['username']}; unassigned sessions: {len(rows)}")
        for row in rows:
            print(f"  {row['id']}  {row['title'] or '(untitled)'}")
        if not args.apply:
            print("Preview only. Add --apply after confirming ownership.")
            return
        if rows:
            cur.execute("UPDATE sessions SET user_id = %s WHERE id = ANY(%s::uuid[]) AND user_id IS NULL",
                        (user["id"], [str(row["id"]) for row in rows]))
            print(f"Assigned {cur.rowcount} sessions.")
        conn.commit()


if __name__ == "__main__":
    main()
