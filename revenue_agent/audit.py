"""Local audit trail (SQLite). Stores masked text only."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "data" / "audit.db"


def _conn():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS audit(
        id INTEGER PRIMARY KEY, ts TEXT, officer TEXT, thread TEXT, domain TEXT,
        task_type TEXT, request_masked TEXT, decision TEXT, output_masked TEXT)""")
    return c


def log(officer, thread, domain, task_type, request_masked, decision, output_masked):
    with _conn() as c:
        c.execute("INSERT INTO audit(ts,officer,thread,domain,task_type,request_masked,decision,output_masked) "
                  "VALUES(?,?,?,?,?,?,?,?)",
                  (datetime.now().isoformat(timespec="seconds"), officer, thread, domain, task_type,
                   request_masked, decision, output_masked))


def recent(limit=200):
    with _conn() as c:
        cur = c.execute("SELECT ts,officer,domain,task_type,decision,request_masked FROM audit ORDER BY id DESC LIMIT ?", (limit,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
