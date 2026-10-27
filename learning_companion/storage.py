"""Session metadata and complete transcripts, separate from compacted checkpoints."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


class SessionNotFound(ValueError):
    pass


class SQLiteStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS learning_sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                pending_message_id TEXT
            );
            CREATE TABLE IF NOT EXISTS learning_messages (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL REFERENCES learning_sessions(id),
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS learning_messages_session
            ON learning_messages(session_id, sequence);
            CREATE TABLE IF NOT EXISTS learning_sandbox_runs (
                reply_id TEXT NOT NULL REFERENCES learning_messages(id),
                call_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (reply_id, call_id)
            );
        """)

    def close(self) -> None:
        self.conn.close()

    def create_session(self) -> str:
        session_id = uuid4().hex
        timestamp = now()
        with self.conn:
            self.conn.execute(
                "INSERT INTO learning_sessions VALUES (?, ?, ?, ?, NULL)",
                (session_id, "新的学习会话", timestamp, timestamp),
            )
        return session_id

    def session(self, session_id: str) -> sqlite3.Row:
        row = self.conn.execute(
            "SELECT * FROM learning_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise SessionNotFound("找不到该会话，请用 --list 查看完整会话 ID。")
        return row

    def sessions(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM learning_sessions ORDER BY updated_at DESC"
        ).fetchall()

    def history(self, session_id: str) -> list[sqlite3.Row]:
        self.session(session_id)
        return self.conn.execute(
            "SELECT * FROM learning_messages WHERE session_id = ? ORDER BY sequence",
            (session_id,),
        ).fetchall()

    def start_turn(self, session_id: str, content: str) -> str:
        message_id = uuid4().hex
        timestamp = now()
        with self.conn:
            cursor = self.conn.execute(
                "UPDATE learning_sessions SET pending_message_id = ?, updated_at = ? "
                "WHERE id = ? AND pending_message_id IS NULL",
                (message_id, timestamp, session_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("该会话有尚未完成的请求，请输入 /retry，或 /new 新建会话。")
            self.conn.execute(
                "INSERT INTO learning_messages (id, session_id, role, content, created_at) "
                "VALUES (?, ?, 'user', ?, ?)",
                (message_id, session_id, content, timestamp),
            )
            self.conn.execute(
                "UPDATE learning_sessions SET title = ? WHERE id = ? AND "
                "(SELECT COUNT(*) FROM learning_messages WHERE session_id = ?) = 1",
                (" ".join(content.split())[:40], session_id, session_id),
            )
        return message_id

    def sandbox_history(self, session_id: str) -> list[dict]:
        return [
            json.loads(row["payload"])
            for row in self.conn.execute(
                "SELECT r.payload FROM learning_sandbox_runs r "
                "JOIN learning_messages m ON m.id=r.reply_id WHERE m.session_id=? "
                "ORDER BY m.sequence, r.rowid",
                (session_id,),
            )
        ]

    def finish_turn(
        self, session_id: str, message_id: str, content: str, runs: list[dict] | None = None
    ) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO learning_messages "
                "(id, session_id, role, content, created_at) VALUES (?, ?, 'assistant', ?, ?)",
                (message_id, session_id, content, now()),
            )
            self.conn.execute(
                "UPDATE learning_sessions SET pending_message_id = NULL, updated_at = ? "
                "WHERE id = ?",
                (now(), session_id),
            )
            for run in runs or []:
                self.conn.execute(
                    "INSERT OR IGNORE INTO learning_sandbox_runs VALUES (?, ?, ?)",
                    (message_id, run["call_id"], json.dumps(run, ensure_ascii=False)),
                )
