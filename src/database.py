"""
SQLite store for images uploaded so the model can be retrained.

The file is data/bean_leaf.db. Each upload is one row in uploaded_images,
including the original bytes. Preprocessing writes the resized image back
into the same row. Retraining then reads those rows and marks them used.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "bean_leaf.db"
TABLE = "uploaded_images"

_lock = threading.RLock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS uploaded_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_name TEXT NOT NULL,
    label TEXT NOT NULL,
    raw_path TEXT,
    content BLOB NOT NULL,
    content_sha TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    width INTEGER,
    height INTEGER,
    uploaded_at TEXT NOT NULL,
    preprocessed INTEGER NOT NULL DEFAULT 0,
    processed BLOB,
    processed_width INTEGER,
    processed_height INTEGER,
    processed_path TEXT,
    preprocessed_at TEXT,
    used_for_training INTEGER NOT NULL DEFAULT 0,
    trained_in_version TEXT
);
"""


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db() -> None:
    with _lock:
        conn = connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()


def insert_upload(original_name: str, label: str, content: bytes,
                  width: int | None, height: int | None, raw_path: str | None) -> int:
    """Save one uploaded image. The bytes live in the database, not only on disk."""
    init_db()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sha = hashlib.sha256(content).hexdigest()
    with _lock:
        conn = connect()
        try:
            cur = conn.execute(
                """INSERT INTO uploaded_images
                   (original_name, label, raw_path, content, content_sha, byte_size,
                    width, height, uploaded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (original_name, label, raw_path, content, sha, len(content), width, height, now),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()


def _all(sql: str, params: tuple = ()) -> list[dict]:
    init_db()
    with _lock:
        conn = connect()
        try:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()


def count_uploads() -> int:
    init_db()
    with _lock:
        conn = connect()
        try:
            return int(conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0])
        finally:
            conn.close()


def list_uploads(limit: int = 50) -> list[dict]:
    rows = _all(
        """SELECT id, original_name, label, byte_size, width, height, uploaded_at,
                  preprocessed, processed_width, processed_height, processed_path,
                  preprocessed_at, used_for_training, trained_in_version
           FROM uploaded_images ORDER BY id DESC LIMIT ?""",
        (limit,),
    )
    for r in rows:
        r["preprocessed"] = bool(r["preprocessed"])
        r["used_for_training"] = bool(r["used_for_training"])
    return rows


def unpreprocessed(limit: int = 500) -> list[dict]:
    return _all(
        """SELECT id, label, content FROM uploaded_images
           WHERE preprocessed = 0 ORDER BY id LIMIT ?""",
        (limit,),
    )


def ready_for_training() -> list[dict]:
    return _all(
        """SELECT id, label, raw_path, processed_path, processed
           FROM uploaded_images
           WHERE preprocessed = 1 AND used_for_training = 0
           ORDER BY id"""
    )


def mark_preprocessed(row_id: int, processed: bytes, width: int, height: int, path: str) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _lock:
        conn = connect()
        try:
            conn.execute(
                """UPDATE uploaded_images
                   SET preprocessed = 1, processed = ?, processed_width = ?, processed_height = ?,
                       processed_path = ?, preprocessed_at = ?
                   WHERE id = ?""",
                (processed, width, height, path, now, row_id),
            )
            conn.commit()
        finally:
            conn.close()


def mark_used(ids: list[int]) -> None:
    if not ids:
        return
    with _lock:
        conn = connect()
        try:
            conn.executemany(
                "UPDATE uploaded_images SET used_for_training = 1 WHERE id = ?",
                [(i,) for i in ids],
            )
            conn.commit()
        finally:
            conn.close()


def set_trained_version(ids: list[int], version: str) -> None:
    if not ids:
        return
    with _lock:
        conn = connect()
        try:
            conn.executemany(
                "UPDATE uploaded_images SET trained_in_version = ? WHERE id = ?",
                [(version, i) for i in ids],
            )
            conn.commit()
        finally:
            conn.close()
