"""SQLite-based storage backend for Celebro v2 memories (simplified).
"""

import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Optional, List, Any

from .models import Memory, MemoryType, MemoryImportance

# Simple, thread-safe SQLite-backed store for memories used by Hermes Celebro v2.

class _SqliteStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path.as_posix(), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.initialize_db()

    def initialize_db(self):
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance INTEGER NOT NULL,
                    tags TEXT,
                    metadata TEXT,
                    source TEXT NOT NULL,
                    session_id TEXT,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER NOT NULL
                )
                """
            )
            self.conn.execute("PRAGMA journal_mode=WAL;")

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        tags = json.loads(row["tags"] or "[]")
        metadata = json.loads(row["metadata"] or "{}")
        return Memory(
            id=row["id"],
            type=MemoryType(row["type"]),
            content=row["content"],
            importance=MemoryImportance(int(row["importance"])),
            tags=tags,
            metadata=metadata,
            source=row["source"],
            session_id=row["session_id"] or "",
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            access_count=int(row["access_count"] or 0),
        )

    def add(self, memory: Memory) -> Memory:
        if not memory.id:
            memory.id = str(uuid.uuid4())
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO memories (id, type, content, importance, tags, metadata, source, session_id, created_at, last_accessed, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.type.value,
                    memory.content,
                    int(memory.importance.value),
                    json.dumps(memory.tags),
                    json.dumps(memory.metadata),
                    memory.source,
                    memory.session_id,
                    memory.created_at,
                    memory.last_accessed,
                    memory.access_count,
                ),
            )
            self.conn.commit()
        return memory

    def get_by_id(self, memory_id: str) -> Optional[Memory]:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
            row = cur.fetchone()
        return self._row_to_memory(row) if row else None

    def update(self, memory: Memory) -> bool:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                UPDATE memories SET
                    type = ?, content = ?, importance = ?, tags = ?, metadata = ?, source = ?, session_id = ?, created_at = ?, last_accessed = ?, access_count = ?
                WHERE id = ?
                """,
                (
                    memory.type.value,
                    memory.content,
                    int(memory.importance.value),
                    json.dumps(memory.tags),
                    json.dumps(memory.metadata),
                    memory.source,
                    memory.session_id,
                    memory.created_at,
                    memory.last_accessed,
                    memory.access_count,
                    memory.id,
                ),
            )
            self.conn.commit()
            return cur.rowcount > 0

    def delete(self, memory_id: str) -> int:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self.conn.commit()
            return cur.rowcount

    def find_by_content_substring(self, substring: str, *, source: str | None = None) -> List[Memory]:
        with self._lock:
            if substring is None:
                return []
            s = str(substring).strip().lower()[:80]
            sql = "SELECT * FROM memories WHERE lower(content) LIKE ?"
            params = (f"%{s}%",)
            if source:
                sql += " AND source = ?"
                params += (source,)
            cur = self.conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [self._row_to_memory(r) for r in rows]

    def find_by_exact_content(self, content: str, *, source: str | None = None) -> List[Memory]:
        s = content.strip().lower()
        if not s:
            return []
        with self._lock:
            sql = "SELECT * FROM memories WHERE lower(content) = ?"
            params = (s,)
            if source:
                sql += " AND source = ?"
                params += (source,)
            cur = self.conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [self._row_to_memory(r) for r in rows]

    def delete_by_content_substring(self, substring: str, *, source: str | None = None) -> int:
        memories = self.find_by_content_substring(substring, source=source)
        deleted = 0
        for m in memories:
            if self.delete(m.id) > 0:
                deleted += 1
        return deleted

    def count(self) -> int:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM memories")
            return cur.fetchone()[0]

    def close(self) -> None:
        with self._lock:
            if self.conn:
                self.conn.close()
                self.conn = None
