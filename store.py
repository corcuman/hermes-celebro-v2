"""SQLite cold store for Celebro v2."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .models import Memory, MemoryImportance, MemoryType, SearchResult

_STOPWORDS = {
    "a", "al", "de", "del", "el", "en", "es", "la", "las", "lo", "los",
    "me", "mi", "mis", "para", "por", "que", "se", "su", "sus", "tu",
    "un", "una", "unas", "unos", "y", "the", "and", "for", "with",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    content TEXT NOT NULL UNIQUE,
    importance INTEGER NOT NULL DEFAULT 2,
    tags TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    source TEXT NOT NULL DEFAULT 'celebro_v2',
    session_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_memories_access ON memories(last_accessed DESC);

CREATE TABLE IF NOT EXISTS profiles (
    user_id TEXT PRIMARY KEY,
    name TEXT,
    preferences TEXT NOT NULL DEFAULT '{}',
    facts TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    summary TEXT,
    message_count INTEGER NOT NULL DEFAULT 0
);
"""


class MemoryStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def add(self, memory: Memory) -> Memory:
        with self._lock:
            now = _format_dt(memory.created_at)
            try:
                self._conn.execute(
                    """
                    INSERT INTO memories
                    (id, type, content, importance, tags, metadata, source, session_id,
                     created_at, last_accessed, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory.id,
                        memory.type.value,
                        memory.content.strip(),
                        int(memory.importance.value),
                        json.dumps(memory.tags, ensure_ascii=False),
                        json.dumps(memory.metadata, ensure_ascii=False),
                        memory.source,
                        memory.session_id,
                        now,
                        _format_dt(memory.last_accessed),
                        memory.access_count,
                    ),
                )
            except sqlite3.IntegrityError:
                row = self._conn.execute(
                    "SELECT * FROM memories WHERE content = ?", (memory.content.strip(),)
                ).fetchone()
                return self._row_to_memory(row)
            self._conn.commit()
            return memory

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def search(
        self,
        query: str,
        *,
        memory_type: MemoryType | None = None,
        min_importance: MemoryImportance = MemoryImportance.LOW,
        tags: list[str] | None = None,
        source: str | None = None,
        limit: int = 8,
    ) -> list[SearchResult]:
        terms = _query_terms(query)
        where = ["importance >= ?"]
        params: list = [int(min_importance.value)]

        if memory_type:
            where.append("type = ?")
            params.append(memory_type.value)

        if source:
            where.append("source = ?")
            params.append(source)

        if tags:
            tag_clauses = " AND ".join("lower(tags) LIKE ?" for _ in tags)
            where.append(f"({tag_clauses})")
            params.extend(f"%{tag.lower()}%" for tag in tags)

        if terms:
            where.append("(" + " OR ".join("lower(content) LIKE ?" for _ in terms) + ")")
            params.extend(f"%{term}%" for term in terms)

        sql = f"""
            SELECT * FROM memories
            WHERE {' AND '.join(where)}
            ORDER BY importance DESC, last_accessed DESC
            LIMIT ?
        """
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
            if not rows and query:
                fallback_where = ["importance >= ?"]
                fallback_params: list = [int(min_importance.value)]
                if memory_type:
                    fallback_where.append("type = ?")
                    fallback_params.append(memory_type.value)
                if source:
                    fallback_where.append("source = ?")
                    fallback_params.append(source)
                if tags:
                    tag_clauses = " AND ".join("lower(tags) LIKE ?" for _ in tags)
                    fallback_where.append(f"({tag_clauses})")
                    fallback_params.extend(f"%{tag.lower()}%" for tag in tags)
                fallback_params.append(limit)
                rows = self._conn.execute(
                    f"""
                    SELECT * FROM memories
                    WHERE {' AND '.join(fallback_where)}
                    ORDER BY importance DESC, last_accessed DESC
                    LIMIT ?
                    """,
                    fallback_params,
                ).fetchall()
            return [self._row_to_result(row, terms) for row in rows]

    def list_recent(self, limit: int = 20) -> list[Memory]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [self._row_to_memory(row) for row in rows]

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    def update_access(self, memory_id: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE memories
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id = ?
                """,
                (_format_dt(datetime.now(timezone.utc)), memory_id),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            type=MemoryType(row["type"]),
            content=row["content"],
            importance=MemoryImportance(int(row["importance"])),
            tags=json.loads(row["tags"] or "[]"),
            metadata=json.loads(row["metadata"] or "{}"),
            source=row["source"],
            session_id=row["session_id"] or "",
            created_at=_parse_dt(row["created_at"]),
            last_accessed=_parse_dt(row["last_accessed"]),
            access_count=int(row["access_count"] or 0),
        )

    @classmethod
    def _row_to_result(cls, row: sqlite3.Row, terms: list[str]) -> SearchResult:
        memory = cls._row_to_memory(row)
        content = memory.content.lower()
        matches = sum(1 for term in terms if term in content)
        score = 0.25 if terms and matches == 0 else 1.0
        if terms and matches:
            score = min(1.0, 0.45 + (matches / len(terms)) * 0.55)
        return SearchResult(memory=memory, score=round(score, 4))


def _query_terms(query: str) -> list[str]:
    words = re.findall(r"[\wáéíóúüñÁÉÍÓÚÜÑ]+", (query or "").lower())
    terms: list[str] = []
    for word in words:
        if len(word) <= 2 or word in _STOPWORDS:
            continue
        terms.append(word)
        if len(word) >= 7:
            terms.append(word[:6])
        elif len(word) >= 5:
            terms.append(word[:4])
    return list(dict.fromkeys(terms))


def _format_dt(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _parse_dt(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return datetime.now(timezone.utc)
