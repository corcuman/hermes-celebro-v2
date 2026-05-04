"""Data models for the Celebro v2 Hermes memory provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class MemoryType(str, Enum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class MemoryImportance(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Memory:
    type: MemoryType
    content: str
    importance: MemoryImportance = MemoryImportance.MEDIUM
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=utc_now)
    last_accessed: datetime = field(default_factory=utc_now)
    access_count: int = 0
    session_id: str = ""
    source: str = "celebro_v2"


@dataclass
class SearchResult:
    memory: Memory
    score: float
