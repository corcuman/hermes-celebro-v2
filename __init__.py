"""Celebro v2 local persistent memory provider for Hermes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .extractor import extract_user_memories
from .models import Memory, MemoryImportance, MemoryType
from .store import MemoryStore

logger = logging.getLogger(__name__)


REMEMBER_SCHEMA = {
    "name": "celebro_remember",
    "description": "Persist a durable user preference, fact, decision, event, or procedure to Celebro v2 memory.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Memory content as one clear sentence."},
            "memory_type": {"type": "string", "enum": ["semantic", "episodic", "procedural"], "description": "Kind of memory."},
            "importance": {"type": "integer", "enum": [1, 2, 3, 4], "description": "1 low, 2 medium, 3 high, 4 critical."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags."},
        },
        "required": ["content"],
    },
}

SEARCH_SCHEMA = {
    "name": "celebro_search",
    "description": "Search Celebro v2 persistent memory for relevant past facts, preferences, events, and procedures.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "top_k": {"type": "integer", "description": "Maximum results, default 8."},
            "memory_type": {"type": "string", "enum": ["semantic", "episodic", "procedural"], "description": "Optional type filter."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tag filter — only return memories that have ALL specified tags."},
            "source": {"type": "string", "description": "Optional source filter (e.g. 'tool', 'builtin_mirror', 'session_extract')."},
        },
        "required": ["query"],
    },
}

FORGET_SCHEMA = {
    "name": "celebro_forget",
    "description": "Delete a Celebro v2 memory by id.",
    "parameters": {
        "type": "object",
        "properties": {"memory_id": {"type": "string", "description": "Memory id."}},
        "required": ["memory_id"],
    },
}

STATS_SCHEMA = {
    "name": "celebro_stats",
    "description": "Show Celebro v2 memory counts and backend status.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


class CelebroV2MemoryProvider(MemoryProvider):
    def __init__(self) -> None:
        self._config = _load_config()
        self._store: MemoryStore | None = None
        self._vector = None
        self._session_id = ""
        self._read_only = False
        self._min_score = float(self._config.get("min_score", 0.3))
        self._top_k = int(self._config.get("top_k", 8))

    @property
    def name(self) -> str:
        return "celebro_v2"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = Path(kwargs.get("hermes_home") or _hermes_home())
        agent_context = kwargs.get("agent_context", "primary")
        self._read_only = agent_context not in ("", "primary")
        base_dir = _expand_path(self._config.get("data_dir", "$HERMES_HOME/celebro_v2"), hermes_home)
        db_path = _expand_path(self._config.get("db_path", str(base_dir / "celebro_v2.db")), hermes_home)

        self._session_id = session_id
        self._store = MemoryStore(db_path)

        if str(self._config.get("vector_enabled", "true")).lower() == "true":
            try:
                from .retrieval import VectorIndex
                self._vector = VectorIndex(
                    _expand_path(self._config.get("chroma_dir", str(base_dir / "chroma")), hermes_home),
                    embedding_model=str(self._config.get("embedding_model", "nomic-embed-text")),
                    ollama_host=str(self._config.get("ollama_host", "http://localhost:11434")),
                )
            except Exception as e:
                logger.warning("Celebro v2 vector index disabled: %s", e)
                self._vector = None

    def system_prompt_block(self) -> str:
        count = self._store.count() if self._store else 0
        vector_status = "vector recall enabled" if self._vector else "SQLite recall only"
        return (
            "# Celebro v2 Memory\n"
            f"Active with {count} stored memories ({vector_status}).\n"
            "Use celebro_search before answering questions that may depend on prior context. "
            "Use celebro_remember for durable user preferences, facts, decisions, and procedures."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._store or not query:
            return ""
        results = self._search(query, top_k=self._top_k)
        if not results:
            return ""
        lines = []
        for result in results:
            m = result.memory
            self._store.update_access(m.id)
            tag_str = f" [{', '.join(m.tags)}]" if m.tags else ""
            lines.append(f"- ({result.score:.2f}) [{m.type.value}]{tag_str} {m.content}")
        return "## Celebro v2 Memory\n" + "\n".join(lines)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        return None

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        return None

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if self._read_only or not self._store:
            return
        if str(self._config.get("auto_extract", "true")).lower() != "true":
            return
        for item in extract_user_memories(messages):
            self._remember(
                content=item["content"],
                memory_type=item["type"],
                importance=item["importance"],
                tags=item["tags"],
                source="session_extract",
            )

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if self._read_only or action != "add" or not content:
            return
        memory_type = MemoryType.SEMANTIC if target == "user" else MemoryType.PROCEDURAL
        tags = ["builtin", target]
        self._remember(content=content, memory_type=memory_type, importance=MemoryImportance.HIGH, tags=tags, source="builtin_mirror")

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [REMEMBER_SCHEMA, SEARCH_SCHEMA, FORGET_SCHEMA, STATS_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            if tool_name == "celebro_remember":
                return self._tool_remember(args)
            if tool_name == "celebro_search":
                return self._tool_search(args)
            if tool_name == "celebro_forget":
                return self._tool_forget(args)
            if tool_name == "celebro_stats":
                return self._tool_stats()
            return tool_error(f"Unknown Celebro v2 tool: {tool_name}")
        except Exception as e:
            return tool_error(f"Celebro v2 error: {e}")

    def get_config_schema(self):
        return [
            {"key": "data_dir", "description": "Celebro v2 data directory", "default": "$HERMES_HOME/celebro_v2"},
            {"key": "vector_enabled", "description": "Enable Chroma/Ollama semantic recall", "default": "true", "choices": ["true", "false"]},
            {"key": "embedding_model", "description": "Ollama embedding model", "default": "nomic-embed-text"},
            {"key": "ollama_host", "description": "Ollama host URL", "default": "http://localhost:11434"},
            {"key": "auto_extract", "description": "Extract simple durable memories at session end", "default": "true", "choices": ["true", "false"]},
            {"key": "top_k", "description": "Prefetch result count", "default": "8"},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        import yaml

        path = Path(hermes_home) / "config.yaml"
        config = {}
        if path.exists():
            config = yaml.safe_load(path.read_text()) or {}
        config.setdefault("memory", {})
        config["memory"]["celebro_v2"] = values
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def shutdown(self) -> None:
        if self._store:
            self._store.close()
        self._store = None
        self._vector = None

    def _tool_remember(self, args: dict) -> str:
        memory = self._remember(
            content=str(args["content"]),
            memory_type=MemoryType(args.get("memory_type", "semantic")),
            importance=MemoryImportance(int(args.get("importance", 2))),
            tags=list(args.get("tags") or []),
            source="tool",
        )
        return json.dumps({"ok": True, "id": memory.id, "content": memory.content}, ensure_ascii=False)

    def _tool_search(self, args: dict) -> str:
        memory_type = MemoryType(args["memory_type"]) if args.get("memory_type") else None
        tags = list(args["tags"]) if args.get("tags") else None
        source = str(args["source"]) if args.get("source") else None
        results = self._search(
            str(args["query"]),
            top_k=int(args.get("top_k", self._top_k)),
            memory_type=memory_type,
            tags=tags,
            source=source,
        )
        payload = [
            {
                "id": r.memory.id,
                "score": r.score,
                "type": r.memory.type.value,
                "importance": int(r.memory.importance.value),
                "tags": r.memory.tags,
                "content": r.memory.content,
            }
            for r in results
        ]
        return json.dumps({"results": payload}, ensure_ascii=False)

    def _tool_forget(self, args: dict) -> str:
        if not self._store:
            return tool_error("Celebro v2 is not initialized")
        memory_id = str(args["memory_id"])
        deleted = self._store.delete(memory_id)
        if self._vector:
            try:
                self._vector.delete(memory_id)
            except Exception:
                pass
        return json.dumps({"ok": deleted, "id": memory_id}, ensure_ascii=False)

    def _tool_stats(self) -> str:
        count = self._store.count() if self._store else 0
        vector_count = 0
        if self._vector:
            try:
                vector_count = self._vector.count()
            except Exception:
                vector_count = 0
        return json.dumps({"memories": count, "vector_indexed": vector_count, "read_only": self._read_only}, ensure_ascii=False)

    def _remember(
        self,
        *,
        content: str,
        memory_type: MemoryType,
        importance: MemoryImportance,
        tags: list[str],
        source: str,
    ) -> Memory:
        if not self._store:
            raise RuntimeError("Celebro v2 is not initialized")
        memory = Memory(
            content=content.strip(),
            type=memory_type,
            importance=importance,
            tags=tags,
            source=source,
            session_id=self._session_id,
        )
        saved = self._store.add(memory)
        if self._vector:
            try:
                self._vector.add(saved)
            except Exception as e:
                logger.warning("Celebro v2 vector index disabled after add failure: %s", e)
                self._vector = None
        return saved

    def _search(self, query: str, *, top_k: int, memory_type: MemoryType | None = None, tags: list[str] | None = None, source: str | None = None):
        merged = {}
        if self._vector:
            try:
                for result in self._vector.search(query, top_k=top_k, memory_type=memory_type, min_score=self._min_score):
                    merged[result.memory.id] = result
            except Exception as e:
                logger.warning("Celebro v2 vector index disabled after search failure: %s", e)
                self._vector = None
        if self._store:
            for result in self._store.search(query, memory_type=memory_type, tags=tags, source=source, limit=top_k):
                merged.setdefault(result.memory.id, result)
        return sorted(merged.values(), key=lambda r: (r.score, int(r.memory.importance.value)), reverse=True)[:top_k]


def register(ctx) -> None:
    ctx.register_memory_provider(CelebroV2MemoryProvider())


def _load_config() -> dict:
    try:
        import yaml
        path = Path(_hermes_home()) / "config.yaml"
        data = yaml.safe_load(path.read_text()) if path.exists() else {}
        memory = (data or {}).get("memory", {})
        return memory.get("celebro_v2", {}) or {}
    except Exception:
        return {}


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home()


def _expand_path(value: str | Path, hermes_home: Path) -> Path:
    text = str(value).replace("$HERMES_HOME", str(hermes_home)).replace("${HERMES_HOME}", str(hermes_home))
    return Path(text).expanduser()
