"""Small deterministic extractor for durable user memories."""

from __future__ import annotations

import re

from .models import MemoryImportance, MemoryType

_PATTERNS = [
    (re.compile(r"\bme llamo\s+([^.,;\n]+)", re.I), "El usuario se llama {value}", ["perfil"], MemoryImportance.HIGH),
    (re.compile(r"\bmi nombre es\s+([^.,;\n]+)", re.I), "El usuario se llama {value}", ["perfil"], MemoryImportance.HIGH),
    (re.compile(r"\bprefiero\s+([^.,;\n]+)", re.I), "El usuario prefiere {value}", ["preferencias"], MemoryImportance.HIGH),
    (re.compile(r"\bno me gusta\s+([^.,;\n]+)", re.I), "Al usuario no le gusta {value}", ["preferencias"], MemoryImportance.HIGH),
    (re.compile(r"\busa siempre\s+([^.,;\n]+)", re.I), "El usuario quiere que se use siempre {value}", ["instrucciones"], MemoryImportance.CRITICAL),
    (re.compile(r"\btrabajo (?:con|en|como)\s+([^.,;\n]+)", re.I), "El usuario trabaja con/en/como {value}", ["perfil"], MemoryImportance.HIGH),
]


def extract_user_memories(messages: list[dict], limit: int = 8) -> list[dict]:
    memories: list[dict] = []
    seen: set[str] = set()
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = _text_content(msg.get("content", ""))
        for pattern, template, tags, importance in _PATTERNS:
            for match in pattern.finditer(content):
                value = match.group(1).strip()
                if not value or len(value) > 160:
                    continue
                memory = template.format(value=value)
                key = memory.lower()
                if key in seen:
                    continue
                seen.add(key)
                memories.append({
                    "content": memory,
                    "type": MemoryType.SEMANTIC,
                    "importance": importance,
                    "tags": tags,
                })
                if len(memories) >= limit:
                    return memories
    return memories


def _text_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "\n".join(parts)
    return str(content)
