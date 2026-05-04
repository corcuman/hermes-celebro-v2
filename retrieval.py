"""Optional Chroma vector index for Celebro v2."""

from __future__ import annotations

import logging
from pathlib import Path

from .embeddings import OllamaEmbedder
from .models import Memory, MemoryImportance, MemoryType, SearchResult

logger = logging.getLogger(__name__)


class VectorIndex:
    def __init__(
        self,
        persist_dir: str | Path,
        *,
        embedding_model: str = "nomic-embed-text",
        ollama_host: str = "http://localhost:11434",
    ) -> None:
        import chromadb
        from chromadb.config import Settings

        persist_dir = Path(persist_dir).expanduser()
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._embedder = OllamaEmbedder(model=embedding_model, host=ollama_host)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="celebro_v2_memories",
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, memory: Memory) -> None:
        self._collection.upsert(
            ids=[memory.id],
            embeddings=[self._embedder.embed(memory.content)],
            documents=[memory.content],
            metadatas=[{
                "type": memory.type.value,
                "importance": int(memory.importance.value),
                "tags": ",".join(memory.tags),
                "source": memory.source,
                "session_id": memory.session_id,
            }],
        )

    def delete(self, memory_id: str) -> None:
        self._collection.delete(ids=[memory_id])

    def search(
        self,
        query: str,
        *,
        memory_type: MemoryType | None = None,
        top_k: int = 8,
        min_score: float = 0.3,
    ) -> list[SearchResult]:
        count = self._collection.count()
        if count == 0:
            return []
        where = {"type": memory_type.value} if memory_type else None
        results = self._collection.query(
            query_embeddings=[self._embedder.embed(query)],
            n_results=min(top_k, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        found = []
        for idx, (doc, meta, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            score = 1.0 - (float(distance) / 2.0)
            if score < min_score:
                continue
            memory = Memory(
                id=results["ids"][0][idx],
                type=MemoryType(meta.get("type", "semantic")),
                content=doc,
                importance=MemoryImportance(int(meta.get("importance", 2))),
                tags=[t for t in str(meta.get("tags", "")).split(",") if t],
                source=meta.get("source", "celebro_v2"),
                session_id=meta.get("session_id", ""),
            )
            found.append(SearchResult(memory=memory, score=round(score, 4)))
        return found

    def count(self) -> int:
        return self._collection.count()
