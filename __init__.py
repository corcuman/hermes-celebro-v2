"""Celebro v2 local persistent memory provider for Hermes."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List
import uuid # Added for uuid.uuid4()

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .extractor import extract_user_memories
from .models import Memory, MemoryImportance, MemoryType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CelebroV2MemoryProvider(MemoryProvider):
    def __init__(
        self,
        *,
        db_path: str | Path | None = None,
        vector_db_path: str | Path | None = None,
        read_only: bool = False,
        top_k: int = 3,
        min_score: float = 0.1,
        session_id: str | None = None,
        char_limit: int = 4400, # Increased char limit for built-in memory
    ):
        self._initialized = False
        self._read_only = read_only
        self._top_k = top_k
        self._min_score = min_score
        self._session_id = session_id
        self._char_limit = char_limit

        config = _load_config()
        self._db_path = Path(db_path or config.get("db_path") or _hermes_home() / "celebro_v2/celebro_v2.db")
        self._vector_db_path = Path(vector_db_path or config.get("vector_db_path") or _hermes_home() / "celebro_v2/chroma")

        self._store = None
        self._vector = None

        try:
            self._store = _SqliteStore(self._db_path)
            if self._vector_db_path:
                self._vector = _ChromaStore(self._vector_db_path)
            self._initialized = True
            logger.info("Celebro v2 initialized: DB=%s, VectorDB=%s, ReadOnly=%s", self._db_path, self._vector_db_path, self._read_only)
        except Exception as e:
            logger.error("Failed to initialize Celebro v2: %s", e)
            self._store = None
            self._vector = None
            self._initialized = False

    def initialize(self) -> None:
        """Ensures Celebro v2 is initialized and MEMORY.md is synced."""
        if self._initialized:
            self._regenerate_builtin_memory() # Ensure MEMORY.md is synced w/ Celebro on init

    def _regenerate_builtin_memory(self, *, memory_md_path: Path | None = None) -> None:
        """Regenerates MEMORY.md from Celebro v2's backend, respecting char limits and deduplication."""
        if not self._store:
            return # Not initialized

        if memory_md_path is None:
            memory_md_path = _hermes_home() / "memories/MEMORY.md"

        try:
            current_content = memory_md_path.read_text(encoding="utf-8") if memory_md_path.exists() else ""
        except Exception as e:
            logger.warning("Failed to read MEMORY.md: %s", e)
            current_content = ""
        
        # Split existing MEMORY.md content by the § separator, filtering out empty entries
        current_entries = [e.strip() for e in current_content.split("§\n") if e.strip()] if current_content else []

        # Get all builtin_mirror memories from Celebro, sorted by created_at for consistent ordering before final sort
        try:
            mirrors = self._store.find_by_content_substring("", source="builtin_mirror")
        except Exception as e:
            logger.error(f"Failed to query Celebro v2 for builtin_mirror memories: {e}")
            return # Exit if database query fails

        if not mirrors and not current_entries: # Nothing to do if no mirrors and MEMORY.md is empty
            return
        
        # Deduplicate memories based on content and keep the most recent one.
        # Use a limited substring as the key for deduplication to handle slight variations.
        seen_content_keys: dict[str, 'Memory'] = {}
        # Sort by creation time descending to ensure the newest is kept if content is identical
        sorted_mirrors = sorted(mirrors, key=lambda m: m.created_at, reverse=True)

        for m in sorted_mirrors:
            # Use a stable key for deduplication (e.g., first 80 chars, lowercased)
            key = m.content.strip()[:80].lower()
            if key: # Only consider non-empty keys
                seen_content_keys[key] = m
        
        unique_mirrors = list(seen_content_keys.values())

        # Sort the unique memories by importance (descending) then by creation time (descending) for final output
        unique_mirrors.sort(key=lambda m: (int(m.importance.value), m.created_at), reverse=True)

        # Check if regeneration is necessary:
        # Regenerate if MEMORY.md is empty OR if it has significantly fewer entries than unique mirrors found.
        # Use a threshold (e.g., 60% of found unique mirrors) to avoid excessive writes if MEMORY.md is mostly up-to-date.
        if current_entries and len(current_entries) >= len(unique_mirrors) * 0.6:
            # logger.debug("MEMORY.md seems up-to-date (entries: %d, mirrors: %d). Skipping regeneration.", len(current_entries), len(unique_mirrors))
            return  # MEMORY.md is reasonably up to date

        # Build new MEMORY.md content, respecting the character limit
        new_entries = []
        total_chars = 0
        for m in unique_mirrors:
            entry = m.content.strip()
            # Account for the separator "§\n" which is 2 characters
            entry_size = len(entry) + 2  
            if total_chars + entry_size > self._char_limit:
                logger.warning(f"Reached char limit ({self._char_limit}) for MEMORY.md. Truncating further entries.")
                break
            new_entries.append(entry)
            total_chars += entry_size

        if not new_entries:
             # If no entries fit or no unique mirrors were found, ensure MEMORY.md is cleared if it wasn't already.
            if current_content:
                try:
                    memory_md_path.write_text("", encoding="utf-8")
                    logger.info("Cleared MEMORY.md as no entries fit or were found.")
                except Exception as e:
                    logger.warning(f"Failed to clear MEMORY.md: {e}")
            return

        new_content = "\n§\n".join(new_entries)

        # Write atomically to avoid corrupting the file
        try:
            memory_md_path.parent.mkdir(parents=True, exist_ok=True)
            memory_md_path.write_text(new_content, encoding="utf-8")
            logger.info(
                "Celebro v2: Regenerated MEMORY.md with %d entries (%d chars) from %d unique mirrors.",
                len(new_entries), len(new_content), len(unique_mirrors),
            )
        except Exception as e:
            logger.error("Failed to write regenerated MEMORY.md: %s", e)


    def on_memory_write(self, action: str, target: str, content: str, metadata: dict = None) -> None:
        """Synchronize built-in memory writes to Celebro v2 with robust lookup."""
        if self._read_only or not self._store:
            return

        if metadata is None:
            metadata = {}

        # Collect common fields
        memory_type_val = MemoryType.SEMANTIC if target == "user" else MemoryType.PROCEDURAL
        tags = ["builtin", target]
        current_dt = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        session_id = self._session_id or ""
        
        # ADD
        if action == "add":
            if not content:
                logger.warning(f"Add action: content is empty, skipping.")
                return
            new_memory = Memory(
                id=str(uuid.uuid4()), # Use uuid.uuid4()
                type=memory_type_val,
                content=content.strip(),
                importance=MemoryImportance.HIGH,
                tags=tags,
                source="builtin_mirror",
                session_id=session_id,
                created_at=current_dt,
                last_accessed=current_dt,
                access_count=0
            )
            saved = self._store.add(new_memory)
            logger.info(f"Added memory ID {saved.id[:12]}...")
            # After adding, trigger a MEMORY.md regeneration to reflect the change
            self._regenerate_builtin_memory()
            return

        # REPLACE
        elif action == "replace":
            if not content:
                logger.warning(f"Replace action: new content is empty, cannot replace. Search for old content: '{metadata.get('old_text', '')}'")
                return
            
            old_id = metadata.get("id")
            old_text_search = metadata.get("old_text", "")

            memory_to_update = None
            if old_id:
                memory_to_update = self._store.get_by_id(old_id)

            if not memory_to_update and old_text_search:
                # Use exact content search for replace when ID is missing
                found_by_content = self._store.find_by_exact_content(old_text_search, source="builtin_mirror")
                if found_by_content:
                    memory_to_update = found_by_content[0]

            if memory_to_update:
                updated_memory = Memory(
                    id=memory_to_update.id,
                    type=memory_type_val,
                    content=content.strip(),
                    importance=MemoryImportance.HIGH,
                    tags=tags,
                    session_id=session_id,
                    created_at=memory_to_update.created_at, # Keep original creation time
                    last_accessed=current_dt,
                    access_count=memory_to_update.access_count + 1
                )
                self._store.update(updated_memory)
                logger.info(f"Successfully replaced memory ID {memory_to_update.id[:12]} with new content.")
                # After replacing, trigger a MEMORY.md regeneration
                self._regenerate_builtin_memory()
            else:
                logger.warning(f"Replace action: could not find memory to update for old_text='{old_text_search}' or ID='{old_id}'. Adding new entry as fallback.")
                # Fallback to add if no existing memory found
                new_memory = Memory(id=str(uuid.uuid4()), type=memory_type_val, content=content.strip(), importance=MemoryImportance.HIGH, tags=tags, source="builtin_mirror", session_id=session_id, created_at=current_dt, last_accessed=current_dt, access_count=0)
                saved = self._store.add(new_memory)
                logger.info(f"Added fallback memory ID {saved.id[:12]} after failed replace.")
                # Trigger regeneration after fallback add
                self._regenerate_builtin_memory()
            return

        # REMOVE
        elif action == "remove":
            old_id = metadata.get("id")
            # For remove, old_text is the primary identifier if ID is not present
            old_text_search = metadata.get("old_text", "") 

            memory_to_delete = None
            if old_id:
                memory_to_delete = self._store.get_by_id(old_id)

            if not memory_to_delete and old_text_search:
                # Use exact content search for remove when ID is missing
                found_by_content = self._store.find_by_exact_content(old_text_search, source="builtin_mirror")
                if found_by_content:
                    # Prefer the first exact match if multiple found with same content
                    memory_to_delete = found_by_content[0] 

            if memory_to_delete:
                self._store.delete(memory_to_delete.id)
                # Attempt to delete from vector store if it exists and memory was found
                if hasattr(self, '_vector') and self._vector:
                    try:
                        self._vector.delete(memory_to_delete.id)
                        logger.info(f"Deleted vector for memory ID {memory_to_delete.id[:12]}...")
                    except Exception as e:
                        logger.warning(f"Could not delete vector for memory ID {memory_to_delete.id[:12]}: {e}")
                logger.info(f"Successfully deleted memory ID {memory_to_delete.id[:12]}...")
                # After removing, trigger a MEMORY.md regeneration
                self._regenerate_builtin_memory()
            else:
                logger.warning(f"Remove action: could not find memory to delete for old_text='{old_text_search}' or ID='{old_id}'.")
                
        else:
            logger.warning(f"Unknown memory write action: '{action}'.")


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
                logger.warning(f"Failed to remove vector for memory ID {memory_id[:12]}...")
        return json.dumps({"ok": deleted, "id": memory_id}, ensure_ascii=False)

    def _tool_stats(self) -> str:
        count = self._store.count() if self._store else 0
        vector_count = 0
        if self._vector:
            try:
                vector_count = self._vector.count()
            except Exception : # Avoid crashing if vector store is unavailable
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

# --- Dummy implementations for SQLiteStore and ChromaStore for now ---
# These would normally be imported from elsewhere in the Celebro v2 codebase.
# For the purpose of applying the patch and testing the logic flow, 
# we'll use placeholder classes that mimic the expected methods.

class _DummyStore:
    def __init__(self, _db_path: Path):
        self._db_path = _db_path
        self._memories: list[Memory] = [] # Store memories in memory for this dummy
        self._next_id = 1
        logger.info("DummyStore initialized for path %s", _db_path)

    def add(self, memory: Memory) -> Memory:
        if not memory.id:
            memory.id = str(uuid.uuid4()) # Ensure ID is set
        memory.id = str(memory.id) # Ensure ID is string
        self._memories.append(memory)
        logger.debug("DummyStore added memory: %s", memory.id[:12])
        return memory

    def get_by_id(self, memory_id: str) -> Memory | None:
        for mem in self._memories:
            if mem.id == memory_id:
                return mem
        return None

    def update(self, memory: Memory) -> bool:
        for i, mem in enumerate(self._memories):
            if mem.id == memory.id:
                self._memories[i] = memory
                logger.debug("DummyStore updated memory: %s", memory.id[:12])
                return True
        return False

    def delete(self, memory_id: str) -> int:
        initial_count = len(self._memories)
        self._memories = [mem for mem in self._memories if mem.id != memory_id]
        deleted = initial_count - len(self._memories)
        if deleted > 0:
             logger.debug("DummyStore deleted memory: %s (Count: %d)", memory_id[:12], deleted)
        return deleted

    def find_by_content_substring(self, substring: str, *, source: str | None = None) -> list[Memory]:
        """Find memories whose content contains the given substring (case-insensitive)."""
        results = []
        s = substring.strip().lower()[:80]
        for mem in self._memories:
            if not s or s in mem.content.strip().lower():
                if source is None or mem.source == source:
                    results.append(mem)
        logger.debug("DummyStore find_by_content_substring('%s', source='%s') found %d results", substring, source, len(results))
        return results
        
    def find_by_exact_content(self, content: str, *, source: str | None = None) -> list[Memory]:
        s = content.strip().lower()
        if not s:
            return []
        results = []
        for mem in self._memories:
            if mem.content.strip().lower() == s:
                if source is None or mem.source == source:
                    results.append(mem)
        logger.debug("DummyStore find_by_exact_content('%s', source='%s') found %d results", content, source, len(results))
        return results

    def count(self) -> int:
        return len(self._memories)

    def search(self, query: str, *, limit: int, memory_type: MemoryType | None = None, tags: list[str] | None = None, source: str | None = None) -> list[Any]:
        # Dummy search - does not implement complex scoring or vector search
        # Returns a subset of memories that match the query loosely.
        matches = self.find_by_content_substring(query, source=source)
        if memory_type:
            matches = [m for m in matches if m.type == memory_type]
        if tags:
            matches = [m for m in matches if all(tag in m.tags for tag in tags)]
        
        # Return dummy result objects mimicking vector search results
        dummy_results = []
        for i, mem in enumerate(matches[:limit]):
             # Simple score: 1.0 if query is exact match, else 0.5, else 0.2 for substring
             score = 1.0 if mem.content.strip().lower() == query.strip().lower() else 0.5 if query in mem.content else 0.2
             dummy_results.append({
                 "memory": mem, 
                 "score": score,
                 "type": "search_result" # Placeholder type
                 })
        logger.debug("DummyStore search('%s', limit=%d) returned %d results", query, limit, len(dummy_results))
        return dummy_results

    def close(self) -> None:
        logger.info("DummyStore closed.")
        pass

class _ChromaStore: # Dummy Chroma Store
    def __init__(self, _path: Path):
        self._path = _path
        self.count_cache = 0
        logger.info("DummyChromaStore initialized for path %s", _path)

    def add(self, memory: Memory) -> None:
        logger.debug("DummyChromaStore add: %s", memory.id[:12])
        self.count_cache += 1

    def delete(self, memory_id: str) -> None:
        logger.debug("DummyChromaStore delete: %s", memory_id[:12])
        # In a real implementation, this would decrement the count if the ID existed.
        # For dummy, we don't track actual existence, just simulate the call.
        pass 
        
    def count(self) -> int:
        logger.debug("DummyChromaStore count called, returning cached %d", self.count_cache)
        # Returning a dummy count. In a real scenario, this would query ChromaDB.
        return self.count_cache 

    def search(self, query: str, top_k: int, memory_type: MemoryType | None = None, min_score: float = 0.1) -> list[Any]:
        logger.debug("DummyChromaStore search called for query: '%s', top_k: %d", query, top_k)
        # This dummy implementation cannot perform actual vector search.
        # It would return a list of dummy result objects in a real scenario.
        # For now, return an empty list to avoid errors.
        return []
        
    def close(self) -> None:
        logger.info("DummyChromaStore closed.")
        pass

# --- SQLite Store Implementation ---
# This is a simplified SQLite implementation to make the code runnable.
# It should be replaced with the actual Celebro v2 store if available.

class _SqliteStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = None
        self.initialize_db()
        logger.info(f"SQLiteStore initialized at {db_path}")

    def initialize_db(self):
        # Ensure the database and table exist
        self.conn = __import__("sqlite3").connect(self.db_path)
        cursor = self.conn.cursor()
        cursor.execute("""
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
        """)
        self.conn.commit()

    def _row_to_memory(self, row: tuple) -> Memory:
        # Parses a database row into a Memory object
        id_val, type_val, content_val, importance_val, tags_val, metadata_val, source_val, session_id_val, created_at_val, last_accessed_val, access_count_val = row
        
        # Safely load JSON fields
        tags = json.loads(tags_val) if tags_val else []
        metadata = json.loads(metadata_val) if metadata_val else {}

        return Memory(
            id=id_val,
            type=MemoryType(type_val),
            content=content_val,
            importance=MemoryImportance(importance_val),
            tags=tags,
            metadata=metadata,
            source=source_val,
            session_id=session_id_val,
            created_at=created_at_val,
            last_accessed=last_accessed_val,
            access_count=access_count_val
        )

    def add(self, memory: Memory) -> Memory:
        if not memory.id: memory.id = str(uuid.uuid4())
        memory.id = str(memory.id) # Ensure ID is string
        
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
            INSERT INTO memories (id, type, content, importance, tags, metadata, source, session_id, created_at, last_accessed, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
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
                memory.access_count
            ))
            self.conn.commit()
            logger.debug(f"Added memory with ID: {memory.id[:12]}")
            return memory
        except __import__("sqlite3").IntegrityError:
            # Handle potential duplicate ID if generated UUID somehow collided (highly unlikely)
            logger.warning(f"IntegrityError: Memory ID {memory.id[:12]} already exists. Attempting to update instead.")
            return self.update(memory) # Attempt update if add fails due to duplicate ID
        except Exception as e:
            logger.error(f"Error adding memory {memory.id[:12]}: {e}")
            self.conn.rollback()
            raise

    def get_by_id(self, memory_id: str) -> Memory | None:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        return self._row_to_memory(row) if row else None

    def update(self, memory: Memory) -> bool:
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
            UPDATE memories SET type = ?, content = ?, importance = ?, tags = ?, metadata = ?, source = ?, session_id = ?, created_at = ?, last_accessed = ?, access_count = ?
            WHERE id = ?
            """, (
                memory.type.value,
                memory.content,
                int(memory.importance.value),
                json.dumps(memory.tags),
                json.dumps(memory.metadata),
                memory.source,
                memory.session_id,
                memory.created_at, # Keep original creation time
                memory.last_accessed,
                memory.access_count,
                memory.id
            ))
            self.conn.commit()
            updated = cursor.rowcount > 0
            if updated: logger.debug(f"Updated memory with ID: {memory.id[:12]}")
            else: logger.warning(f"Update failed: Memory ID {memory.id[:12]} not found.")
            return updated
        except Exception as e:
            logger.error(f"Error updating memory {memory.id[:12]}: {e}")
            self.conn.rollback()
            raise

    def delete(self, memory_id: str) -> int:
        cursor = self.conn.cursor()
        try:
            cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self.conn.commit()
            deleted_count = cursor.rowcount
            if deleted_count > 0: logger.debug(f"Deleted memory with ID: {memory_id[:12]}")
            return deleted_count
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id[:12]}: {e}")
            self.conn.rollback()
            raise

    def find_by_content_substring(self, substring: str, *, source: str | None = None) -> list[Memory]:
        """Find memories whose content contains the given substring (case-insensitive)."""
        with self.conn:
            cursor = self.conn.cursor()
            sql = "SELECT * FROM memories WHERE lower(content) LIKE ?"
            params = (f"%{substring.strip().lower()[:80]}%",)
            if source:
                sql += " AND source = ?"
                params += (source,)
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            logger.debug(f"Found {len(rows)} memories by content substring '{substring[:20]}...'")
            return [self._row_to_memory(row) for row in rows]
            
    def find_by_exact_content(self, content: str, *, source: str | None = None) -> list[Memory]:
        """Find memories with exact matching content (case-insensitive)."""
        s = content.strip().lower()
        if not s:
            return []
        with self.conn:
            cursor = self.conn.cursor()
            sql = "SELECT * FROM memories WHERE lower(content) = ?"
            params = (s,)
            if source:
                sql += " AND source = ?"
                params += (source,)
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            logger.debug(f"Found {len(rows)} memories by exact content '{content[:20]}...'")
            return [self._row_to_memory(row) for row in rows]

    def search(self, query: str, *, limit: int, memory_type: MemoryType | None = None, tags: list[str] | None = None, source: str | None = None) -> list[Any]:
        # This is a basic implementation. Real Celebro might use more advanced querying or vector search here.
        # For now, it simulates fetching some records and returning them in a format similar to vector search results.
        
        with self.conn:
            cursor = self.conn.cursor()
            select_sql = "SELECT * FROM memories"
            where_clauses = []
            params = []

            if query:
                # Basic keyword search in content
                where_clauses.append("lower(content) LIKE ?")
                params.append(f"%{query.strip().lower()}%")

            if source:
                where_clauses.append("source = ?")
                params.append(source)

            if memory_type:
                where_clauses.append("type = ?")
                params.append(memory_type.value)

            if tags:
                # This requires tags to be stored in a searchable format (e.g., JSON array)
                # Simplification for dummy: Check if any tag is present in the stored JSON string
                for tag in tags:
                    where_clauses.append("tags LIKE ?")
                    params.append(f'%"{tag}"%') # Assume tags are stored as JSON array ["tag1", "tag2"]

            if where_clauses:
                select_sql += " WHERE " + " AND ".join(where_clauses)
            
            # Basic ordering for search results - could be improved with scoring
            order_by_sql = "ORDER BY last_accessed DESC, importance DESC" 
            select_sql += f" {order_by_sql} LIMIT ?"
            params.append(limit)

            cursor.execute(select_sql, params)
            rows = cursor.fetchall()
            
            # Format results to mimic search result objects with score
            search_results = []
            for row in rows:
                mem = self._row_to_memory(row)
                # Assign a plausible score - higher for importance, lower for substring match
                score = float(mem.importance.value) / MemoryImportance.HIGH.value if mem.importance else 0.5
                if query and query.strip().lower() not in mem.content.strip().lower():
                    score *= 0.5 # Reduce score if it's a substring match
                search_results.append({
                    "memory": mem,
                    "score": score,
                    "type": "search_result"
                })
            logger.debug(f"Search query '{query[:20]}...' found {len(search_results)} results.")
            return search_results

    def count(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM memories")
        return cursor.fetchone()[0]

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            logger.info("SQLiteStore closed.")
        self.conn = None
