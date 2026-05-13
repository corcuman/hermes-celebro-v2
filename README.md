# Celebro v2 — Hermes Persistent Memory Plugin

Celebro v2 is a **Hermes Agent plugin** that implements a local persistent memory provider.

Its purpose is to make Hermes remember durable information across sessions while still keeping the normal Hermes **built-in memory** flow compatible with the rest of the agent.

In short:

- Hermes keeps using its built-in memory interface.
- Celebro v2 stores memory durably in a local database.
- Celebro v2 mirrors built-in memory writes into its own store.
- Celebro v2 can regenerate `MEMORY.md` from its database so the most relevant facts are injected into future sessions.

This plugin is intended for local, private, operator-controlled memory. It is not a cloud memory service.

---

## What this repository contains

This repository contains the Celebro v2 plugin code for Hermes:

- `plugin.yaml` — plugin metadata and hook declarations.
- `__init__.py` — Hermes plugin entrypoint and memory provider implementation.
- `store.py` — SQLite persistence layer.
- `models.py` — memory dataclasses and enums.
- `embeddings.py` — optional embedding integration.
- `retrieval.py` — retrieval helpers.
- `extractor.py` — optional memory extraction helpers.
- `README.md` — this documentation.

Celebro v2 is loaded by Hermes as a plugin and registers itself as a `MemoryProvider`.

---

## Why Celebro v2 exists

Hermes already has a built-in memory system, typically represented by files such as:

- `MEMORY.md` — durable facts, environment notes, project conventions, tool quirks.
- `USER.md` — durable user profile and preferences.

Those files are extremely useful because their content is injected into future turns, so the agent starts each session with important context.

However, plain built-in memory files have limitations:

- They are size-limited because they are injected into the model context.
- They are not optimized for search or semantic retrieval.
- They can accumulate duplicates or stale entries if not managed carefully.
- They are good as an injection layer, but not ideal as the only long-term storage backend.

Celebro v2 adds a proper local memory backend below that built-in layer.

---

## Mental model

Think of Celebro v2 as two layers working together:

```text
User / Agent actions
        │
        ▼
Hermes built-in memory API
(memory add / replace / remove)
        │
        ▼
Celebro v2 plugin hook: on_memory_write
        │
        ▼
Local persistent Celebro database
(SQLite cold store + optional vector index)
        │
        ▼
Regenerated MEMORY.md subset
(top / important / deduplicated memories)
        │
        ▼
Injected into future Hermes sessions
```

The important point: **Celebro v2 does not replace the usefulness of `MEMORY.md`; it backs it with a structured persistent store.**

`MEMORY.md` remains the small, high-value context that gets injected into the model.

Celebro v2 becomes the deeper local memory source that can store more entries, search them, deduplicate them, and regenerate the built-in memory view.

---

## Interaction with Hermes built-in memory

Celebro v2 integrates with Hermes through the `MemoryProvider` interface and plugin hooks.

The plugin declares these hooks in `plugin.yaml`:

```yaml
hooks:
  - on_session_end
  - on_memory_write
```

The most important hook is:

```python
on_memory_write(action, target, content, metadata)
```

Hermes calls this when the built-in memory system writes a memory. Celebro v2 listens to that event and mirrors the change into its own database.

### Add flow

When Hermes adds a memory:

```text
memory.add(target="user" or "memory", content="...")
```

Celebro v2:

1. Receives `on_memory_write(action="add", ...)`.
2. Converts the entry into a Celebro `Memory` object.
3. Stores it in SQLite with source `builtin_mirror`.
4. Tags it with context such as `builtin` and the target (`user` or `memory`).
5. Regenerates `MEMORY.md` from the stored mirror entries.

### Replace flow

When Hermes replaces an existing built-in memory:

1. Celebro v2 tries to locate the existing mirrored memory by ID if available.
2. If no ID is available, it searches by the previous text (`old_text`).
3. If found, it updates the Celebro record while preserving its original creation time.
4. If not found, it safely falls back to adding a new mirrored entry.
5. It regenerates `MEMORY.md` afterwards.

This makes replacements robust even if Hermes only provides textual metadata.

### Remove flow

When Hermes removes a built-in memory:

1. Celebro v2 tries to locate the mirrored memory by ID or previous text.
2. It deletes the SQLite record if found.
3. If a vector entry exists, it attempts to delete that too.
4. It regenerates `MEMORY.md` afterwards.

---

## Built-in memory vs Celebro v2

### Built-in memory

Built-in memory is the visible, compact memory layer that Hermes injects into the prompt.

Use it for facts that must be immediately available in every future session:

- User preferences.
- Stable environment facts.
- Important project conventions.
- Tool quirks and operational constraints.
- Critical reminders that should affect the agent's behavior.

Because it is injected into context, it must stay compact and high-value.

### Celebro v2

Celebro v2 is the persistent backend.

Use it for:

- Longer-term memory storage.
- Searchable memory history.
- Semantic recall when vector search is enabled.
- Deduplication and controlled regeneration of `MEMORY.md`.
- Storing more entries than can reasonably fit in the built-in memory file.

Celebro v2 can then decide which subset of durable memories should be mirrored back into `MEMORY.md`.

---

## Storage

Celebro v2 uses local storage under `$HERMES_HOME` by default.

Default paths:

```text
$HERMES_HOME/celebro_v2/celebro_v2.db
$HERMES_HOME/celebro_v2/chroma
```

Storage components:

- **SQLite cold store** — canonical local memory database.
- **Optional Chroma vector index** — semantic retrieval layer.
- **Optional Ollama embeddings** — local embedding generation, commonly with `nomic-embed-text`.

If Chroma or Ollama is unavailable, Celebro v2 falls back to SQLite recall.

That fallback is intentional: memory should continue working even when the vector stack is unavailable.

---

## Memory model

Celebro v2 stores memory entries with structured metadata.

The core model includes:

- `id` — unique memory identifier.
- `type` — memory category.
- `content` — memory text.
- `importance` — ranking signal.
- `tags` — flexible labels.
- `metadata` — extra structured metadata.
- `created_at` — creation timestamp.
- `last_accessed` — last access timestamp.
- `access_count` — usage counter.
- `session_id` — optional session context.
- `source` — origin of the memory.

Supported memory types:

- `semantic` — facts and user preferences.
- `episodic` — session-like memories or events.
- `procedural` — workflows, conventions, tool-specific knowledge.

Supported importance levels:

- `LOW`
- `MEDIUM`
- `HIGH`
- `CRITICAL`

Built-in mirrored entries currently use source:

```text
builtin_mirror
```

Tool-created entries use source:

```text
tool
```

---

## MEMORY.md regeneration

Celebro v2 can regenerate Hermes' built-in `MEMORY.md` from the Celebro backend.

This is a key design point.

The plugin does not blindly append forever to `MEMORY.md`. Instead it:

1. Reads existing `MEMORY.md` when available.
2. Loads mirrored built-in memories from Celebro (`source="builtin_mirror"`).
3. Deduplicates entries by normalized content keys.
4. Sorts entries by importance and recency.
5. Applies a character limit to keep the injected memory compact.
6. Writes a regenerated `MEMORY.md`.

The current provider has a configurable character budget (`char_limit`) intended to prevent built-in memory from growing without bound.

This means Celebro v2 acts as the durable backend while `MEMORY.md` remains the curated injection layer.

---

## Tools exposed by the plugin

Celebro v2 exposes memory tools for direct interaction:

- `celebro_remember`
- `celebro_search`
- `celebro_forget`
- `celebro_stats`

Typical uses:

### Remember

Store a memory explicitly in Celebro v2.

```text
celebro_remember(content="...", memory_type="semantic", importance=3, tags=["project"])
```

### Search

Search memories by text, optional type, tags, or source.

```text
celebro_search(query="proxmox ssh key", top_k=8)
```

### Forget

Delete a memory by ID.

```text
celebro_forget(memory_id="...")
```

### Stats

Return memory counts and vector index status.

```text
celebro_stats()
```

---

## Configuration

Example Hermes configuration:

```yaml
memory:
  provider: celebro_v2
  celebro_v2:
    data_dir: $HERMES_HOME/celebro_v2
    db_path: $HERMES_HOME/celebro_v2/celebro_v2.db
    vector_db_path: $HERMES_HOME/celebro_v2/chroma
    vector_enabled: "true"
    embedding_model: nomic-embed-text
    ollama_host: http://localhost:11434
    auto_extract: "true"
    top_k: "8"
```

Notes:

- `db_path` controls the SQLite database location.
- `vector_db_path` controls the vector index location.
- `top_k` controls default recall count.
- If vector dependencies are unavailable, SQLite search remains available.

---

## Security and privacy model

Celebro v2 is designed for local memory.

Recommended policy:

- Do **not** store API keys, passwords, tokens, private keys, or secrets as memory entries.
- Do store stable operational facts, paths, workflows, preferences, and non-sensitive conventions.
- Do not commit local database files to Git.
- Do not commit vector indexes to Git.
- Do not commit `__pycache__` or `.pyc` files.

This repository should contain code and documentation only, not user memory data.

Recommended `.gitignore` entries:

```gitignore
__pycache__/
*.pyc
celebro_v2.db
chroma/
```

---

## Operational guidance

Use built-in memory for compact facts the agent must always know.

Use Celebro v2 for deeper local recall and structured persistence.

Good candidates for durable memory:

- User communication preferences.
- Stable infrastructure facts.
- Reusable paths and command conventions.
- Long-lived project architecture notes.
- Tool quirks and known failure modes.

Poor candidates:

- Temporary task progress.
- PR numbers or commit SHAs that will be stale soon.
- One-off session state.
- Secrets or credentials.
- Large raw logs.

If a fact will be stale in a few days, it usually should not become durable memory.

---

## Development workflow

When changing the plugin locally:

1. Edit files under the local Hermes plugin directory.
2. Test in Hermes.
3. Compare with this repository.
4. Copy changed source files into a branch.
5. Open a PR.
6. Merge after review.

Typical files to sync:

```text
__init__.py
store.py
models.py
embeddings.py
retrieval.py
extractor.py
plugin.yaml
README.md
```

Do not sync:

```text
__pycache__/
*.pyc
local database files
local vector index files
```

---

## Current status

Celebro v2 currently provides:

- Hermes `MemoryProvider` registration.
- SQLite-backed persistence.
- Built-in memory mirroring through `on_memory_write`.
- Regeneration of `MEMORY.md` from Celebro mirror entries.
- Optional vector recall path with graceful fallback.
- Direct tools for remember/search/forget/stats.

The plugin's main architectural role is to bridge Hermes' compact built-in memory layer with a richer local persistent memory backend.
