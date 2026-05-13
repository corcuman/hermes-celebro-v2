# Celebro v2 Memory Provider

Local persistent memory provider for Hermes.

## Storage

- SQLite cold store: `$HERMES_HOME/celebro_v2/celebro_v2.db`
- Optional Chroma vector index: `$HERMES_HOME/celebro_v2/chroma`
- Optional Ollama embeddings: `nomic-embed-text`

If Chroma or Ollama is unavailable, the provider falls back to SQLite recall.

## Tools

- `celebro_remember`
- `celebro_search`
- `celebro_forget`
- `celebro_stats`

## Activation

```yaml
memory:
  provider: celebro_v2
  celebro_v2:
    data_dir: $HERMES_HOME/celebro_v2
    vector_enabled: "true"
    embedding_model: nomic-embed-text
    ollama_host: http://localhost:11434
    auto_extract: "true"
    top_k: "8"
```
