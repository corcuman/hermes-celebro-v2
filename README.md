<div align="center">

<pre>
  ██████╗███████╗██╗     ███████╗██████╗ ██████╗  ██████╗     ██╗   ██╗██████╗ 
 ██╔════╝██╔════╝██║     ██╔════╝██╔══██╗██╔══██╗██╔═══██╗    ██║   ██║╚════██╗
 ██║     █████╗  ██║     █████╗  ██████╔╝██████╔╝██║   ██║    ██║   ██║ █████╔╝
 ██║     ██╔══╝  ██║     ██╔══╝  ██╔══██╗██╔══██╗██║   ██║    ╚██╗ ██╔╝██╔═══╝ 
 ╚██████╗███████╗███████╗███████╗██████╔╝██║  ██║╚██████╔╝     ╚████╔╝ ███████╗
  ╚═════╝╚══════╝╚══════╝╚══════╝╚═════╝ ╚═╝  ╚═╝ ╚═════╝       ╚═══╝  ╚══════╝
</pre>

**Plugin de memoria persistente local para [Hermes](https://github.com/NousResearch/hermes-agent).**

SQLite · ChromaDB · Ollama · 100% local · Sin APIs externas

---

</div>

## ¿Qué es esto?

Celebro v2 es un plugin para Hermes que reemplaza su memoria nativa (un archivo de texto con límite de tamaño) por una capa de memoria persistente con búsqueda semántica local.

**Problema con la memoria nativa de Hermes:** es un archivo de texto plano que se inyecta completo en el system prompt de cada sesión. Si crece demasiado, ocupa tokens de contexto valiosos o se trunca.

**Solución:** Celebro v2 almacena los recuerdos en una base vectorial local y recupera **solo los más relevantes** para cada sesión por similitud semántica — puedes tener cientos de recuerdos sin penalizar el contexto.

> Para un sistema de memoria standalone (sin Hermes), mira el repo original: [celebro](https://github.com/corcuman/celebro)

---

## Arquitectura

```
GUARDAR recuerdo:
  texto → nomic-embed-text (Ollama) → vector → ChromaDB
                                             → SQLite (metadatos + índice)

RECUPERAR recuerdo:
  query del usuario → nomic-embed-text → vector → búsqueda coseno en ChromaDB
                                               → fallback SQLite (LIKE + filtros)
                                               → top_k más relevantes → inyectados en contexto
```

**Stack:**
- **ChromaDB** — búsqueda vectorial por similitud coseno (HNSW)
- **SQLite** — índice de metadatos, fallback de búsqueda por texto, filtros
- **Ollama** (`nomic-embed-text`) — embeddings 100% locales, sin enviar datos a la nube
- **Hermes MemoryProvider** — integración con el ciclo de vida del agente (prefetch, hooks)

---

## Instalación

### Requisitos

- [Hermes](https://github.com/NousResearch/hermes-agent) instalado
- [Ollama](https://ollama.com/) corriendo localmente con `nomic-embed-text`:
  ```bash
  ollama pull nomic-embed-text
  ```

### Pasos

1. Copia el directorio del plugin a tu carpeta de plugins de Hermes:
   ```bash
   cp -r celebro_v2/ ~/.hermes/plugins/celebro_v2/
   ```

2. Instala las dependencias Python en el entorno de Hermes:
   ```bash
   pip install chromadb ollama
   ```

3. Activa el plugin en `~/.hermes/config.yaml`:
   ```yaml
   memory:
     provider: celebro_v2
     celebro_v2:
       data_dir: $HERMES_HOME/celebro_v2      # Dónde se guardan los datos
       vector_enabled: "true"                  # Habilitar ChromaDB + Ollama
       embedding_model: nomic-embed-text        # Modelo Ollama para embeddings
       ollama_host: http://localhost:11434      # Host de Ollama
       auto_extract: "true"                    # Extraer recuerdos al cierre de sesión
       top_k: "8"                              # Recuerdos a recuperar por sesión
   ```

4. Reinicia Hermes. El plugin se inicializa automáticamente.

---

## Herramientas disponibles

Una vez activo, el agente tiene acceso a estas herramientas MCP:

### `celebro_remember`
Guarda un recuerdo en la memoria persistente.

```python
celebro_remember(
    content="El usuario prefiere respuestas en español",
    memory_type="semantic",   # semantic | episodic | procedural
    importance=3,             # 1 (baja) → 4 (crítica)
    tags=["preferencias", "idioma"]
)
```

### `celebro_search`
Busca recuerdos relevantes con filtros opcionales.

```python
# Búsqueda semántica simple
celebro_search(query="preferencias del usuario")

# Con filtros
celebro_search(
    query="configuración proxmox",
    memory_type="procedural",
    tags=["proxmox", "homelab"],
    source="tool",            # tool | builtin_mirror | session_extract
    top_k=5
)
```

### `celebro_forget`
Elimina un recuerdo por ID.

```python
celebro_forget(memory_id="uuid-del-recuerdo")
```

### `celebro_stats`
Estado del sistema de memoria.

```python
celebro_stats()
# → {"memories": 21, "vector_indexed": 21, "read_only": false}
```

---

## Tipos de memoria

| Tipo | Uso |
|------|-----|
| `semantic` | Hechos y preferencias del usuario |
| `episodic` | Eventos pasados y decisiones tomadas |
| `procedural` | Flujos, protocolos y comandos establecidos |

## Niveles de importancia

| Nivel | Valor | Uso |
|-------|-------|-----|
| Low | 1 | Contexto general, trivial |
| Medium | 2 | Información útil pero no crítica |
| High | 3 | Preferencias importantes, convenciones |
| Critical | 4 | Credenciales, rutas clave, flujos críticos |

---

## Comportamiento automático

- **Prefetch por sesión:** al inicio de cada mensaje, Celebro recupera automáticamente los recuerdos más similares a la query del usuario e los inyecta en el contexto bajo `## Celebro v2 Memory`.
- **Extracción automática:** al cierre de sesión (`on_session_end`), el plugin analiza la conversación y extrae recuerdos duraderos automáticamente.
- **Espejo de memoria nativa:** cuando el agente guarda algo en la memoria nativa de Hermes, Celebro lo replica también en su base de datos (`on_memory_write`).
- **Fallback robusto:** si ChromaDB u Ollama no están disponibles, el sistema cae automáticamente a búsqueda SQLite por texto.

---

## Estructura de archivos

```
celebro_v2/
├── __init__.py      # Provider principal, herramientas MCP, integración Hermes
├── store.py         # SQLite: almacenamiento, búsqueda con filtros, fallback
├── retrieval.py     # ChromaDB: índice vectorial, búsqueda semántica
├── embeddings.py    # OllamaEmbedder: generación de vectores local
├── extractor.py     # Extracción automática de recuerdos al cierre de sesión
├── models.py        # Dataclasses: Memory, SearchResult, MemoryType, etc.
└── plugin.yaml      # Manifiesto del plugin para Hermes
```

Los datos (DB, vectores) se almacenan en `$HERMES_HOME/celebro_v2/` y **no forman parte del repositorio**.

---

## Diferencia con Celebro v1

| | [Celebro v1](https://github.com/corcuman/celebro) | Celebro v2 |
|---|---|---|
| Uso | Script standalone, cualquier agente | Plugin nativo de Hermes |
| Vector backend | Qdrant | ChromaDB |
| Integración | Manual vía CLI | Automática (hooks, prefetch) |
| Filtros búsqueda | source, tags (Qdrant) | type, importance, tags, source (SQLite + Chroma) |
| Dependencias Hermes | Ninguna | `agent.memory_provider`, `tools.registry` |

---

## Licencia

MIT
