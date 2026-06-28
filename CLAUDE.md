# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Ingest PDFs into the vector store (run from project root)
uv run python -m src.ingest

# Run the Streamlit app
uv run streamlit run src/app.py
```

## Architecture

This is a RAG pipeline where ingestion and retrieval are decoupled, and the
vector store is pluggable via `VECTOR_STORE_BACKEND` (default `weaviate`, or
`chroma`). `src/ingest.py` dispatches to the selected backend behind a common
API (`build_vectorstore` / `load_vectorstore` / `build_retriever`), each
returning `(vectorstore, client_or_None)` — the client is non-`None` only for
Weaviate.

**Ingestion** (`src/ingest.py`): PDFs from `references/` are loaded with `PyPDFDirectoryLoader`, split into chunks, and embedded with HuggingFace `all-MiniLM-L6-v2`. Running ingestion always rebuilds the store from scratch.
- **Weaviate** (default): runs an embedded server persisting to `weaviate_db/`. Rebuild drops/recreates the collection via the client API (`collections.delete` + `from_documents`) rather than deleting the data dir, since the embedded server may be holding it open. PDF metadata keys are sanitized (`_sanitize_metadata_keys`) to valid GraphQL property names.
- **Chroma**: persists to `chroma_db/`; rebuild wipes the directory with `shutil.rmtree`.

**Embedded Weaviate lifecycle** (`src/ingest.py`): `connect_to_embedded()` spawns a server holding ports 8079/50050 that outlives its Python parent. The client that starts it is tracked as the *owner* (`is_embedded_owner`); an `atexit` handler (plus best-effort SIGTERM/SIGINT handlers when on the main thread) closes it exactly once at process exit, stopping the server and freeing the ports. If the ports are already bound, `_get_weaviate_client` falls back to `connect_to_local` and that client is **not** an owner, so it is never torn down. Per-session / re-ingest cleanup skips the owner, so an ending session or a rebuild can't stop a server other tabs are using.

**RAG chain** (`src/rag_chain.py`): At query time the retriever fetches chunks — Weaviate uses hybrid search (BM25 + vector, weighted by `HYBRID_ALPHA`); Chroma uses MMR — injects them as context into a prompt, and passes it to Claude (`claude-sonnet-4-5`) via `langchain-anthropic`. The chain is built with LCEL and supports multi-turn chat history via `MessagesPlaceholder`.

**App** (`src/app.py`): Streamlit UI that holds the embedding model, retriever, chain, and (for Weaviate) the client in `st.session_state` to avoid reloading on every interaction. On startup it auto-loads the existing store if present (checks `WEAVIATE_DATA_DIR` or `CHROMA_DB_DIR` depending on the backend). `load_dotenv` uses an explicit absolute path (`Path(__file__).parent.parent / ".env"`) because Streamlit may not run from the project root.

**Config** (`src/config.py`): Single source of truth for all paths and constants (chunk size, model names, collection names, `VECTOR_STORE_BACKEND`, `HYBRID_ALPHA`, etc.).

## Known constraints

- A hard `kill -9` of the app (before `atexit` runs) orphans the embedded Weaviate server, leaving ports 8079/50050 held. The next start reuses the orphan via `connect_to_local`; to fully clear it, `kill -9 $(pgrep -f weaviate-embedded/weaviate)`.
- `transformers` is pinned to `>=4.40.0,<5.0.0` — versions 5.x crash on Python 3.13 due to a bug in their import machinery.
- `src/app.py` uses a `sys.path.insert` hack to make the `src` package importable when launched via `streamlit run` (which doesn't set `PYTHONPATH`). A cleaner alternative is to configure a `[build-system]` in `pyproject.toml` and install the project in editable mode.
