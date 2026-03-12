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

This is a RAG pipeline where ingestion and retrieval are decoupled:

**Ingestion** (`src/ingest.py`): PDFs from `references/` are loaded with `PyPDFDirectoryLoader`, split into chunks, embedded with HuggingFace `all-MiniLM-L6-v2`, and persisted to a Chroma vector store at `chroma_db/`. Running ingestion always wipes and rebuilds the store from scratch.

**RAG chain** (`src/rag_chain.py`): At query time, an MMR retriever fetches chunks from Chroma, injects them as context into a prompt, and passes it to Claude (`claude-sonnet-4-5`) via `langchain-anthropic`. The chain is built with LCEL and supports multi-turn chat history via `MessagesPlaceholder`.

**App** (`src/app.py`): Streamlit UI that holds the embedding model, retriever, and chain in `st.session_state` to avoid reloading on every interaction. On startup it auto-loads the existing Chroma store if present. `load_dotenv` uses an explicit absolute path (`Path(__file__).parent.parent / ".env"`) because Streamlit may not run from the project root.

**Config** (`src/config.py`): Single source of truth for all paths and constants (chunk size, model names, collection name, etc.).

## Known constraints

- `transformers` is pinned to `>=4.40.0,<5.0.0` — versions 5.x crash on Python 3.13 due to a bug in their import machinery.
- `src/app.py` uses a `sys.path.insert` hack to make the `src` package importable when launched via `streamlit run` (which doesn't set `PYTHONPATH`). A cleaner alternative is to configure a `[build-system]` in `pyproject.toml` and install the project in editable mode.
