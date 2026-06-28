# NLP Chatbot

A RAG (Retrieval-Augmented Generation) chatbot for answering questions about Natural Language Processing, grounded in *Speech and Language Processing* (Jurafsky & Martin, 3rd ed.).

## Stack

- **LangChain** (LCEL) — chain orchestration
- **Weaviate** (embedded) — default vector store, with hybrid BM25 + vector search. **Chroma** is an optional alternative backend.
- **HuggingFace `all-MiniLM-L6-v2`** — local embeddings
- **Claude (`claude-sonnet-4-5`)** via `langchain-anthropic` — LLM
- **Streamlit** — chat UI

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

Copy the example and add your Anthropic API key:

```bash
cp .env.example .env
```

```env
ANTHROPIC_API_KEY=your_api_key_here
```

Get your API key at [console.anthropic.com/settings/api-keys](https://console.anthropic.com/settings/api-keys).

#### Vector store backend

Controlled by `VECTOR_STORE_BACKEND` in `.env`:

- `weaviate` (default) — embedded Weaviate with hybrid search (BM25 + vector).
  The weighting is set by `HYBRID_ALPHA` (0 = pure keyword, 1 = pure vector;
  default `0.75`). The embedded server is started automatically and torn down
  when the app process exits.
- `chroma` — persistent local Chroma store with MMR retrieval.

### 3. Add PDFs

Place your Jurafsky textbook PDF(s) in the `references/` directory.

### 4. Ingest documents

```bash
uv run python -m src.ingest
```

### 5. Run the app

```bash
uv run streamlit run src/app.py
```

## Usage

- Ask NLP questions in the chat input
- Answers cite the source file and page number
- Use **Re-ingest Documents** in the sidebar to update the vector store after adding new PDFs
- Use **Clear conversation** to reset chat history
