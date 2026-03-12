from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

REFERENCES_DIR = ROOT_DIR / "references"
CHROMA_DB_DIR = ROOT_DIR / "chroma_db"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVER_K = 5
LLM_MODEL = "claude-sonnet-4-5"
CHROMA_COLLECTION_NAME = "nlp_textbook"
