import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent

REFERENCES_DIR = ROOT_DIR / "references"
CHROMA_DB_DIR = ROOT_DIR / "chroma_db"
WEAVIATE_DATA_DIR = ROOT_DIR / "weaviate_db"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVER_K = 5
LLM_MODEL = "claude-sonnet-4-5"

CHROMA_COLLECTION_NAME = "nlp_textbook"
WEAVIATE_COLLECTION_NAME = "NlpTextbook"  # must start with uppercase letter

VECTOR_STORE_BACKEND: str = os.environ.get(
    "VECTOR_STORE_BACKEND", "weaviate").lower()
HYBRID_ALPHA: float = float(os.environ.get("HYBRID_ALPHA", "0.75"))
