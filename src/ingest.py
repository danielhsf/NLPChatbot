import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import (
    REFERENCES_DIR,
    CHROMA_DB_DIR,
    WEAVIATE_DATA_DIR,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_COLLECTION_NAME,
    WEAVIATE_COLLECTION_NAME,
    VECTOR_STORE_BACKEND,
    RETRIEVER_K,
    HYBRID_ALPHA,
)

load_dotenv()


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def load_and_split_documents():
    if not REFERENCES_DIR.exists() or not list(REFERENCES_DIR.glob("*.pdf")):
        raise FileNotFoundError(
            f"No PDF files found in {REFERENCES_DIR}. "
            "Please add PDF files before running ingestion."
        )

    loader = PyPDFDirectoryLoader(str(REFERENCES_DIR))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    return chunks


# --- Chroma helpers ---

def _build_chroma(chunks, embeddings) -> Chroma:
    if CHROMA_DB_DIR.exists():
        shutil.rmtree(CHROMA_DB_DIR)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(CHROMA_DB_DIR),
    )


def _load_chroma(embeddings) -> Chroma:
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_DIR),
    )


# --- Weaviate helpers ---

def _get_weaviate_client():
    import weaviate
    WEAVIATE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return weaviate.connect_to_embedded(
        persistence_data_path=str(WEAVIATE_DATA_DIR)
    )


def _build_weaviate(chunks, embeddings):
    from langchain_weaviate.vectorstores import WeaviateVectorStore
    if WEAVIATE_DATA_DIR.exists():
        shutil.rmtree(WEAVIATE_DATA_DIR)
    client = _get_weaviate_client()
    try:
        vs = WeaviateVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=client,
            index_name=WEAVIATE_COLLECTION_NAME,
            text_key="text",
        )
    except Exception:
        client.close()
        raise
    return vs, client


def _load_weaviate(embeddings):
    from langchain_weaviate.vectorstores import WeaviateVectorStore
    client = _get_weaviate_client()
    try:
        vs = WeaviateVectorStore(
            client=client,
            index_name=WEAVIATE_COLLECTION_NAME,
            text_key="text",
            embedding=embeddings,
        )
    except Exception:
        client.close()
        raise
    return vs, client


# --- Public API ---

def build_vectorstore(chunks, embeddings):
    """Returns (vectorstore, client_or_None)."""
    if VECTOR_STORE_BACKEND == "weaviate":
        return _build_weaviate(chunks, embeddings)
    if VECTOR_STORE_BACKEND == "chroma":
        return _build_chroma(chunks, embeddings), None
    raise ValueError(f"Unknown VECTOR_STORE_BACKEND: {VECTOR_STORE_BACKEND!r}")


def load_vectorstore(embeddings):
    """Returns (vectorstore, client_or_None)."""
    if VECTOR_STORE_BACKEND == "weaviate":
        return _load_weaviate(embeddings)
    if VECTOR_STORE_BACKEND == "chroma":
        return _load_chroma(embeddings), None
    raise ValueError(f"Unknown VECTOR_STORE_BACKEND: {VECTOR_STORE_BACKEND!r}")


def build_retriever(vectorstore):
    if VECTOR_STORE_BACKEND == "weaviate":
        return vectorstore.as_retriever(
            search_type="hybrid",
            search_kwargs={"k": RETRIEVER_K, "alpha": HYBRID_ALPHA},
        )
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVER_K, "fetch_k": RETRIEVER_K * 3},
    )


def run_ingestion():
    print("Loading embedding model...")
    embeddings = get_embeddings()

    print("Loading and splitting documents...")
    chunks = load_and_split_documents()
    print(f"Created {len(chunks)} chunks from {REFERENCES_DIR}")

    print("Building vector store...")
    vectorstore, client = build_vectorstore(chunks, embeddings)
    print("Ingestion complete.")
    return vectorstore, client


if __name__ == "__main__":
    run_ingestion()
