import atexit
import re
import shutil
import signal
import threading
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

_INVALID_PROP_CHARS = re.compile(r"[^_0-9A-Za-z]")


def _sanitize_metadata_keys(chunks):
    """Weaviate property names must match /[_A-Za-z][_0-9A-Za-z]{0,230}/.

    PDF metadata often carries keys with spaces, dots, hyphens or colons
    (e.g. 'Content-Type'), which Weaviate rejects. Rewrite each key to a
    valid GraphQL name, dropping any that become empty or collide.
    """
    for chunk in chunks:
        clean = {}
        for key, value in chunk.metadata.items():
            new_key = _INVALID_PROP_CHARS.sub("_", key)
            if new_key and new_key[0].isdigit():
                new_key = f"_{new_key}"
            if not new_key or new_key in clean:
                continue
            clean[new_key] = value
        chunk.metadata = clean
    return chunks


# Default ports used by weaviate.connect_to_embedded().
_WEAVIATE_HTTP_PORT = 8079
_WEAVIATE_GRPC_PORT = 50050

# Teardown of the embedded server is owned by the single process that
# started it, and runs once at process exit — never per Streamlit session
# and never on re-ingest — so an ending session or rebuild can't stop a
# server other tabs are still using.
_embedded_lock = threading.Lock()
_embedded_owner_client = None  # the client that started the embedded server
_cleanup_registered = False


def _close_embedded_owner():
    global _embedded_owner_client
    with _embedded_lock:
        client = _embedded_owner_client
        _embedded_owner_client = None
    if client is not None:
        try:
            client.close()  # stops the embedded server and frees the ports
        except Exception:
            pass


def _register_embedded_cleanup():
    """Guarantee the embedded server is stopped when this process exits."""
    global _cleanup_registered
    if _cleanup_registered:
        return
    _cleanup_registered = True

    atexit.register(_close_embedded_owner)

    # Signal handlers can only be installed from the main thread; under
    # Streamlit's script runner this runs on a worker thread, so the
    # signal.signal() calls below raise ValueError and are skipped. That's
    # fine — atexit still covers normal exit and Ctrl-C, which Streamlit
    # turns into a graceful shutdown.
    def _handle_signal(signum, frame, previous):
        _close_embedded_owner()
        if callable(previous) and previous not in (signal.SIG_DFL, signal.SIG_IGN):
            previous(signum, frame)
        else:
            raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            prev = signal.getsignal(sig)
            signal.signal(sig, lambda s, f, p=prev: _handle_signal(s, f, p))
        except (ValueError, OSError):
            pass


def is_embedded_owner(client) -> bool:
    """True if `client` is the one that started the embedded server here."""
    with _embedded_lock:
        return client is not None and client is _embedded_owner_client


def _get_weaviate_client():
    global _embedded_owner_client
    import weaviate
    from weaviate.exceptions import WeaviateStartUpError
    WEAVIATE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        client = weaviate.connect_to_embedded(
            persistence_data_path=str(WEAVIATE_DATA_DIR)
        )
    except WeaviateStartUpError:
        # An embedded instance is already listening on the default ports
        # (a prior rerun/session in this process, or a leftover orphan from
        # a hard kill). Reuse it — but we are not its owner, so we must not
        # tear it down.
        return weaviate.connect_to_local(
            port=_WEAVIATE_HTTP_PORT,
            grpc_port=_WEAVIATE_GRPC_PORT,
        )
    # connect_to_embedded started the server in this process: we own its
    # teardown, which happens exactly once at process exit.
    with _embedded_lock:
        _embedded_owner_client = client
    _register_embedded_cleanup()
    return client


def _build_weaviate(chunks, embeddings):
    from langchain_weaviate.vectorstores import WeaviateVectorStore
    chunks = _sanitize_metadata_keys(chunks)
    client = _get_weaviate_client()
    try:
        # Rebuild from scratch (like Chroma). Drop the collection via the
        # API rather than deleting WEAVIATE_DATA_DIR: the embedded server
        # may already be running and holding that directory open, and on
        # re-ingest we deliberately keep that shared server up.
        if client.collections.exists(WEAVIATE_COLLECTION_NAME):
            client.collections.delete(WEAVIATE_COLLECTION_NAME)
        vs = WeaviateVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=client,
            index_name=WEAVIATE_COLLECTION_NAME,
            text_key="text",
        )
    except Exception:
        # Only close ad-hoc connections; never the embedded owner, whose
        # lifecycle is managed at process exit.
        if not is_embedded_owner(client):
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
        if not is_embedded_owner(client):
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
        # langchain-weaviate's similarity_search is already a hybrid
        # (BM25 + vector) query under the hood; there is no "hybrid"
        # search_type. The hybrid weighting is controlled by `alpha`,
        # which is forwarded to weaviate's query.hybrid().
        return vectorstore.as_retriever(
            search_type="similarity",
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
