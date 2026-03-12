import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import (
    REFERENCES_DIR,
    CHROMA_DB_DIR,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_COLLECTION_NAME,
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


def build_vectorstore(chunks, embeddings) -> Chroma:
    if CHROMA_DB_DIR.exists():
        shutil.rmtree(CHROMA_DB_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(CHROMA_DB_DIR),
    )
    return vectorstore


def load_vectorstore(embeddings) -> Chroma:
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_DIR),
    )


def run_ingestion() -> Chroma:
    print("Loading embedding model...")
    embeddings = get_embeddings()

    print("Loading and splitting documents...")
    chunks = load_and_split_documents()
    print(f"Created {len(chunks)} chunks from {REFERENCES_DIR}")

    print("Building vector store...")
    vectorstore = build_vectorstore(chunks, embeddings)
    print("Ingestion complete.")
    return vectorstore


if __name__ == "__main__":
    run_ingestion()
