import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from src.config import CHROMA_DB_DIR, WEAVIATE_DATA_DIR, VECTOR_STORE_BACKEND
from src.ingest import (
    get_embeddings,
    run_ingestion,
    load_vectorstore,
    build_retriever,
    is_embedded_owner,
)
from src.rag_chain import build_rag_chain

load_dotenv(Path(__file__).parent.parent / ".env")

st.set_page_config(page_title="NLP Chatbot", page_icon="📚", layout="wide")
st.title("📚 NLP Textbook Chatbot")
st.caption("Ask questions about Speech and Language Processing (Jurafsky & Martin)")


def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "weaviate_client" not in st.session_state:
        st.session_state.weaviate_client = None


def _close_weaviate_client():
    # Close only ad-hoc connections (e.g. a previous re-ingest's
    # connect_to_local). The client that owns the embedded server is never
    # closed here: its teardown is owned by the process and runs at exit, so
    # ending a session or re-ingesting can't stop a server other tabs use.
    client = st.session_state.get("weaviate_client")
    if client is not None and not is_embedded_owner(client):
        try:
            client.close()
        except Exception:
            pass
    st.session_state.weaviate_client = None


def load_embeddings_once():
    if st.session_state.embeddings is None:
        with st.spinner("Loading embedding model..."):
            st.session_state.embeddings = get_embeddings()


def auto_load_vectorstore():
    if st.session_state.rag_chain is not None:
        return
    store_exists = (
        WEAVIATE_DATA_DIR.exists() if VECTOR_STORE_BACKEND == "weaviate"
        else CHROMA_DB_DIR.exists()
    )
    if store_exists:
        with st.spinner("Loading existing vector store..."):
            vectorstore, client = load_vectorstore(st.session_state.embeddings)
            st.session_state.weaviate_client = client
            retriever = build_retriever(vectorstore)
            st.session_state.rag_chain, st.session_state.retriever = build_rag_chain(retriever)


init_session_state()
load_embeddings_once()
auto_load_vectorstore()

# Sidebar
with st.sidebar:
    st.header("Controls")

    if st.session_state.rag_chain is not None:
        st.success("Vector store loaded")
        ingest_label = "Re-ingest Documents"
    else:
        st.warning("Vector store not loaded")
        ingest_label = "Ingest Documents"

    if st.button(ingest_label, use_container_width=True):
        with st.spinner("Ingesting documents..."):
            try:
                _close_weaviate_client()
                vectorstore, client = run_ingestion()
                st.session_state.weaviate_client = client
                retriever = build_retriever(vectorstore)
                st.session_state.rag_chain, st.session_state.retriever = build_rag_chain(retriever)
                st.success("Ingestion complete!")
                st.rerun()
            except FileNotFoundError as e:
                st.error(str(e))

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Chat history display
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Chat input
question = st.chat_input("Ask an NLP question...")

if question:
    if st.session_state.rag_chain is None:
        st.error("Please ingest documents first using the sidebar button.")
    else:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                source_docs = st.session_state.retriever.invoke(question)
                answer = st.session_state.rag_chain.invoke({
                    "question": question,
                    "chat_history": st.session_state.chat_history,
                })

            st.markdown(answer)

            if source_docs:
                with st.expander(f"Sources ({len(source_docs)} chunks)"):
                    for doc in source_docs:
                        source = doc.metadata.get("source", "unknown")
                        page = doc.metadata.get("page", "?")
                        filename = os.path.basename(source)
                        st.markdown(f"**{filename}** — Page {int(page) + 1 if isinstance(page, (int, float)) else page}")
                        st.text(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))
                        st.divider()

        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(content=answer))
