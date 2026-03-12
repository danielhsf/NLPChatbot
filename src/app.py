import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from src.config import CHROMA_DB_DIR
from src.ingest import get_embeddings, run_ingestion, load_vectorstore
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


def load_embeddings_once():
    if st.session_state.embeddings is None:
        with st.spinner("Loading embedding model..."):
            st.session_state.embeddings = get_embeddings()


def auto_load_vectorstore():
    if st.session_state.rag_chain is None and CHROMA_DB_DIR.exists():
        with st.spinner("Loading existing vector store..."):
            vectorstore = load_vectorstore(st.session_state.embeddings)
            st.session_state.rag_chain, st.session_state.retriever = build_rag_chain(vectorstore)


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
                vectorstore = run_ingestion()
                st.session_state.rag_chain, st.session_state.retriever = build_rag_chain(vectorstore)
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
