"""
Streamlit UI for document-grounded Q&A with Chroma + Groq.
"""
from __future__ import annotations

import os
from pathlib import Path

# Reduce Chroma telemetry noise in local/Spaces logs
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

import streamlit as st
from dotenv import load_dotenv

from chain import build_rag_chain, get_chat_model
from chunking import semantic_chunk_documents
from ingest import ingest_corpus, ingest_uploaded_files
from vectorstore import build_vectorstore, load_vectorstore

load_dotenv()


def _i18n_err_groq() -> str:
    return (
        "Missing **GROQ_API_KEY**. Add it to a `.env` file (see `.env.example`) or your environment "
        "before chatting."
    )


st.set_page_config(page_title="Document Intelligence RAG", layout="wide")
st.title("Document Intelligence RAG")
st.caption("Semantic chunking · ChromaDB · Groq LLaMA 3.1 · LangChain")

with st.sidebar:
    st.header("Corpus")
    corpus_path = st.text_input(
        "Corpus folder path",
        value=str(Path(__file__).resolve().parent / "sample_corpus"),
        help="Absolute or relative path to a folder containing .txt / .md / .pdf files.",
    )
    uploads = st.file_uploader(
        "Or upload documents",
        type=["txt", "md", "markdown", "pdf"],
        accept_multiple_files=True,
    )
    st.subheader("Indexing")
    reset_index = st.checkbox("Rebuild from scratch", value=True)
    overlap_chars = st.number_input("Chunk overlap (chars)", min_value=0, max_value=2000, value=120, step=10)
    sem_pct = st.slider("Semantic percentile threshold", 80, 99, 95)
    st.subheader("Retrieval")
    top_k = st.number_input("Top-k", min_value=1, max_value=20, value=int(os.environ.get("RAG_TOP_K", "4")))
    max_distance = st.slider(
        "Max retrieval distance (lower is stricter)",
        min_value=0.05,
        max_value=2.0,
        value=float(os.environ.get("RAG_SIMILARITY_THRESHOLD", "1.6")),
        step=0.01,
        help="Chroma returns L2 distance on normalized embeddings; smaller distance means closer match.",
    )
    if st.button("Build / rebuild index", type="primary"):
        with st.spinner("Ingesting and indexing…"):
            if uploads:
                docs = ingest_uploaded_files(uploads)
            else:
                docs = ingest_corpus(corpus_path)
            chunks = semantic_chunk_documents(
                docs,
                breakpoint_threshold_amount=float(sem_pct),
                overlap_chars=int(overlap_chars),
            )
            store = build_vectorstore(chunks, reset=reset_index)
            st.session_state["store"] = store
            st.session_state["vs_path"] = os.environ.get("CHROMA_PERSIST_DIR", ".chroma_index")
            st.success(f"Indexed {len(chunks)} chunks from {len(docs)} document(s).")
    if st.button("Load existing index"):
        try:
            st.session_state["store"] = load_vectorstore()
            st.success("Loaded persisted index.")
        except Exception as exc:  # pragma: no cover - UI path
            st.error(f"Could not load index: {exc}")

st.subheader("Chat")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for role, content in st.session_state["messages"]:
    with st.chat_message(role):
        st.markdown(content)

prompt = st.chat_input("Ask a question about your documents…")
if prompt:
    st.session_state["messages"].append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    if "store" not in st.session_state:
        st.error("Build or load an index first (sidebar).")
    else:
        try:
            get_chat_model()
        except EnvironmentError:
            st.error(_i18n_err_groq())
            st.stop()

        chain = build_rag_chain(
            st.session_state["store"],
            top_k=int(top_k),
            similarity_threshold=float(max_distance),
        )
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = chain.invoke({"question": prompt})
            answer = result.get("answer", "")
            conf = result.get("confidence", 0.0)
            st.markdown(answer)
            st.metric("Retrieval confidence (heuristic)", f"{conf:.3f}")
            st.markdown("**Sources**")
            for src in result.get("sources", []):
                with st.expander(f"[{src['rank']}] {src['source']} · dist={src['distance']:.4f}"):
                    st.json({k: v for k, v in src.items() if k != "excerpt"})
                    st.text_area("Chunk text", src.get("excerpt", ""), height=220, key=f"{src['rank']}-{hash(prompt)}")
        st.session_state["messages"].append(("assistant", answer))
