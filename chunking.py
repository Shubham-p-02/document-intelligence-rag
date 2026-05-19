"""
Semantic chunking using LangChain Experimental `SemanticChunker` backed by
sentence-transformers via `HuggingFaceEmbeddings`.

Optional character overlap merges context across chunk boundaries (post-split),
since SemanticChunker itself does not expose a classic "chunk_overlap" slider.
"""
from __future__ import annotations

import os
from typing import List, Sequence

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding_model(model_name: str | None = None) -> HuggingFaceEmbeddings:
    name = model_name or os.environ.get(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    return HuggingFaceEmbeddings(
        model_name=name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _apply_char_overlap(chunks: Sequence[Document], overlap_chars: int) -> List[Document]:
    if overlap_chars <= 0:
        return list(chunks)
    out: List[Document] = []
    prev_tail = ""
    for i, doc in enumerate(chunks):
        text = doc.page_content
        if prev_tail:
            merged = (prev_tail + "\n" + text).strip()
        else:
            merged = text
        meta = dict(doc.metadata)
        meta["overlap_chars"] = overlap_chars
        meta["chunk_index"] = i
        out.append(Document(page_content=merged, metadata=meta))
        prev_tail = text[-overlap_chars:] if len(text) >= overlap_chars else text
    return out


def semantic_chunk_documents(
    documents: Sequence[Document],
    embedding_model: str | None = None,
    breakpoint_threshold_type: str | None = None,
    breakpoint_threshold_amount: float | None = None,
    overlap_chars: int | None = None,
) -> List[Document]:
    """
    Split documents using embedding-aware semantic boundaries, then optionally
    prepend the tail of the previous chunk for soft overlap (cross-boundary recall).
    """
    embedder = get_embedding_model(embedding_model)
    btype = breakpoint_threshold_type or os.environ.get(
        "SEMANTIC_BREAKPOINT_THRESHOLD_TYPE", "percentile"
    )
    bamount = breakpoint_threshold_amount
    if bamount is None:
        bamount = float(os.environ.get("SEMANTIC_BREAKPOINT_AMOUNT_THRESHOLD", "95"))

    splitter = SemanticChunker(
        embedder,
        breakpoint_threshold_type=btype,
        breakpoint_threshold_amount=bamount,
    )
    flat: List[Document] = []
    for doc in documents:
        for chunk in splitter.split_documents([doc]):
            meta = dict(doc.metadata)
            meta.update(chunk.metadata)
            flat.append(Document(page_content=chunk.page_content, metadata=meta))

    oc = overlap_chars
    if oc is None:
        oc = int(os.environ.get("CHUNK_OVERLAP_CHARS", "120"))
    return _apply_char_overlap(flat, max(0, oc))
