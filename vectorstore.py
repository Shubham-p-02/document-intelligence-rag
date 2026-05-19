"""
ChromaDB vector store helpers with cosine-friendly embeddings and retrieval tuning.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from chunking import get_embedding_model


def _persist_dir() -> str:
    return os.environ.get("CHROMA_PERSIST_DIR", ".chroma_index")


def _collection_name() -> str:
    return os.environ.get("COLLECTION_NAME", "documents")


def build_vectorstore(
    chunks: Sequence[Document],
    persist_directory: str | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
    reset: bool = False,
) -> Chroma:
    """
    Create or replace a persisted Chroma store from chunked documents.
    """
    persist = persist_directory or _persist_dir()
    name = collection_name or _collection_name()
    if reset and Path(persist).exists():
        shutil.rmtree(persist, ignore_errors=True)
    embeddings = get_embedding_model(embedding_model)
    return Chroma.from_documents(
        documents=list(chunks),
        embedding=embeddings,
        persist_directory=persist,
        collection_name=name,
    )


def load_vectorstore(
    persist_directory: str | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
) -> Chroma:
    persist = persist_directory or _persist_dir()
    name = collection_name or _collection_name()
    if not Path(persist).exists():
        raise FileNotFoundError(
            f"No index found at '{persist}'. Build the index first from the Streamlit app "
            f"or evaluation script."
        )
    embeddings = get_embedding_model(embedding_model)
    return Chroma(
        persist_directory=persist,
        collection_name=name,
        embedding_function=embeddings,
    )


def retrieve_with_scores(
    store: Chroma,
    query: str,
    top_k: int,
    similarity_threshold: float,
) -> List[Tuple[Document, float]]:
    """
    Return (document, distance) pairs from Chroma similarity search.

    Chroma returns lower distance for closer vectors (L2 on normalized embeddings
    is monotonic with cosine similarity for normalized vectors). Rows with distance
    above `similarity_threshold` are dropped (higher distance = less similar).
    """
    pairs = store.similarity_search_with_score(query, k=max(1, top_k))
    filtered: List[Tuple[Document, float]] = []
    for doc, dist in pairs:
        if dist <= similarity_threshold:
            filtered.append((doc, float(dist)))
    # If threshold is too strict for this corpus/metric, keep top-k rather than returning nothing.
    if not filtered and pairs:
        filtered = [(doc, float(dist)) for doc, dist in pairs]
    return filtered


def distance_to_confidence(distance: float) -> float:
    """
    Map Chroma L2 distance (normalized embeddings) to a heuristic confidence in [0,1].
    This is not a calibrated probability; see README limitations.
    """
    # Heuristic: exp decay — tune similarity_threshold alongside this display.
    import math

    return float(max(0.0, min(1.0, math.exp(-distance))))
