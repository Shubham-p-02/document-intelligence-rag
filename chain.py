"""
LangChain LCEL RAG chain: retrieval + Groq chat + citation-friendly prompt.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langchain_chroma import Chroma

from vectorstore import distance_to_confidence, retrieve_with_scores


class RagState(TypedDict, total=False):
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float


def _require_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your Groq API key, "
            "or export GROQ_API_KEY in your shell. Keys are available at https://console.groq.com/"
        )
    return key


def get_chat_model(model: str | None = None, temperature: float = 0.2) -> ChatGroq:
    _require_groq_key()
    name = model or os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    return ChatGroq(
        model=name,
        temperature=temperature,
        api_key=os.environ["GROQ_API_KEY"],
    )


def build_rag_chain(
    store: Chroma,
    top_k: int = 4,
    similarity_threshold: float = 1.6,
    llm: ChatGroq | None = None,
):
    """
    Build a Runnable chain (LCEL: prompt | llm | parser inside retrieval step) that
    returns a dict with answer, sources, and confidence.

    Confidence uses the best transformed retrieval score among kept chunks
    (see vectorstore.distance_to_confidence); not a calibrated probability.
    """
    llm = llm or get_chat_model()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful assistant for document-grounded Q&A. "
                "Answer using ONLY the provided context. If the context is insufficient, say so. "
                "You may cite sources as [1], [2] matching the numbered context blocks.",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )
    generator = prompt | llm | StrOutputParser()

    def rag_step(inputs: Dict[str, Any]) -> RagState:
        q = inputs["question"]
        pairs = retrieve_with_scores(store, q, top_k=top_k, similarity_threshold=similarity_threshold)
        context_blocks: List[str] = []
        sources: List[Dict[str, Any]] = []
        for i, (doc, dist) in enumerate(pairs, start=1):
            src = doc.metadata.get("source", "unknown")
            block = f"[{i}] (source: {src})\n{doc.page_content}"
            context_blocks.append(block)
            sources.append(
                {
                    "rank": i,
                    "source": src,
                    "distance": dist,
                    "chunk_confidence": distance_to_confidence(dist),
                    "excerpt": doc.page_content,
                    "metadata": doc.metadata,
                }
            )
        context = "\n\n".join(context_blocks) if context_blocks else "(no relevant context retrieved)"
        conf = max((s["chunk_confidence"] for s in sources), default=0.0)
        answer = generator.invoke({"question": q, "context": context})
        return {
            "question": q,
            "answer": answer,
            "sources": sources,
            "confidence": float(conf),
        }

    return RunnableLambda(rag_step)


def invoke_rag(chain, question: str) -> RagState:
    return chain.invoke({"question": question})
