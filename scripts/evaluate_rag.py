#!/usr/bin/env python3
"""
Run RAGAS evaluation on 200+ synthetic Q&A pairs derived from indexed chunks,
with optional merge of a small golden JSON file.

Usage:
  export GROQ_API_KEY=...
  python scripts/evaluate_rag.py --corpus sample_corpus --target-queries 220

Requires network access for Groq during evaluation (judge + answer generation).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

from chain import build_rag_chain, get_chat_model  # noqa: E402
from chunking import semantic_chunk_documents  # noqa: E402
from ingest import ingest_corpus  # noqa: E402
from vectorstore import build_vectorstore, load_vectorstore  # noqa: E402


def _load_golden(path: Path | None) -> List[Dict[str, str]]:
    if not path:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in data:
        rows.append(
            {
                "question": item["question"],
                "ground_truth": item.get("ground_truth") or item.get("reference") or "",
            }
        )
    return rows


def _snippet(text: str, limit: int = 200) -> str:
    t = " ".join(text.split())
    return t if len(t) <= limit else t[:limit] + "…"


def build_synthetic_rows(chunks: Sequence[Any], target: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    if not chunks:
        raise ValueError("No chunks available to synthesize evaluation questions.")
    templates = [
        "According to the indexed documents, what is discussed in the passage beginning: {snippet}?",
        "What factual details are stated in the material that includes: {snippet}?",
        "Summarize the main points implied by the excerpt that starts with: {snippet}.",
        "Which topics appear in the corpus segment: {snippet}?",
        "Answer precisely using only the corpus: what does the text say near: {snippet}?",
    ]
    rows: List[Dict[str, str]] = []
    idx_perm = list(range(len(chunks)))
    rng.shuffle(idx_perm)
    i = 0
    while len(rows) < target:
        chunk = chunks[idx_perm[i % len(idx_perm)]]
        tpl = templates[i % len(templates)]
        snippet = _snippet(chunk.page_content)
        q = tpl.format(snippet=snippet)
        rows.append({"question": q, "ground_truth": chunk.page_content[:6000]})
        i += 1
    return rows[:target]


def run_rag_batch(chain, questions: Sequence[str], throttle_s: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for q in questions:
        res = chain.invoke({"question": q})
        contexts = [s.get("excerpt", "") for s in res.get("sources", [])]
        if not contexts:
            contexts = ["(no retrieval)"]
        out.append(
            {
                "question": q,
                "answer": res.get("answer", ""),
                "contexts": contexts,
                "confidence": res.get("confidence", 0.0),
            }
        )
        if throttle_s > 0:
            time.sleep(throttle_s)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS evaluation for Document Intelligence RAG")
    parser.add_argument("--corpus", type=str, default=str(ROOT / "sample_corpus"))
    parser.add_argument("--golden", type=str, default=str(ROOT / "eval" / "golden_example.json"))
    parser.add_argument("--target-queries", type=int, default=220)
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("RAG_TOP_K", "4")))
    parser.add_argument(
        "--max-distance",
        type=float,
        default=float(os.environ.get("RAG_SIMILARITY_THRESHOLD", "1.6")),
    )
    parser.add_argument("--overlap-chars", type=int, default=int(os.environ.get("CHUNK_OVERLAP_CHARS", "120")))
    parser.add_argument(
        "--semantic-percentile",
        type=float,
        default=float(os.environ.get("SEMANTIC_BREAKPOINT_AMOUNT_THRESHOLD", "95")),
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--throttle-seconds", type=float, default=0.05, help="Sleep between Groq calls.")
    parser.add_argument("--persist-dir", type=str, default=os.environ.get("CHROMA_PERSIST_DIR", ".chroma_eval"))
    parser.add_argument("--output", type=str, default=str(ROOT / "eval" / "ragas_last_run.json"))
    parser.add_argument("--skip-ragas", action="store_true", help="Only build data + run RAG; skip judge metrics.")
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Build index and write synthetic eval rows only (no Groq / RAGAS).",
    )
    args = parser.parse_args()

    os.environ["CHROMA_PERSIST_DIR"] = args.persist_dir

    docs = ingest_corpus(args.corpus)
    chunks = semantic_chunk_documents(
        docs,
        overlap_chars=args.overlap_chars,
        breakpoint_threshold_amount=args.semantic_percentile,
    )
    build_vectorstore(chunks, reset=True, persist_directory=args.persist_dir)

    golden_path = Path(args.golden) if args.golden else None
    golden_rows = _load_golden(golden_path) if golden_path and golden_path.exists() else []

    synth_target = max(0, args.target_queries - len(golden_rows))
    synth_rows = build_synthetic_rows(chunks, synth_target, args.seed) if synth_target > 0 else []
    eval_rows = golden_rows + synth_rows
    if len(eval_rows) < args.target_queries:
        base = list(eval_rows)
        rng = random.Random(args.seed)
        while len(eval_rows) < args.target_queries:
            row = rng.choice(base)
            eval_rows.append(
                {
                    "question": row["question"] + f" (repeat #{len(eval_rows)})",
                    "ground_truth": row["ground_truth"],
                }
            )

    if args.index_only:
        out_path = Path(args.output)
        payload = {
            "meta": {
                "corpus": args.corpus,
                "target_queries": args.target_queries,
                "eval_rows": len(eval_rows),
                "chunks_indexed": len(chunks),
                "mode": "index-only",
            },
            "eval_rows_sample": eval_rows[:5],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Indexed {len(chunks)} chunks; prepared {len(eval_rows)} eval questions.")
        print(f"Wrote summary to {out_path}")
        return

    try:
        get_chat_model()
    except EnvironmentError as exc:
        raise SystemExit(str(exc)) from exc

    store = load_vectorstore(persist_directory=args.persist_dir)
    chain = build_rag_chain(store, top_k=args.top_k, similarity_threshold=args.max_distance)

    questions = [r["question"] for r in eval_rows]
    preds = run_rag_batch(chain, questions, throttle_s=args.throttle_seconds)

    payload: Dict[str, Any] = {
        "meta": {
            "corpus": args.corpus,
            "target_queries": args.target_queries,
            "top_k": args.top_k,
            "max_distance": args.max_distance,
            "overlap_chars": args.overlap_chars,
            "semantic_percentile": args.semantic_percentile,
        },
        "predictions": preds,
    }

    if args.skip_ragas:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote predictions to {args.output} (RAGAS skipped).")
        return

    from datasets import Dataset

    try:
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError as exc:
        raise SystemExit(
            "RAGAS import failed. Ensure `pip install -r requirements.txt` succeeded. "
            f"Original error: {exc}"
        ) from exc

    from langchain_community.embeddings import HuggingFaceEmbeddings

    judge_llm = LangchainLLMWrapper(get_chat_model(temperature=0.0))
    base_emb = HuggingFaceEmbeddings(
        model_name=os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    judge_emb = LangchainEmbeddingsWrapper(base_emb)

    ds = Dataset.from_dict(
        {
            "question": [p["question"] for p in preds],
            "answer": [p["answer"] for p in preds],
            "contexts": [p["contexts"] for p in preds],
            "ground_truth": [r["ground_truth"] for r in eval_rows],
        }
    )

    scores = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_emb,
    )
    payload["ragas"] = scores.to_dict()

    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["ragas"], indent=2))
    print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
