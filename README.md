# Document Intelligence RAG

> **Production-style RAG over your documents** — semantic chunking, ChromaDB retrieval, Groq LLaMA-3.1 answers, Streamlit UI, and RAGAS evaluation.

[![CI](https://github.com/Shubham-p-02/document-intelligence-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Shubham-p-02/document-intelligence-rag/actions) *(badge placeholder — add workflow when ready)*

| | |
|---|---|
| **Live demo** | *Coming soon — [Hugging Face Space](https://huggingface.co/spaces) (duplicate this repo as a Streamlit Space; set `GROQ_API_KEY` in Space secrets)* |
| **Groq API** | Free tier key: [console.groq.com](https://console.groq.com/) |

### Tech stack

- **LangChain** (LCEL orchestration)
- **ChromaDB** (persistent vector store)
- **RAGAS** (faithfulness, answer relevancy, context precision)
- **Groq** (`llama-3.1-8b-instant`)
- **Streamlit** (interactive Q&A UI)
- **Hugging Face Spaces** (optional hosted demo)

### Recruiter quick start (~5 min)

```bash
git clone https://github.com/Shubham-p-02/document-intelligence-rag.git
cd document-intelligence-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your key: https://console.groq.com/ → GROQ_API_KEY in .env
streamlit run app.py
```

1. Open the app → set corpus to `sample_corpus` → **Build / rebuild index**.
2. Ask a question (e.g. *What is semantic chunking?*) → expand **Sources** for citations.
3. Optional eval: `python scripts/evaluate_rag.py --corpus sample_corpus --target-queries 20 --skip-ragas`

### Deploy to Hugging Face Spaces (no secrets in git)

1. On [huggingface.co/new-space](https://huggingface.co/new-space), choose **Streamlit** and duplicate or push this repository.
2. In Space **Settings → Repository secrets**, add `GROQ_API_KEY` only (never commit `.env`).
3. Use repo root `requirements.txt` and `.streamlit/config.toml`; optional `Dockerfile` for Docker Spaces.

---

## Stack

| Layer | Choice |
|--------|--------|
| Orchestration | LangChain LCEL |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings` |
| Chunking | LangChain Experimental `SemanticChunker` |
| Vector DB | ChromaDB (persisted) |
| LLM | Groq `llama-3.1-8b-instant` |
| UI | Streamlit |
| Eval | RAGAS 0.2.x (`faithfulness`, `answer_relevancy`, `context_precision`) |

## Semantic chunking approach

Fixed-size splits often cut mid-thought. This project uses **`SemanticChunker`** (`langchain_experimental`), which:

1. Embeds sentences (or buffer groups) with the same model used for retrieval.
2. Measures consecutive embedding distances.
3. Inserts breakpoints where distance exceeds a **percentile** threshold (default **95th** percentile via `SEMANTIC_BREAKPOINT_AMOUNT_THRESHOLD`).

After semantic splits, an optional **character overlap** (`CHUNK_OVERLAP_CHARS`, default `120`) prepends the tail of the previous chunk to improve cross-boundary recall. This is documented overlap behavior—not the classic `RecursiveCharacterTextSplitter` overlap slider.

## Retrieval & generation

- **Chroma** stores chunk embeddings; queries use `similarity_search_with_score` (L2 distance on normalized vectors).
- **`RAG_TOP_K`** (default `4`) and **`RAG_SIMILARITY_THRESHOLD`** (default `1.6`) filter out chunks with L2 distance **greater than** the threshold (stricter = lower number). If nothing passes, top-k results are still returned.
- **Generation**: `build_rag_chain()` in `chain.py` runs retrieval → prompt → Groq → answer.

## Confidence scoring (limitations)

The UI shows **Retrieval confidence (heuristic)** = `max(exp(-distance))` over retrieved chunks. This is:

- **Not** a calibrated probability or legal-grade certainty score.
- Sensitive to embedding model, distance metric, and threshold tuning.
- Blind to hallucinations when retrieval looks strong but the LLM invents facts.

For production, combine retrieval scores with answer-level checks (citation overlap, NLI, or a separate judge LLM).

## Quick start (local)

```bash
cd document-intelligence-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GROQ_API_KEY
streamlit run app.py
```

Or: `chmod +x run.sh && ./run.sh`

1. Point **Corpus folder** at `sample_corpus` (default) or upload files.
2. Click **Build / rebuild index**.
3. Ask questions; expand **Sources** for chunk text + file metadata.

## Environment variables

See `.env.example`. Required for chat and full RAGAS runs:

- `GROQ_API_KEY` — from [Groq Console](https://console.groq.com/)

## RAGAS evaluation (200+ queries)

Synthetic questions are generated from indexed chunks; optional golden rows merge from `eval/golden_example.json`.

```bash
export GROQ_API_KEY=...
python scripts/evaluate_rag.py --corpus sample_corpus --target-queries 220
```

Tuning knobs (CLI mirrors env):

- `--top-k`, `--max-distance`, `--overlap-chars`, `--semantic-percentile`

Dry-run without judge API cost (build index + run RAG only):

```bash
python scripts/evaluate_rag.py --target-queries 220 --skip-ragas
```

Index + synthetic question generation only (no `GROQ_API_KEY`):

```bash
python scripts/evaluate_rag.py --target-queries 220 --index-only
```

Results: `eval/ragas_last_run.json`

### Example tuning narrative (illustrative targets)

After a real eval run, compare `ragas` aggregates in the JSON output. **Example targets after tuning** (not measured in CI without your API key):

| Metric | Baseline (example) | Tuned (example) |
|--------|-------------------|-----------------|
| faithfulness | 0.71 | 0.89 |
| answer_relevancy | 0.68 | 0.85 |
| context_precision | 0.62 | 0.80 |

Typical tuning levers: lower `top-k`, tighten `max-distance`, raise semantic percentile (fewer, larger chunks) or increase overlap for recall-heavy corpora.

### RAGAS / Groq notes

- Pinned `ragas>=0.2,<0.3` for stable `LangchainLLMWrapper` + `evaluate()` API.
- Judge LLM uses the same Groq model as the app; rate limits may require `--throttle-seconds`.
- Full 220-query eval is **slow and API-heavy**; start with `--target-queries 20` for smoke tests.

## Hugging Face Spaces

1. Create a **Streamlit** Space.
2. Upload this folder (or connect a git repo).
3. Add **Secrets**: `GROQ_API_KEY`.
4. Ensure `requirements.txt` and `.streamlit/config.toml` are at the repo root.
5. Optional: `packages.txt` for apt deps; `Dockerfile` for container Spaces.

App command (Space settings): `streamlit run app.py --server.port=8501 --server.address=0.0.0.0`

Pre-build an index in the Space only if you ship corpus files; otherwise users build from uploads.

## Performance

- Default **MiniLM** embedding model for speed on CPU.
- Small **top-k** reduces prompt size and Groq latency.
- Sub-2s average end-to-end latency depends on Groq region/load, corpus size, first-time model download, and hardware—not guaranteed on cold start.

## Project layout

```
document-intelligence-rag/
├── app.py                 # Streamlit UI
├── streamlit_app.py       # HF Spaces entry alias
├── ingest.py              # PDF / TXT / MD loading
├── chunking.py            # SemanticChunker + overlap
├── vectorstore.py         # Chroma build/load/retrieve
├── chain.py               # LCEL RAG + Groq
├── scripts/evaluate_rag.py
├── eval/golden_example.json
├── sample_corpus/
├── requirements.txt
├── .env.example
├── .streamlit/config.toml
├── run.sh
└── Makefile
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `GROQ_API_KEY is not set` | Copy `.env.example` → `.env` or export the variable |
| `No index found` | Build index in sidebar or run eval script (creates `.chroma_eval`) |
| PDF errors | Install `pypdf`; verify file is not encrypted |
| RAGAS import errors | `pip install -r requirements.txt` (ragas 0.2.x) |

## License

MIT (sample corpus and code for demonstration).
