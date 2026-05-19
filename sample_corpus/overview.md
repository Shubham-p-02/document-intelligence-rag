# Sample corpus for local smoke tests

This folder ships with the project so you can build an index without preparing data first.

## Section A — Retrieval basics

Document Intelligence combines **ingestion**, **semantic chunking**, **vector retrieval**, and **LLM generation**.
The retrieval step returns the most relevant text chunks for a user question.

## Section B — Evaluation with RAGAS

RAGAS metrics such as **faithfulness** and **answer relevancy** help compare baseline and tuned pipelines.
Synthetic questions can be generated from chunks when a curated golden set is unavailable.

## Section C — Operational notes

- Keep `top-k` small for latency.
- Tune the max distance threshold alongside embedding model choice.
- Store `GROQ_API_KEY` outside the repository; never commit secrets.
