.PHONY: install app eval eval-skip clean

install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

app:
	.venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0

eval:
	.venv/bin/python scripts/evaluate_rag.py --target-queries 220

eval-skip:
	.venv/bin/python scripts/evaluate_rag.py --target-queries 220 --skip-ragas

clean:
	rm -rf .chroma_index .chroma_eval .venv __pycache__ */__pycache__
