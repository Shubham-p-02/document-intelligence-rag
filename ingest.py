"""
Load documents from disk: .txt, .md, and optional .pdf (requires pypdf).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)

SUPPORTED_TEXT = {".txt", ".md", ".markdown"}
SUPPORTED_PDF = {".pdf"}


def _read_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_file(path: Path) -> List[Document]:
    """Load a single file into one or more LangChain Documents."""
    path = path.resolve()
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_TEXT:
        text = _read_utf8(path)
        return [Document(page_content=text, metadata={"source": str(path), "file_type": suffix})]
    if suffix in SUPPORTED_PDF:
        try:
            loader = PyPDFLoader(str(path))
            docs = loader.load()
            for d in docs:
                d.metadata.setdefault("source", str(path))
                d.metadata.setdefault("file_type", ".pdf")
            return docs
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to load PDF %s: %s", path, exc)
            raise RuntimeError(
                f"Could not read PDF '{path}'. Install pypdf and ensure the file is not corrupted. "
                f"Original error: {exc}"
            ) from exc
    raise ValueError(
        f"Unsupported file type '{suffix}' for '{path}'. "
        f"Supported: {sorted(SUPPORTED_TEXT | SUPPORTED_PDF)}"
    )


def iter_corpus_files(corpus_dir: Path) -> Iterable[Path]:
    corpus_dir = corpus_dir.expanduser().resolve()
    if not corpus_dir.is_dir():
        raise FileNotFoundError(
            f"Corpus directory does not exist or is not a directory: {corpus_dir}"
        )
    exts = SUPPORTED_TEXT | SUPPORTED_PDF
    for p in sorted(corpus_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def ingest_corpus(corpus_dir: str | Path) -> List[Document]:
    """
    Recursively load all supported documents under corpus_dir.
    Each returned Document includes metadata['source'] as absolute file path.
    """
    corpus_dir = Path(corpus_dir)
    documents: List[Document] = []
    for file_path in iter_corpus_files(corpus_dir):
        try:
            documents.extend(load_file(file_path))
        except ValueError:
            continue
    if not documents:
        raise ValueError(
            f"No supported documents found under {corpus_dir}. "
            f"Add .txt, .md, or .pdf files."
        )
    return documents


def ingest_uploaded_files(uploaded_file_like) -> List[Document]:
    """
    Build Documents from Streamlit UploadedFile objects (name + bytes).
    `uploaded_file_like` should be iterable of objects with `.name` and `.read()` / `.getbuffer()`.
    """
    import tempfile

    documents: List[Document] = []
    for uf in uploaded_file_like:
        name = getattr(uf, "name", "upload")
        suffix = Path(name).suffix.lower()
        data = uf.read() if hasattr(uf, "read") else bytes(uf.getbuffer())
        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
            tmp.write(data)
            tmp.flush()
            tmp_path = Path(tmp.name)
            if suffix in SUPPORTED_TEXT:
                text = _read_utf8(tmp_path)
                documents.append(
                    Document(page_content=text, metadata={"source": name, "file_type": suffix})
                )
            elif suffix in SUPPORTED_PDF:
                for d in load_file(tmp_path):
                    meta = dict(d.metadata)
                    meta.setdefault("source", name)
                    documents.append(Document(page_content=d.page_content, metadata=meta))
            else:
                raise ValueError(f"Unsupported upload type: {suffix}")
    if not documents:
        raise ValueError("No documents produced from uploads.")
    return documents
