"""Plaintext document loading for long-form corpora."""

from __future__ import annotations

from pathlib import Path

from rag_engine.app import RAGEngine
from rag_engine.config.loader import load_engine_config
from rag_engine.documents.loader import load_documents_plaintext


def test_load_documents_plaintext_one_doc(tmp_path: Path) -> None:
    p = tmp_path / "chapter_one.txt"
    p.write_text("First paragraph.\n\nSecond paragraph with more words.\n", encoding="utf-8")
    docs = load_documents_plaintext(p)
    assert len(docs) == 1
    assert docs[0].document_id == "chapter_one"
    assert "First paragraph" in docs[0].text
    assert docs[0].metadata.get("source") == "plaintext_file"


def test_load_documents_plaintext_custom_id(tmp_path: Path) -> None:
    p = tmp_path / "x.md"
    p.write_text("# Title\n\nBody here.\n", encoding="utf-8")
    docs = load_documents_plaintext(p, document_id="my_book")
    assert docs[0].document_id == "my_book"


def test_ingest_documents_plaintext_mock_embed(tmp_path: Path) -> None:
    cfg = load_engine_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    cfg.embedding.provider = "mock_vertex"
    cfg.embedding.mock_vertex = cfg.embedding.mock_vertex.model_copy(update={"dimensions": 24})
    cfg.vector_store.provider = "numpy"
    cfg.retrieval.top_k = 2
    eng = RAGEngine(config=cfg)
    p = tmp_path / "corpus.txt"
    p.write_text("alpha beta gamma.\n\n" * 50, encoding="utf-8")
    docs = load_documents_plaintext(p)
    n = eng.ingest_documents(docs)
    assert n >= 1
    hits = eng.search_raw("gamma")
    assert len(hits) >= 1
