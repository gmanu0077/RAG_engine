"""End-to-end retrieval (plan §13)."""

from __future__ import annotations

from pathlib import Path

from rag_engine.app import RAGEngine
from rag_engine.config.loader import load_engine_config


def test_strategy_a_and_b_top_k() -> None:
    cfg = load_engine_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    cfg.embedding.provider = "mock_vertex"
    cfg.embedding.mock_vertex = cfg.embedding.mock_vertex.model_copy(update={"dimensions": 24})
    cfg.vector_store.provider = "numpy"
    cfg.retrieval.top_k = 3
    eng = RAGEngine(config=cfg)
    n = eng.ingest(Path(__file__).resolve().parents[1] / "data" / "technical_paragraphs.json")
    assert n > 0
    raw = eng.search_raw("peak load")
    exp = eng.search_expanded("peak load")
    assert len(raw) <= 3
    assert len(exp.results) <= 3
    assert exp.expanded_query
