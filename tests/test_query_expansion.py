"""Query expansion (plan §13)."""

from __future__ import annotations

from pathlib import Path

from rag_engine.config.loader import load_engine_config
from rag_engine.retrieval.query_expander import MockGenerativeModel, QueryExpander


def test_expansion_known_query() -> None:
    cfg = load_engine_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    ex = QueryExpander(MockGenerativeModel(), cfg.query_expansion)
    out = ex.expand("How does the system handle peak load?")
    assert "autoscaling" in out.lower() or "scalability" in out.lower()


def test_expansion_fallback_original() -> None:
    cfg = load_engine_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    ex = QueryExpander(MockGenerativeModel(), cfg.query_expansion)
    q = "totally unknown query xyz123"
    assert ex.expand(q) == q
