"""Recursive / fixed chunkers (plan §13)."""

from __future__ import annotations

from pathlib import Path

from rag_engine.chunking.factory import create_chunker
from rag_engine.config.loader import load_engine_config
from rag_engine.config.schema import EngineConfig
from rag_engine.documents.models import Document


def _cfg() -> EngineConfig:
    c = load_engine_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    c.embedding.provider = "mock_vertex"
    c.vector_store.provider = "numpy"
    return c


def test_recursive_chunker_respects_token_budget() -> None:
    cfg = _cfg()
    cfg.chunking.algorithm = "recursive"
    cfg.chunking.recursive.chunk_size_tokens = 80
    cfg.chunking.recursive.chunk_overlap_tokens = 10
    ch = create_chunker(cfg)
    docs = [Document("d1", "para1\n\n" + ("word " * 200), {})]
    chunks = ch.split_documents(docs)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.token_count <= cfg.chunking.recursive.chunk_size_tokens + 80


def test_fixed_character_chunker() -> None:
    cfg = _cfg()
    cfg.chunking.algorithm = "fixed_character"
    cfg.chunking.fixed_character.chunk_size_chars = 100
    cfg.chunking.fixed_character.chunk_overlap_chars = 10
    ch = create_chunker(cfg)
    docs = [Document("d1", "x" * 250, {})]
    chunks = ch.split_documents(docs)
    assert len(chunks) >= 2
