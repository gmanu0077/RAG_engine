"""Embedders (plan §13)."""

from __future__ import annotations

import numpy as np

from rag_engine.embeddings.mock_vertex_embedder import MockVertexEmbedder
from rag_engine.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder


def test_mock_vertex_normalized() -> None:
    e = MockVertexEmbedder(dimensions=32, normalize_embeddings=True)
    v = np.asarray(e.embed_query("hello world"), dtype=np.float32)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-3


def test_sentence_transformer_mini_lm_shapes() -> None:
    e = SentenceTransformerEmbedder(
        "sentence-transformers/all-MiniLM-L6-v2",
        batch_size=8,
        normalize_embeddings=True,
    )
    batch = e.embed_documents(["a", "b"])
    assert len(batch) == 2
    assert len(batch[0]) == 384
    q = e.embed_query("probe")
    assert len(q) == 384
