"""FAISS store (plan §13)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rag_engine.config.loader import load_engine_config
from rag_engine.config.schema import EngineConfig
from rag_engine.documents.models import Chunk
from rag_engine.embeddings.mock_vertex_embedder import MockVertexEmbedder
from rag_engine.vectorstores.faiss_store import FaissVectorStore


def _tiny_cfg(index_type: str) -> EngineConfig:
    c = load_engine_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    c.embedding.provider = "mock_vertex"
    c.embedding.mock_vertex = c.embedding.mock_vertex.model_copy(update={"dimensions": 16})
    c.similarity.metric = "cosine"
    c.vector_store.provider = "faiss"
    c.vector_store.faiss = c.vector_store.faiss.model_copy(
        update={"index_type": index_type, "index_selection_policy": "manual"}
    )
    return c


def test_faiss_flat_search_order() -> None:
    cfg = _tiny_cfg("flat")
    store = FaissVectorStore(cfg)
    emb = MockVertexEmbedder(16, normalize_embeddings=True)
    base = np.zeros(16, dtype=np.float32)
    base[0] = 1.0
    v1 = base.tolist()
    v2 = (np.roll(base, 1)).tolist()
    q = base.tolist()
    chunks = [
        Chunk("a", "d", "t1", 0, 1, {}),
        Chunk("b", "d", "t2", 1, 1, {}),
    ]
    store.add(chunks, [v1, v2])
    hits = store.search(q, top_k=2)
    assert hits[0].chunk_id == "a"


def test_faiss_hnsw_search() -> None:
    cfg = _tiny_cfg("hnsw")
    store = FaissVectorStore(cfg)
    emb = MockVertexEmbedder(16, normalize_embeddings=True)
    vecs = [emb.embed_query(f"x{i}") for i in range(12)]
    chunks = [Chunk(f"c{i}", "d", f"t{i}", i, 1, {}) for i in range(12)]
    store.add(chunks, vecs)
    hits = store.search(vecs[3], top_k=3)
    assert len(hits) == 3
