"""Chroma store (plan §13)."""

from __future__ import annotations

from pathlib import Path

from rag_engine.config.loader import load_engine_config
from rag_engine.config.schema import EngineConfig
from rag_engine.documents.models import Chunk
from rag_engine.embeddings.mock_vertex_embedder import MockVertexEmbedder
from rag_engine.vectorstores.chroma_store import ChromaVectorStore


def _cfg(tmp: Path) -> EngineConfig:
    c = load_engine_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    c.embedding.provider = "mock_vertex"
    c.embedding.mock_vertex = c.embedding.mock_vertex.model_copy(update={"dimensions": 32})
    c.vector_store.provider = "chroma"
    c.vector_store.chroma = c.vector_store.chroma.model_copy(
        update={
            "persist_directory": tmp / "chroma_db",
            "collection_name": "test_collection",
        }
    )
    return c


def test_chroma_add_and_query(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    store = ChromaVectorStore(cfg)
    emb = MockVertexEmbedder(32, normalize_embeddings=True)
    texts = ["redis cache ttl", "kubernetes autoscaling hpa"]
    vecs = [emb.embed_query(t) for t in texts]
    chunks = [
        Chunk("c0", "d0", texts[0], 0, 5, {}),
        Chunk("c1", "d0", texts[1], 1, 5, {}),
    ]
    store.add(chunks, vecs)
    hits = store.search(emb.embed_query("cache"), top_k=2)
    assert len(hits) >= 1
