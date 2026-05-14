"""Vector store factory (plan §6.3)."""

from __future__ import annotations

from rag_engine.config.schema import EngineConfig
from rag_engine.vectorstores.base import BaseVectorStore
from rag_engine.vectorstores.chroma_store import ChromaVectorStore
from rag_engine.vectorstores.faiss_store import FaissVectorStore
from rag_engine.vectorstores.numpy_store import NumpyVectorStore


def create_vector_store(cfg: EngineConfig) -> BaseVectorStore:
    p = cfg.vector_store.provider
    if p == "numpy":
        return NumpyVectorStore()
    if p == "chroma":
        return ChromaVectorStore(cfg)
    if p == "faiss":
        return FaissVectorStore(cfg)
    raise ValueError(f"Unknown vector_store.provider: {p}")
