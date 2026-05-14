"""Deterministic mock embeddings (fast tests, plan §6.2)."""

from __future__ import annotations

import hashlib

import numpy as np

from rag_engine.embeddings.base import BaseEmbedder


class MockVertexEmbedder(BaseEmbedder):
    def __init__(self, dimensions: int, normalize_embeddings: bool) -> None:
        self._dim = dimensions
        self._normalize = normalize_embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Cannot embed empty text.")
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        seed = int.from_bytes(digest, "little")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self._dim).astype(np.float32)
        if self._normalize:
            v = v / (np.linalg.norm(v) + 1e-9)
        return v.astype(float).tolist()
