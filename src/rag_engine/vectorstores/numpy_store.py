"""Exact NumPy cosine / IP search baseline (plan §8 vector_store.options.numpy)."""

from __future__ import annotations

import numpy as np

from rag_engine.documents.models import Chunk
from rag_engine.retrieval.result_models import SearchResult
from rag_engine.vectorstores.base import BaseVectorStore


class NumpyVectorStore(BaseVectorStore):
    """Normalized inner product (cosine on unit vectors); deterministic tie-break."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch.")
        if not chunks:
            return
        mat = np.asarray(embeddings, dtype=np.float32)
        if not np.all(np.isfinite(mat)):
            raise ValueError("Embeddings must be finite.")
        norms = np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-12)
        normalized = (mat / norms).astype(np.float32)
        if self._embeddings is None:
            self._embeddings = normalized
        else:
            if self._embeddings.shape[1] != normalized.shape[1]:
                raise ValueError("Embedding dimension mismatch.")
            self._embeddings = np.vstack([self._embeddings, normalized])
        self._chunks.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        if self._embeddings is None or not self._chunks:
            return []
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(q)):
            raise ValueError("Query embedding must be finite.")
        if q.shape[0] != self._embeddings.shape[1]:
            raise ValueError("Dimension mismatch.")
        q = q / (np.linalg.norm(q) + 1e-12)
        scores = self._embeddings @ q
        k = min(top_k, len(self._chunks))
        indices = list(range(len(self._chunks)))
        indices.sort(key=lambda i: (-float(scores[i]), self._chunks[i].chunk_id, i))
        out: list[SearchResult] = []
        for rank, idx in enumerate(indices[:k], start=1):
            ch = self._chunks[idx]
            out.append(
                SearchResult(
                    rank=rank,
                    chunk_id=ch.chunk_id,
                    score=float(scores[idx]),
                    text=ch.text,
                    metadata={**ch.metadata, "document_id": ch.document_id},
                ),
            )
        return out
