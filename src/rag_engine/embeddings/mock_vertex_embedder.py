"""Deterministic mock embeddings (fast tests, plan §6.2).

Uses ``vertexai.language_models.TextEmbeddingModel`` / ``TextEmbeddingInput`` when stubs
are installed (see :mod:`rag_engine.vertex_stubs`).
"""

from __future__ import annotations

import numpy as np

from rag_engine.embeddings.base import BaseEmbedder
from rag_engine.vertex_stubs import ensure_vertexai_stub_modules

ensure_vertexai_stub_modules()
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel  # noqa: E402


class MockVertexEmbedder(BaseEmbedder):
    def __init__(self, dimensions: int, normalize_embeddings: bool) -> None:
        self._normalize = normalize_embeddings
        self._sdk = TextEmbeddingModel.from_pretrained("textembedding-gecko-mock", dimensions=dimensions)

    def _maybe_norm(self, vec: list[float]) -> list[float]:
        v = np.asarray(vec, dtype=np.float32)
        if self._normalize:
            v = v / (np.linalg.norm(v) + 1e-9)
        return v.astype(float).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if any(not t.strip() for t in texts):
            raise ValueError("Cannot embed empty string in batch.")
        embs = self._sdk.get_embeddings([TextEmbeddingInput(text=t.strip()) for t in texts])
        return [self._maybe_norm(list(e.values)) for e in embs]

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Cannot embed empty text.")
        embs = self._sdk.get_embeddings([TextEmbeddingInput(text=text.strip())])
        return self._maybe_norm(list(embs[0].values))
