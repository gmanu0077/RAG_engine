"""Local sentence-transformers embedder."""

from __future__ import annotations

import numpy as np

from rag_engine.embeddings.base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, batch_size: int, normalize_embeddings: bool) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        self._batch_size = batch_size
        self._normalize = normalize_embeddings

    @property
    def tokenizer(self):
        return self._model.tokenizer

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        cleaned = [t.strip() for t in texts]
        if any(not t for t in cleaned):
            raise ValueError("Cannot embed empty string in batch.")
        out: list[list[float]] = []
        for i in range(0, len(cleaned), self._batch_size):
            batch = cleaned[i : i + self._batch_size]
            mat = self._model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
            )
            for j in range(mat.shape[0]):
                out.append(self._validate(mat[j]))
        return out

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Cannot embed empty query.")
        vec = self._model.encode(
            [text.strip()],
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )[0]
        return self._validate(vec)

    @staticmethod
    def _validate(vec: np.ndarray) -> list[float]:
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(v)):
            raise ValueError("Non-finite embedding.")
        if float(np.linalg.norm(v)) == 0.0:
            raise ValueError("Zero-norm embedding.")
        return v.astype(float).tolist()
