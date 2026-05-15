"""Local sentence-transformers embedder."""

from __future__ import annotations

import numpy as np

from rag_engine.embeddings.base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str,
        batch_size: int,
        normalize_embeddings: bool,
        *,
        query_instruction: str = "",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        self._batch_size = batch_size
        self._normalize = normalize_embeddings
        self._query_instruction = query_instruction.strip() if query_instruction else ""
        self._align_max_seq_length_with_transformer()

    def _align_max_seq_length_with_transformer(self) -> None:
        """Raise ST's truncation window when it is below the backbone's ``max_position_embeddings``.

        Some sentence-transformer packages cap ``max_seq_length`` below the backbone’s
        ``max_position_embeddings``. Lifting the cap avoids silent truncation for chunk sizes
        near the model window (e.g. ~350-token recursive chunks).
        """
        try:
            mod0 = self._model[0]
            internal = getattr(mod0, "auto_model", None)
            if internal is None:
                return
            mpe = getattr(getattr(internal, "config", None), "max_position_embeddings", None)
            if not (isinstance(mpe, int) and mpe > 0):
                return
            if self._model.max_seq_length < mpe:
                self._model.max_seq_length = mpe
            tok = self._model.tokenizer
            # Keep HF tokenizer window in sync so encode(..., truncation=False) is not capped
            # at a stale 256-style limit while chunking targets ~350 tokens (BGE/MiniLM-class).
            mm = getattr(tok, "model_max_length", None)
            if isinstance(mm, int) and mm < mpe:
                tok.model_max_length = mpe
        except Exception:
            return

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
        raw = text.strip()
        if not raw:
            raise ValueError("Cannot embed empty query.")
        q = (self._query_instruction + raw) if self._query_instruction else raw
        vec = self._model.encode(
            [q],
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
