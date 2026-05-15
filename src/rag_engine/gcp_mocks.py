"""Stubs matching the Vertex ``vertexai.language_models`` surface for tests and local mocks.

The assessment brief names ``TextEmbeddingModel`` and ``GenerativeModel`` at this path.
Tests can ``patch("vertexai.language_models.GenerativeModel", ...)``; ``tests/conftest.py``
registers this module so imports resolve without installing ``google-cloud-aiplatform``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from rag_engine.canned_query_expansions import expansion_text_for_prompt


def _mock_embedding_vector(text: str, dimensions: int) -> np.ndarray:
    if not text.strip():
        raise ValueError("Cannot embed empty text.")
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, "little")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dimensions).astype(np.float32)
    return v


@dataclass
class TextEmbeddingInput:
    text: str
    task_type: str | None = None


@dataclass
class Embedding:
    values: list[float]


class TextEmbeddingModel:
    """SDK-shaped stub: ``from_pretrained`` → ``get_embeddings([TextEmbeddingInput])``."""

    def __init__(self, model_name: str, *, dimensions: int) -> None:
        self._model_name = model_name
        self._dimensions = dimensions

    @classmethod
    def from_pretrained(cls, model_name: str, *, dimensions: int | None = None) -> TextEmbeddingModel:
        dim = dimensions if dimensions is not None else 384
        return cls(model_name, dimensions=dim)

    def get_embeddings(self, inputs: list[TextEmbeddingInput]) -> list[Embedding]:
        out: list[Embedding] = []
        for inp in inputs:
            v = _mock_embedding_vector(inp.text, self._dimensions)
            out.append(Embedding(values=v.astype(float).tolist()))
        return out


class GenerativeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class GenerativeModel:
    """SDK-shaped stub wrapping the same canned expansions as :class:`QueryExpander`."""

    def __init__(self, model_name: str = "gemini-mock", *, expansions: dict[str, str] | None = None) -> None:
        self._model_name = model_name
        self._expansions = expansions

    def generate_content(self, prompt: str) -> GenerativeResponse:
        return GenerativeResponse(expansion_text_for_prompt(prompt, self._expansions))
