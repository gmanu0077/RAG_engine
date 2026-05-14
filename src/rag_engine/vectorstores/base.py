"""Vector store protocol (plan §6.3)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_engine.documents.models import Chunk
    from rag_engine.retrieval.result_models import SearchResult


class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        pass

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        pass

    def persist(self, path: str | Path) -> None:
        _ = path

    def load(self, path: str | Path) -> None:
        _ = path
