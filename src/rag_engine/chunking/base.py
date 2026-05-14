"""Chunker protocol (plan §6.1)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_engine.documents.models import Chunk, Document


class BaseChunker(ABC):
    @abstractmethod
    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into ordered chunks with stable ids."""
