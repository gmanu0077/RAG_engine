"""Typed documents and chunks (plan §4.1 metadata example)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_metadata(self, **extra: Any) -> Chunk:
        m = {**self.metadata, **extra}
        return Chunk(
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            text=self.text,
            chunk_index=self.chunk_index,
            token_count=self.token_count,
            metadata=m,
        )
