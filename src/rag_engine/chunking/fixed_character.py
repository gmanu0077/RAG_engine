"""Fixed-size character windows with overlap."""

from __future__ import annotations

from rag_engine.chunking.base import BaseChunker
from rag_engine.config.schema import FixedCharacterChunkingParams
from rag_engine.documents.models import Chunk, Document


class FixedCharacterChunker(BaseChunker):
    def __init__(self, params: FixedCharacterChunkingParams) -> None:
        self._p = params

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        out: list[Chunk] = []
        size, ov = self._p.chunk_size_chars, self._p.chunk_overlap_chars
        if ov >= size:
            raise ValueError("chunk_overlap_chars must be < chunk_size_chars")
        for doc in documents:
            text = doc.text
            start = 0
            ci = 0
            n = len(text)
            while start < n:
                end = min(start + size, n)
                piece = text[start:end].strip()
                if piece:
                    tid = f"{doc.document_id}_chunk_{ci:03d}"
                    out.append(
                        Chunk(
                            chunk_id=tid,
                            document_id=doc.document_id,
                            text=piece,
                            chunk_index=ci,
                            token_count=max(1, len(piece) // 4),
                            metadata={"source": doc.metadata.get("source", "dataset")},
                        )
                    )
                    ci += 1
                if end >= n:
                    break
                start = max(end - ov, start + 1)
        return out
