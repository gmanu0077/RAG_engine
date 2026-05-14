"""Sentence-boundary grouping."""

from __future__ import annotations

import re

from rag_engine.chunking.base import BaseChunker
from rag_engine.config.schema import SentenceChunkingParams
from rag_engine.documents.models import Chunk, Document

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


class SentenceChunker(BaseChunker):
    def __init__(self, params: SentenceChunkingParams) -> None:
        self._p = params

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        out: list[Chunk] = []
        max_s = self._p.max_sentences_per_chunk
        ov_s = self._p.overlap_sentences
        for doc in documents:
            sentences = [s.strip() for s in _SENT_SPLIT.split(doc.text) if s.strip()]
            if not sentences:
                continue
            i = 0
            ci = 0
            while i < len(sentences):
                take = min(max_s, len(sentences) - i)
                block = " ".join(sentences[i : i + take])
                tid = f"{doc.document_id}_chunk_{ci:03d}"
                out.append(
                    Chunk(
                        chunk_id=tid,
                        document_id=doc.document_id,
                        text=block,
                        chunk_index=ci,
                        token_count=max(1, len(block) // 4),
                        metadata={"source": doc.metadata.get("source", "dataset")},
                    )
                )
                ci += 1
                step = max(1, take - ov_s)
                i += step
        return out
