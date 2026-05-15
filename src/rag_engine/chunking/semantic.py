"""Greedy paragraph merge by token budget (lightweight semantic proxy)."""

from __future__ import annotations

from collections.abc import Callable

from rag_engine.chunking.base import BaseChunker
from rag_engine.config.schema import SemanticChunkingParams
from rag_engine.documents.models import Chunk, Document


class SemanticChunker(BaseChunker):
    """Merges coarse paragraphs until ``max_chunk_tokens``; no embedding model required."""

    def __init__(self, params: SemanticChunkingParams, len_tokens: Callable[[str], int]) -> None:
        self._p = params
        self._len_tokens = len_tokens

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        out: list[Chunk] = []
        for doc in documents:
            paras = [p.strip() for p in doc.text.split("\n\n") if p.strip()]
            if not paras:
                continue
            buf = paras[0]
            ci = 0
            for p in paras[1:]:
                cand = buf + "\n\n" + p
                if self._len_tokens(cand) <= self._p.max_chunk_tokens:
                    buf = cand
                else:
                    if buf.strip():
                        tid = f"{doc.document_id}_chunk_{ci:03d}"
                        out.append(
                            Chunk(
                                chunk_id=tid,
                                document_id=doc.document_id,
                                text=buf,
                                chunk_index=ci,
                                token_count=self._len_tokens(buf),
                                metadata={"source": doc.metadata.get("source", "dataset")},
                            ),
                        )
                        ci += 1
                    buf = p
            if buf.strip():
                tid = f"{doc.document_id}_chunk_{ci:03d}"
                out.append(
                    Chunk(
                        chunk_id=tid,
                        document_id=doc.document_id,
                        text=buf,
                        chunk_index=ci,
                        token_count=self._len_tokens(buf),
                        metadata={"source": doc.metadata.get("source", "dataset")},
                    ),
                )
        return out
