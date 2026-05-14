"""Token-id windows using a HuggingFace tokenizer when available."""

from __future__ import annotations

from collections.abc import Callable

from rag_engine.chunking.base import BaseChunker
from rag_engine.config.schema import FixedTokenChunkingParams
from rag_engine.documents.models import Chunk, Document


class FixedTokenChunker(BaseChunker):
    def __init__(
        self,
        params: FixedTokenChunkingParams,
        encode: Callable[[str], list[int]],
        decode: Callable[[list[int]], str],
    ) -> None:
        self._p = params
        self._encode = encode
        self._decode = decode

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        out: list[Chunk] = []
        cs, ov = self._p.chunk_size_tokens, self._p.chunk_overlap_tokens
        if ov >= cs:
            raise ValueError("chunk_overlap_tokens must be < chunk_size_tokens")
        for doc in documents:
            ids = self._encode(doc.text)
            if not ids:
                continue
            start = 0
            ci = 0
            while start < len(ids):
                end = min(start + cs, len(ids))
                piece = self._decode(ids[start:end]).strip()
                if piece:
                    tid = f"{doc.document_id}_chunk_{ci:03d}"
                    out.append(
                        Chunk(
                            chunk_id=tid,
                            document_id=doc.document_id,
                            text=piece,
                            chunk_index=ci,
                            token_count=end - start,
                            metadata={"source": doc.metadata.get("source", "dataset")},
                        )
                    )
                    ci += 1
                if end >= len(ids):
                    break
                start = max(end - ov, start + 1)
        return out
