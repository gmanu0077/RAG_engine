"""Recursive separator-first splitting (plan §8 chunking.recursive)."""

from __future__ import annotations

from collections.abc import Callable

from rag_engine.chunking.base import BaseChunker
from rag_engine.config.schema import RecursiveChunkingParams
from rag_engine.documents.models import Chunk, Document


class RecursiveChunker(BaseChunker):
    def __init__(self, params: RecursiveChunkingParams, len_tokens: Callable[[str], int]) -> None:
        self._p = params
        self._len_tokens = len_tokens

    def _split_oversized(self, text: str, separators: list[str]) -> list[str]:
        if self._len_tokens(text) <= self._p.chunk_size_tokens:
            return [text]
        if not separators:
            return self._hard_slice(text)
        sep = separators[0]
        rest = separators[1:]
        if sep not in text:
            return self._split_oversized(text, rest)
        raw_parts = text.split(sep)
        parts: list[str] = []
        for i, p in enumerate(raw_parts):
            piece = p
            if self._p.keep_separator and sep and i < len(raw_parts) - 1:
                piece = p + sep
            piece = piece.strip()
            if piece:
                parts.extend(self._split_oversized(piece, rest))
        return parts

    def _hard_slice(self, text: str) -> list[str]:
        approx = max(64, self._p.chunk_size_tokens * 4)
        ov = self._p.chunk_overlap_tokens * 4
        out: list[str] = []
        i = 0
        while i < len(text):
            chunk = text[i : i + approx].strip()
            if chunk:
                out.append(chunk)
            i += max(1, approx - ov)
        return out if out else [text]

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        if not pieces:
            return []
        merged: list[str] = []
        buf = pieces[0]
        ov_toks = self._p.chunk_overlap_tokens
        max_t = self._p.chunk_size_tokens
        for p in pieces[1:]:
            cand = buf + "\n" + p if buf else p
            if self._len_tokens(cand) <= max_t:
                buf = cand
            else:
                merged.append(buf.strip())
                tail = self._tail_by_tokens(buf, ov_toks)
                buf = (tail + "\n" + p).strip() if tail else p
        if buf.strip():
            merged.append(buf.strip())
        return merged

    def _tail_by_tokens(self, text: str, n_tokens: int) -> str:
        if n_tokens <= 0 or not text:
            return ""
        low = 0
        high = len(text)
        target = min(n_tokens, self._len_tokens(text))
        while low < high:
            mid = (low + high + 1) // 2
            tail = text[-mid:]
            if self._len_tokens(tail) <= target:
                low = mid
            else:
                high = mid - 1
        return text[-low:] if low else ""

    def split_documents(self, documents: list[Document]) -> list[Chunk]:
        out: list[Chunk] = []
        seps = list(self._p.separators)
        for doc in documents:
            raw = self._split_oversized(doc.text, seps)
            blocks = self._merge_with_overlap(raw)
            for ci, block in enumerate(blocks):
                if not block.strip():
                    continue
                tid = f"{doc.document_id}_chunk_{ci:03d}"
                out.append(
                    Chunk(
                        chunk_id=tid,
                        document_id=doc.document_id,
                        text=block.strip(),
                        chunk_index=ci,
                        token_count=self._len_tokens(block.strip()),
                        metadata={"source": doc.metadata.get("source", "dataset")},
                    ),
                )
        return out
