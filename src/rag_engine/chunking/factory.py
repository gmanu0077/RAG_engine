"""Chunker factory (plan §6.1 implementations)."""

from __future__ import annotations

from collections.abc import Callable

from rag_engine.chunking.base import BaseChunker
from rag_engine.chunking.fixed_character import FixedCharacterChunker
from rag_engine.chunking.fixed_token import FixedTokenChunker
from rag_engine.chunking.recursive import RecursiveChunker
from rag_engine.chunking.semantic import SemanticChunker
from rag_engine.chunking.sentence import SentenceChunker
from rag_engine.config.schema import EngineConfig


def _char_token_proxy(text: str) -> int:
    return max(1, len(text) // 4)


def create_chunker(
    cfg: EngineConfig,
    *,
    encode: Callable[[str], list[int]] | None = None,
    decode: Callable[[list[int]], str] | None = None,
) -> BaseChunker:
    algo = cfg.chunking.algorithm
    if algo == "fixed_character":
        return FixedCharacterChunker(cfg.chunking.fixed_character)

    len_tokens: Callable[[str], int]
    if encode is not None:

        def len_tokens(t: str) -> int:
            return len(encode(t))

    else:
        len_tokens = _char_token_proxy

    if algo == "recursive":
        return RecursiveChunker(cfg.chunking.recursive, len_tokens)
    if algo == "sentence":
        return SentenceChunker(cfg.chunking.sentence)
    if algo == "semantic":
        return SemanticChunker(cfg.chunking.semantic, len_tokens)
    if algo == "fixed_token":
        if encode is None or decode is None:
            raise ValueError("fixed_token chunking requires tokenizer encode/decode from the embedder factory.")
        return FixedTokenChunker(cfg.chunking.fixed_token, encode, decode)
    raise ValueError(f"Unknown chunking algorithm: {algo}")
