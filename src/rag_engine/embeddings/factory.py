"""Embedder factory."""

from __future__ import annotations

from rag_engine.config.schema import EngineConfig
from rag_engine.embeddings.base import BaseEmbedder
from rag_engine.embeddings.mock_vertex_embedder import MockVertexEmbedder
from rag_engine.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder


def create_embedder(cfg: EngineConfig) -> BaseEmbedder:
    if cfg.embedding.provider == "mock_vertex":
        return MockVertexEmbedder(
            dimensions=cfg.embedding.mock_vertex.dimensions,
            normalize_embeddings=cfg.embedding.normalize_embeddings,
        )
    return SentenceTransformerEmbedder(
        model_name=cfg.embedding.model_name,
        batch_size=cfg.embedding.batch_size,
        normalize_embeddings=cfg.embedding.normalize_embeddings,
        query_instruction=cfg.embedding.query_instruction,
    )


def tokenizer_encode_decode(embedder: BaseEmbedder):
    """Return (encode, decode) for token chunkers, or (None, None) for mocks."""
    tok = getattr(embedder, "tokenizer", None)
    if tok is None:
        return None, None

    def encode(text: str) -> list[int]:
        # Chunking must see the *true* token count. Some HF defaults can cap counts or warn
        # when model_max_length is misaligned; never truncate here.
        return tok.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )

    def decode(ids: list[int]) -> str:
        return tok.decode(ids, skip_special_tokens=True)

    return encode, decode
