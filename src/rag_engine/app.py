"""
``RAGEngine`` — library entrypoint for ingest + Strategy A/B search (plan §7).

This is **not** the shell multi-step runner. For pytest / benchmark / smoke / markdown
steps that write under ``output/<RUN_ID>/``, use **``orchestrator.py``** at the repo root.
For **interactive queries** on a long plaintext corpus with real embeddings, use **``query_cli.py``**.
"""

from __future__ import annotations

from pathlib import Path

from rag_engine.chunking.factory import create_chunker
from rag_engine.config.loader import load_engine_config
from rag_engine.config.schema import EngineConfig
from rag_engine.documents.loader import load_documents_json
from rag_engine.documents.models import Document
from rag_engine.embeddings.factory import create_embedder, tokenizer_encode_decode
from rag_engine.retrieval.query_expander import MockGenerativeModel, QueryExpander
from rag_engine.retrieval.retriever import Retriever
from rag_engine.vectorstores.factory import create_vector_store


class RAGEngine:
    """Config-driven ingestion + Strategy A/B retrieval."""

    def __init__(self, config: EngineConfig | None = None, *, config_path: str | Path | None = None) -> None:
        self.config = config if config is not None else load_engine_config(config_path)
        self.embedder = create_embedder(self.config)
        enc, dec = tokenizer_encode_decode(self.embedder)
        self.chunker = create_chunker(self.config, encode=enc, decode=dec)
        self.vector_store = create_vector_store(self.config)
        self.query_expander = QueryExpander(MockGenerativeModel(), self.config.query_expansion)
        self.retriever = Retriever(
            self.embedder,
            self.vector_store,
            self.query_expander,
            self.config,
        )

    def ingest(self, documents_path: str | Path | None = None) -> int:
        path = Path(documents_path) if documents_path is not None else Path(self.config.data.input_path)
        docs = load_documents_json(path, self.config.data.text_field, self.config.data.id_field)
        return self.ingest_documents(docs)

    def ingest_documents(self, documents: list[Document]) -> int:
        """Chunk, embed, and index the given documents (same path as :meth:`ingest`)."""
        chunks = self.chunker.split_documents(documents)
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_documents(texts)
        self.vector_store.add(chunks, embeddings)
        return len(chunks)

    def search_raw(self, query: str):
        return self.retriever.retrieve_raw(query)

    def search_expanded(self, query: str):
        return self.retriever.retrieve_with_expansion(query)
