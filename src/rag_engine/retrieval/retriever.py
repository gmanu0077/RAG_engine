"""Retriever: Strategy A (raw) vs Strategy B (expanded) — plan §7."""

from __future__ import annotations

from rag_engine.config.schema import EngineConfig
from rag_engine.embeddings.base import BaseEmbedder
from rag_engine.retrieval.query_expander import QueryExpander
from rag_engine.retrieval.result_models import ExpandedSearchResult, SearchResult
from rag_engine.vectorstores.base import BaseVectorStore


class Retriever:
    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        query_expander: QueryExpander,
        cfg: EngineConfig,
    ) -> None:
        self._embedder = embedder
        self._store = vector_store
        self._expander = query_expander
        self.engine_cfg = cfg

    def retrieve_raw(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        self._validate_query(query)
        k = top_k if top_k is not None else self.engine_cfg.retrieval.top_k
        fetch = max(k, self.engine_cfg.retrieval.fetch_k)
        qe = self._embedder.embed_query(query.strip())
        hits = self._store.search(qe, top_k=fetch)
        return hits[:k]

    def retrieve_with_expansion(self, query: str, top_k: int | None = None) -> ExpandedSearchResult:
        self._validate_query(query)
        k = top_k if top_k is not None else self.engine_cfg.retrieval.top_k
        fetch = max(k, self.engine_cfg.retrieval.fetch_k)
        expanded = self._expander.expand(query)
        qe = self._embedder.embed_query(expanded)
        hits = self._store.search(qe, top_k=fetch)
        return ExpandedSearchResult(
            original_query=query.strip(),
            expanded_query=expanded,
            results=hits[:k],
        )

    @staticmethod
    def _validate_query(query: str) -> None:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
