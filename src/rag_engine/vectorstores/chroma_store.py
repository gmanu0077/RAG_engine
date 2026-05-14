"""Chroma persistent vector store (plan §8)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import chromadb
from chromadb.config import Settings

from rag_engine.config.schema import EngineConfig
from rag_engine.documents.models import Chunk
from rag_engine.retrieval.result_models import SearchResult
from rag_engine.vectorstores.base import BaseVectorStore


def _l2_normalize(vec: list[float]) -> list[float]:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("Zero vector.")
    return (v / n).astype(float).tolist()


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, engine_cfg: EngineConfig) -> None:
        self._engine_cfg = engine_cfg
        cp = engine_cfg.vector_store.chroma
        cp.persist_directory.mkdir(parents=True, exist_ok=True)
        space = "cosine"
        if engine_cfg.similarity.metric == "euclidean":
            space = "l2"
        self._client = chromadb.PersistentClient(
            path=str(cp.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=cp.collection_name,
            metadata={"hnsw:space": space},
        )

    def _maybe_norm(self, emb: list[float]) -> list[float]:
        if self._engine_cfg.similarity.metric in ("cosine", "inner_product"):
            return _l2_normalize(emb)
        return emb

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch.")
        if not chunks:
            return
        embs = [self._maybe_norm(list(map(float, e))) for e in embeddings]
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embs,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    **{k: (str(v) if not isinstance(v, str) else v) for k, v in c.metadata.items()},
                    "document_id": str(c.document_id),
                    "chunk_index": str(c.chunk_index),
                    "token_count": str(c.token_count),
                }
                for c in chunks
            ],
        )

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self._collection.count() == 0:
            return []
        qe = self._maybe_norm(list(map(float, query_embedding)))
        n = self._collection.count()
        res = self._collection.query(
            query_embeddings=[qe],
            n_results=min(top_k, max(1, n)),
            include=["distances", "documents", "metadatas"],
        )
        ids = (res.get("ids") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        documents = (res.get("documents") or [[]])[0]
        metadatas = (res.get("metadatas") or [[]])[0]
        hits: list[SearchResult] = []
        for rank, (cid, dist, doc, meta) in enumerate(
            zip(ids, distances, documents, metadatas, strict=True), start=1
        ):
            hits.append(
                SearchResult(
                    rank=rank,
                    chunk_id=str(cid),
                    score=-float(dist),
                    text=str(doc or ""),
                    metadata=dict(meta or {}),
                ),
            )
        hits.sort(key=lambda h: (-h.score, h.chunk_id))
        for r, h in enumerate(hits, start=1):
            hits[r - 1] = SearchResult(
                rank=r,
                chunk_id=h.chunk_id,
                score=h.score,
                text=h.text,
                metadata=h.metadata,
            )
        return hits

    def persist(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "chroma_manifest.json").write_text(
            json.dumps(
                {
                    "persist_directory": str(self._engine_cfg.vector_store.chroma.persist_directory),
                    "collection": self._engine_cfg.vector_store.chroma.collection_name,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
