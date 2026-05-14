"""FAISS backends: Flat, HNSW, IVF-Flat, IVF-PQ, and auto policy (plan §8–9)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from rag_engine.config.schema import EngineConfig
from rag_engine.documents.models import Chunk
from rag_engine.retrieval.result_models import SearchResult
from rag_engine.vectorstores.base import BaseVectorStore
from rag_engine.vectorstores.index_policy import choose_faiss_index_type


class FaissVectorStore(BaseVectorStore):
    def __init__(self, engine_cfg: EngineConfig) -> None:
        self._engine_cfg = engine_cfg
        self._chunks: list[Chunk] = []
        self._index: faiss.Index | None = None
        self._dim: int | None = None
        self._use_ip: bool = True

    def _metric_inner_product(self) -> bool:
        return self._engine_cfg.similarity.metric in ("cosine", "inner_product")

    def _prepare_matrix(self, mat: np.ndarray) -> np.ndarray:
        mat = np.asarray(mat, dtype=np.float32, order="C")
        if not np.all(np.isfinite(mat)):
            raise ValueError("Embeddings must be finite.")
        self._use_ip = self._metric_inner_product()
        if self._use_ip:
            faiss.normalize_L2(mat)
        return mat

    def _prepare_query(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float32, order="C").reshape(1, -1)
        if self._use_ip:
            faiss.normalize_L2(q)
        return q

    def _resolve_index_type(self, num_vectors: int) -> str:
        fp = self._engine_cfg.vector_store.faiss
        if fp.index_selection_policy == "auto" or fp.index_type == "auto":
            return choose_faiss_index_type(num_vectors, self._engine_cfg.vector_store.ram_budget_gb)
        return fp.index_type

    def _train_ivf(self, index: faiss.Index, train_mat: np.ndarray, n: int) -> None:
        fp = self._engine_cfg.vector_store.faiss
        max_train = min(n, fp.ivf_training_sample_max)
        if max_train < index.nlist:
            raise ValueError("Too few vectors to train IVF; use flat or hnsw for tiny corpora.")
        rng = np.random.default_rng(self._engine_cfg.project.random_seed)
        if max_train < n:
            pick = rng.choice(n, size=max_train, replace=False)
            xt = train_mat[pick].copy()
        else:
            xt = train_mat.copy()
        index.train(xt)

    def _build_index(self, dim: int, n: int, train_mat: np.ndarray, index_type: str) -> faiss.Index:
        fp = self._engine_cfg.vector_store.faiss
        metric_ip = self._metric_inner_product()

        def flat() -> faiss.Index:
            return faiss.IndexFlatIP(dim) if metric_ip else faiss.IndexFlatL2(dim)

        if index_type == "flat" or n < max(8, fp.ivf_nlist // 4):
            return flat()

        if index_type == "hnsw":
            m = fp.hnsw_m
            if metric_ip:
                idx = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
            else:
                idx = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_L2)
            idx.hnsw.efConstruction = fp.hnsw_ef_construction
            idx.hnsw.efSearch = fp.hnsw_ef_search
            return idx

        if index_type == "ivf_pq":
            pq_m = fp.ivf_pq_m
            if dim % pq_m != 0 or n < fp.ivf_nlist:
                return self._build_index(dim, n, train_mat, "ivf_flat")
            quantizer = faiss.IndexFlatIP(dim) if metric_ip else faiss.IndexFlatL2(dim)
            idx = faiss.IndexIVFPQ(quantizer, dim, fp.ivf_nlist, pq_m, 8)
            self._train_ivf(idx, train_mat, n)
            idx.nprobe = min(fp.ivf_nprobe, fp.ivf_nlist)
            return idx

        quantizer = faiss.IndexFlatIP(dim) if metric_ip else faiss.IndexFlatL2(dim)
        idx = faiss.IndexIVFFlat(
            quantizer,
            dim,
            fp.ivf_nlist,
            faiss.METRIC_INNER_PRODUCT if metric_ip else faiss.METRIC_L2,
        )
        self._train_ivf(idx, train_mat, n)
        idx.nprobe = min(fp.ivf_nprobe, fp.ivf_nlist)
        return idx

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch.")
        if not chunks:
            return
        mat = self._prepare_matrix(np.stack([np.asarray(e, dtype=np.float32) for e in embeddings]))
        dim = mat.shape[1]
        n = mat.shape[0]
        idx_type = self._resolve_index_type(n)
        self._dim = dim
        self._chunks = list(chunks)
        try:
            self._index = self._build_index(dim, n, mat, idx_type)
        except ValueError:
            self._index = self._build_index(dim, n, mat, "flat")
        self._index.add(mat)
        if idx_type == "hnsw" and hasattr(self._index, "hnsw"):
            self._index.hnsw.efSearch = self._engine_cfg.vector_store.faiss.hnsw_ef_search

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        if self._index is None or not self._chunks:
            return []
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        q = self._prepare_query(np.asarray(query_embedding, dtype=np.float32))
        if int(q.shape[1]) != int(self._index.d):
            raise ValueError("Query dimension mismatch.")
        if hasattr(self._index, "hnsw"):
            self._index.hnsw.efSearch = self._engine_cfg.vector_store.faiss.hnsw_ef_search
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = min(
                self._engine_cfg.vector_store.faiss.ivf_nprobe,
                self._engine_cfg.vector_store.faiss.ivf_nlist,
            )
        k = min(top_k, len(self._chunks))
        scores, idxs = self._index.search(q, k)
        row_scores = scores[0]
        row_idx = idxs[0]
        hits: list[tuple[int, float, Chunk]] = []
        for col, i in enumerate(row_idx):
            if int(i) < 0:
                continue
            ch = self._chunks[int(i)]
            raw = float(row_scores[col])
            score = raw if self._use_ip else -raw
            hits.append((int(i), score, ch))
        hits.sort(key=lambda t: (-t[1], t[2].chunk_id, t[0]))
        out: list[SearchResult] = []
        for rank, (_, sc, ch) in enumerate(hits[:k], start=1):
            out.append(
                SearchResult(
                    rank=rank,
                    chunk_id=ch.chunk_id,
                    score=sc,
                    text=ch.text,
                    metadata={**ch.metadata, "document_id": ch.document_id},
                ),
            )
        return out

    def persist(self, path: str | Path) -> None:
        if self._index is None:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(p / "index.faiss"))
        rows = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "text": c.text,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
                "metadata": c.metadata,
            }
            for c in self._chunks
        ]
        (p / "chunks.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        p = Path(path)
        self._index = faiss.read_index(str(p / "index.faiss"))
        rows = json.loads((p / "chunks.json").read_text(encoding="utf-8"))
        self._chunks = [
            Chunk(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                text=r["text"],
                chunk_index=int(r["chunk_index"]),
                token_count=int(r["token_count"]),
                metadata=dict(r.get("metadata") or {}),
            )
            for r in rows
        ]
        self._dim = self._index.d
        self._use_ip = self._metric_inner_product()
