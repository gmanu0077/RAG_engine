"""Pydantic schema aligned with ``plan.md`` §8 (machine-readable keys)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ProjectConfig(BaseModel):
    name: str = "context-aware-retrieval-engine"
    environment: Literal["local", "staging", "prod"] = "local"
    random_seed: int = 42


class DataConfig(BaseModel):
    input_path: Path = Field(default_factory=lambda: Path("data/technical_paragraphs.json"))
    text_field: str = "text"
    id_field: str = "id"


class RecursiveChunkingParams(BaseModel):
    chunk_size_tokens: int = Field(default=350, ge=16)
    chunk_overlap_tokens: int = Field(default=60, ge=0)
    separators: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ". ", " "],
    )
    keep_separator: bool = True


class FixedCharacterChunkingParams(BaseModel):
    chunk_size_chars: int = Field(default=1200, ge=64)
    chunk_overlap_chars: int = Field(default=150, ge=0)


class FixedTokenChunkingParams(BaseModel):
    chunk_size_tokens: int = Field(default=350, ge=16)
    chunk_overlap_tokens: int = Field(default=60, ge=0)


class SentenceChunkingParams(BaseModel):
    max_sentences_per_chunk: int = Field(default=6, ge=1)
    overlap_sentences: int = Field(default=1, ge=0)


class SemanticChunkingParams(BaseModel):
    enabled: bool = False
    breakpoint_threshold: float = Field(default=0.78, ge=0.0, le=1.0)
    min_chunk_tokens: int = Field(default=120, ge=8)
    max_chunk_tokens: int = Field(default=500, ge=32)


class ChunkingConfig(BaseModel):
    algorithm: Literal[
        "recursive",
        "fixed_character",
        "fixed_token",
        "sentence",
        "semantic",
    ] = "recursive"
    recursive: RecursiveChunkingParams = Field(default_factory=RecursiveChunkingParams)
    fixed_character: FixedCharacterChunkingParams = Field(default_factory=FixedCharacterChunkingParams)
    fixed_token: FixedTokenChunkingParams = Field(default_factory=FixedTokenChunkingParams)
    sentence: SentenceChunkingParams = Field(default_factory=SentenceChunkingParams)
    semantic: SemanticChunkingParams = Field(default_factory=SemanticChunkingParams)


class MockVertexEmbeddingParams(BaseModel):
    dimensions: int = Field(default=384, ge=8)


class EmbeddingConfig(BaseModel):
    provider: Literal["sentence_transformers", "mock_vertex"] = "sentence_transformers"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = Field(default=32, ge=1)
    normalize_embeddings: bool = True
    mock_vertex: MockVertexEmbeddingParams = Field(default_factory=MockVertexEmbeddingParams)


class SimilarityConfig(BaseModel):
    metric: Literal["cosine", "euclidean", "inner_product"] = "cosine"


class ChromaStoreParams(BaseModel):
    collection_name: str = "assessment_rag"
    persist_directory: Path = Field(default_factory=lambda: Path("storage/chroma"))


class FaissStoreParams(BaseModel):
    index_type: Literal["flat", "hnsw", "ivf_flat", "ivf_pq", "auto"] = "hnsw"
    index_selection_policy: Literal["manual", "auto"] = "manual"
    hnsw_m: int = Field(default=32, ge=4)
    hnsw_ef_construction: int = Field(default=200, ge=8)
    hnsw_ef_search: int = Field(default=64, ge=4)
    ivf_nlist: int = Field(default=1024, ge=4)
    ivf_nprobe: int = Field(default=16, ge=1)
    ivf_pq_m: int = Field(default=64, ge=4)
    ivf_training_sample_max: int = Field(default=10000, ge=256)


class VectorStoreConfig(BaseModel):
    provider: Literal["faiss", "chroma", "numpy"] = "faiss"
    persist_path: Path = Field(default_factory=lambda: Path("storage/vector_index"))
    ram_budget_gb: float = Field(default=4.0, ge=0.5)
    chroma: ChromaStoreParams = Field(default_factory=ChromaStoreParams)
    faiss: FaissStoreParams = Field(default_factory=FaissStoreParams)


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=3, ge=1)
    fetch_k: int = Field(default=20, ge=1)
    return_scores: bool = True
    return_metadata: bool = True
    no_match_cosine_threshold: float = Field(
        default=0.12,
        ge=0.0,
        le=1.0,
        description="Design note for benchmark text when top similarity is low.",
    )

    @model_validator(mode="after")
    def fetch_ge_top(self) -> RetrievalConfig:
        if self.fetch_k < self.top_k:
            raise ValueError("fetch_k must be >= top_k")
        return self


class QueryExpansionConfig(BaseModel):
    provider: Literal["mock_vertex_generative_model", "none"] = "mock_vertex_generative_model"
    enabled: bool = True
    deterministic: bool = True
    expansion_max_chars: int = Field(default=512, ge=32)
    prompt_template: str = (
        "Rewrite the user query into a search-friendly technical query.\n"
        "Include related technical terms, synonyms, and system behavior concepts.\n"
        "User query: {query}\n"
    )


class BenchmarkConfig(BaseModel):
    output_json: Path = Field(default_factory=lambda: Path("storage/benchmark_results.json"))
    output_markdown: Path = Field(default_factory=lambda: Path("retrieval_benchmark.md"))
    queries: list[str] = Field(
        default_factory=lambda: [
            "How does the system handle peak load?",
            "What happens when downstream services fail?",
            "How are user requests protected from unauthorized access?",
        ],
    )


class EngineConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    similarity: SimilarityConfig = Field(default_factory=SimilarityConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    query_expansion: QueryExpansionConfig = Field(default_factory=QueryExpansionConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    model_config = {"frozen": False}
