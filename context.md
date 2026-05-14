# Context-Aware Retrieval Engine


This document describes a configurable, local Retrieval-Augmented Generation (RAG) architecture for comparing two retrieval strategies:

- **Strategy A: Raw Vector Search** — embed the original query and perform semantic search.
- **Strategy B: AI-Enhanced Retrieval** — rewrite or expand the query using a mocked generative model, then embed and search.

The system is designed to be flexible through configuration, allowing developers to switch vector stores, chunking algorithms, embedding models, similarity metrics, and FAISS index types without changing core application code.

---

## 1. Architecture Overview

```text
                         ┌──────────────────────────┐
                         │        config.yaml        │
                         │ provider, chunking, index │
                         │ embedding, metric, top_k  │
                         └─────────────┬────────────┘
                                       │
                                       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│ Raw Dataset  │ ──▶ │ Document     │ ──▶ │ Chunking Engine  │
│ 5–10 paras   │     │ Loader       │     │ recursive/token  │
└──────────────┘     └──────────────┘     └─────────┬────────┘
                                                     │
                                                     ▼
                                           ┌──────────────────┐
                                           │ Embedding Engine │
                                           │ local / mocked   │
                                           └─────────┬────────┘
                                                     │
                                                     ▼
                                           ┌──────────────────┐
                                           │ Vector Store     │
                                           │ FAISS / Chroma   │
                                           └─────────┬────────┘
                                                     │
                         ┌───────────────────────────┴───────────────────────────┐
                         ▼                                                       ▼
          ┌─────────────────────────┐                         ┌─────────────────────────┐
          │ Strategy A              │                         │ Strategy B              │
          │ Raw Vector Search       │                         │ AI Query Expansion      │
          │ query → embed → search  │                         │ rewrite → embed → search│
          └────────────┬────────────┘                         └────────────┬────────────┘
                       │                                                   │
                       └─────────────────────┬─────────────────────────────┘
                                             ▼
                                  ┌────────────────────┐
                                  │ Benchmark Reporter │
                                  │ JSON + Markdown    │
                                  └────────────────────┘
```

---

## 2. Design Goals

The system is designed around the following principles:

1. **Config-driven development**  
   Developers should be able to change retrieval behavior from `config.yaml` only.

2. **Pluggable components**  
   Embedding, chunking, vector storage, query expansion, and retrieval should each have clean interfaces.

3. **Local-first implementation**  
   The assessment runs locally using `sentence-transformers`, FAISS or Chroma, and mocked Vertex AI classes.

4. **Evaluation-first retrieval**  
   The system must compare Strategy A and Strategy B using a structured benchmark report.

5. **Production migration path**  
   The same abstractions should support migration to Vertex AI Embeddings and Vertex AI Vector Search.

---

## 3. Recommended Default Stack

For the assessment, the best default stack is:

```text
Language: Python
Embedding: sentence-transformers/all-MiniLM-L6-v2
Vector store: FAISS
Default FAISS index: HNSW
Similarity metric: Cosine similarity via normalized inner product
Chunking: Recursive token-aware chunking
Query expansion: Mocked GenerativeModel
Benchmark output: JSON + retrieval_benchmark.md
Testing: pytest
Production target: Vertex AI Embeddings + Vertex AI Vector Search
```

Recommended default configuration:

```yaml
vector_store:
  provider: faiss
  index_type: hnsw

embedding:
  provider: sentence_transformers
  model_name: sentence-transformers/all-MiniLM-L6-v2
  normalize_embeddings: true

similarity:
  metric: cosine

chunking:
  algorithm: recursive
  chunk_size_tokens: 350
  chunk_overlap_tokens: 60

retrieval:
  top_k: 3
  fetch_k: 20

query_expansion:
  provider: mock_vertex_generative_model
  enabled: true
```

---

## 4. Exact System Flow

### 4.1 Ingestion Flow

```text
1. Load raw technical paragraphs.
2. Clean and normalize text.
3. Split documents into chunks using the configured chunking algorithm.
4. Attach metadata to each chunk.
5. Generate embeddings for all chunks.
6. Normalize embeddings if cosine similarity is configured.
7. Build the selected vector index.
8. Store:
   - vector index
   - chunk text
   - metadata
   - config snapshot
```

Example chunk metadata:

```json
{
  "chunk_id": "doc_001_chunk_003",
  "document_id": "doc_001",
  "source": "technical_dataset",
  "chunk_index": 3,
  "text": "During peak load, the system uses autoscaling...",
  "token_count": 142
}
```

---

### 4.2 Strategy A: Raw Vector Search

```text
User query
  ↓
Embed original query
  ↓
Normalize query embedding
  ↓
Search vector database
  ↓
Return top 3 chunks
```

Example:

```text
Input query:
"How does the system handle peak load?"

Search text:
"How does the system handle peak load?"
```

This is the baseline semantic retrieval strategy.

---

### 4.3 Strategy B: AI-Enhanced Retrieval

```text
User query
  ↓
Mock GenerativeModel rewrites / expands query
  ↓
Embed expanded query
  ↓
Normalize query embedding
  ↓
Search vector database
  ↓
Return top 3 chunks
```

Example:

```text
Original query:
"How does the system handle peak load?"

Expanded query:
"system behavior during peak traffic, autoscaling, load balancing, queueing, rate limiting, horizontal scaling, high concurrency, performance under heavy requests"
```

Strategy B is useful because short user queries may not contain the technical terms found inside the documents.

---

### 4.4 Benchmarking Flow

For each benchmark query:

```text
1. Run Strategy A.
2. Run Strategy B.
3. Capture top 3 chunks from both strategies.
4. Compare retrieved chunks.
5. Calculate overlap count.
6. Record scores and result previews.
7. Write output to JSON.
8. Write human-readable comparison to retrieval_benchmark.md.
```

---

## 5. Repository Structure

```text
rag-vector-assessment/
│
├── README.md
├── retrieval_benchmark.md
├── pyproject.toml
├── config/
│   └── config.yaml
│
├── data/
│   └── technical_paragraphs.json
│
├── storage/
│   ├── benchmark_results.json
│   └── vector_index/
│
├── src/
│   └── rag_engine/
│       ├── __init__.py
│       │
│       ├── config/
│       │   ├── schema.py
│       │   └── loader.py
│       │
│       ├── documents/
│       │   ├── loader.py
│       │   └── models.py
│       │
│       ├── chunking/
│       │   ├── base.py
│       │   ├── fixed_character.py
│       │   ├── fixed_token.py
│       │   ├── recursive.py
│       │   ├── sentence.py
│       │   ├── semantic.py
│       │   └── factory.py
│       │
│       ├── embeddings/
│       │   ├── base.py
│       │   ├── sentence_transformer_embedder.py
│       │   ├── mock_vertex_embedder.py
│       │   └── factory.py
│       │
│       ├── vectorstores/
│       │   ├── base.py
│       │   ├── faiss_store.py
│       │   ├── chroma_store.py
│       │   ├── numpy_store.py
│       │   └── factory.py
│       │
│       ├── retrieval/
│       │   ├── retriever.py
│       │   ├── query_expander.py
│       │   ├── strategies.py
│       │   └── result_models.py
│       │
│       ├── evaluation/
│       │   ├── benchmark.py
│       │   ├── reporter.py
│       │   └── metrics.py
│       │
│       └── app.py
│
└── tests/
    ├── test_chunking.py
    ├── test_embeddings.py
    ├── test_faiss_store.py
    ├── test_chroma_store.py
    ├── test_query_expansion.py
    ├── test_retrieval_pipeline.py
    └── test_benchmark_output.py
```

---

## 6. Core Interfaces

### 6.1 Base Chunker

```python
from abc import ABC, abstractmethod

class BaseChunker(ABC):
    @abstractmethod
    def split_documents(self, documents):
        pass
```

Implemented chunkers:

```text
FixedCharacterChunker
FixedTokenChunker
SentenceChunker
RecursiveChunker
SemanticChunker
```

---

### 6.2 Base Embedder

```python
from abc import ABC, abstractmethod

class BaseEmbedder(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        pass
```

Implemented embedders:

```text
SentenceTransformerEmbedder
MockVertexTextEmbeddingModel
```

---

### 6.3 Base Vector Store

```python
from abc import ABC, abstractmethod

class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, chunks, embeddings):
        pass

    @abstractmethod
    def search(self, query_embedding, top_k: int):
        pass

    @abstractmethod
    def persist(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass
```

Implemented stores:

```text
FaissVectorStore
ChromaVectorStore
NumpyVectorStore
```

---

## 7. RAG Engine Orchestrator

```python
class RAGEngine:
    def __init__(self, config):
        self.config = config
        self.chunker = ChunkerFactory.create(config.chunking)
        self.embedder = EmbedderFactory.create(config.embedding)
        self.vector_store = VectorStoreFactory.create(config.vector_store)
        self.query_expander = QueryExpander(config.query_expansion)

    def ingest(self, documents):
        chunks = self.chunker.split_documents(documents)
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts)
        self.vector_store.add(chunks, embeddings)

    def search_raw(self, query: str):
        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.config.retrieval.top_k,
        )

    def search_expanded(self, query: str):
        expanded_query = self.query_expander.expand(query)
        query_embedding = self.embedder.embed_query(expanded_query)
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.config.retrieval.top_k,
        )
        return expanded_query, results
```

---

## 8. Full Configuration File

```yaml
project:
  name: context-aware-retrieval-engine
  environment: local
  random_seed: 42

# -----------------------------
# Data Configuration
# -----------------------------
data:
  input_path: data/technical_paragraphs.json
  text_field: text
  id_field: id

# -----------------------------
# Chunking Configuration
# -----------------------------
chunking:
  algorithm: recursive

  options:
    fixed_character:
      chunk_size_chars: 1200
      chunk_overlap_chars: 150
      use_when: "quick baseline, simple text, not recommended for final quality"

    fixed_token:
      chunk_size_tokens: 350
      chunk_overlap_tokens: 60
      use_when: "strict control over embedding model token limits"

    sentence:
      max_sentences_per_chunk: 6
      overlap_sentences: 1
      use_when: "clean prose, articles, guides, documentation"

    recursive:
      chunk_size_tokens: 350
      chunk_overlap_tokens: 60
      separators:
        - "\n\n"
        - "\n"
        - ". "
        - " "
      keep_separator: true
      use_when: "best general-purpose default for document RAG"

    semantic:
      enabled: false
      breakpoint_threshold: 0.78
      min_chunk_tokens: 120
      max_chunk_tokens: 500
      use_when: "messy long documents where topic boundaries matter"

# -----------------------------
# Embedding Configuration
# -----------------------------
embedding:
  provider: sentence_transformers
  model_name: sentence-transformers/all-MiniLM-L6-v2
  batch_size: 32
  normalize_embeddings: true

  options:
    sentence_transformers:
      model_name: sentence-transformers/all-MiniLM-L6-v2
      device: cpu
      use_when: "local assessment implementation"

    mock_vertex:
      model_name: textembedding-gecko-mock
      dimensions: 384
      use_when: "mocking Vertex AI SDK behavior in tests"

# -----------------------------
# Similarity Configuration
# -----------------------------
similarity:
  metric: cosine

  options:
    cosine:
      normalize_embeddings: true
      faiss_index_metric: inner_product
      description: "Best default for semantic text retrieval. Cosine over normalized vectors equals inner product."

    euclidean:
      normalize_embeddings: false
      faiss_index_metric: l2
      description: "Useful for models trained with L2 distance, but not the default for text RAG."

    inner_product:
      normalize_embeddings: true
      faiss_index_metric: inner_product
      description: "Efficient FAISS implementation when vectors are normalized."

# -----------------------------
# Vector Store Configuration
# -----------------------------
vector_store:
  provider: faiss
  persist_path: storage/vector_index

  options:
    numpy:
      enabled: true
      use_when: "exact search baseline, small datasets, debugging"

    chroma:
      enabled: true
      collection_name: assessment_rag
      persist_directory: storage/chroma
      distance_metric: cosine
      use_when: "simple application integration and metadata management"

    faiss:
      enabled: true
      index_type: hnsw
      index_selection_policy: auto

      index_options:
        flat:
          index_factory: Flat
          use_when: "small corpus, exact baseline, debugging"

        hnsw:
          index_factory: HNSW32
          m: 32
          ef_construction: 200
          ef_search: 64
          use_when: "best default for local RAG quality"

        ivf_flat:
          index_factory: IVF1024,Flat
          nlist: 1024
          nprobe: 16
          training_sample_size: 10000
          use_when: "large corpus, tunable speed/recall"

        ivf_pq:
          index_factory: IVF4096,PQ64
          nlist: 4096
          nprobe: 32
          pq_m: 64
          training_sample_size: 50000
          use_when: "very large corpus, memory constrained"

# -----------------------------
# Retrieval Configuration
# -----------------------------
retrieval:
  top_k: 3
  fetch_k: 20
  return_scores: true
  return_metadata: true

  strategies:
    strategy_a_raw_vector:
      enabled: true
      description: "Embed original query and search."

    strategy_b_query_expansion:
      enabled: true
      expansion_provider: mock_vertex_generative_model
      description: "Rewrite query, embed expanded query, then search."

# -----------------------------
# Query Expansion Configuration
# -----------------------------
query_expansion:
  provider: mock_vertex_generative_model
  deterministic: true
  prompt_template: |
    Rewrite the user query into a search-friendly technical query.
    Include related technical terms, synonyms, and system behavior concepts.
    User query: {query}

# -----------------------------
# Benchmark Configuration
# -----------------------------
benchmark:
  output_json: storage/benchmark_results.json
  output_markdown: retrieval_benchmark.md

  queries:
    - "How does the system handle peak load?"
    - "What happens when downstream services fail?"
    - "How are user requests protected from unauthorized access?"

# -----------------------------
# Production Migration Notes
# -----------------------------
production_migration:
  target: vertex_ai_vector_search
  notes:
    - "Replace local embedding adapter with Vertex AI embeddings."
    - "Replace local vector store with Vertex AI Vector Search."
    - "Keep same chunking and retrieval strategy interfaces."
    - "Use cloud metadata storage for chunk text and document metadata."
```

---

## 9. Index Selection Policy

The system can automatically select a FAISS index based on the number of vectors and available memory.

```python
def choose_faiss_index(num_vectors: int, ram_budget_gb: float) -> str:
    if num_vectors < 50_000:
        return "flat"

    if num_vectors < 2_000_000 and ram_budget_gb >= 4:
        return "hnsw"

    if num_vectors < 10_000_000:
        return "ivf_flat"

    return "ivf_pq"
```

Recommended index decision table:

| Corpus size | Recommended index | Why |
|---|---|---|
| Tiny | FAISS Flat | Exact, simple, great for debugging |
| Small/medium | HNSW | Strong quality/speed default |
| Large | IVF Flat | Tunable speed/recall |
| Very large | IVF + PQ | Saves memory, scales better |
| App-first simple RAG | Chroma | Easier persistence and metadata handling |

---

## 10. FAISS vs Chroma Recommendation

### Use FAISS when:

```text
- You want direct control over Flat, HNSW, IVF Flat, and IVF+PQ.
- You want to demonstrate vector search internals.
- You want custom performance tuning.
- You want a strong assessment implementation.
```

### Use Chroma when:

```text
- You want faster application development.
- You want simple persistence.
- You want built-in metadata handling.
- You do not need low-level index control.
```

### Recommendation for this assessment

Use:

```text
Default: FAISS + HNSW
Optional: Chroma provider
Optional: NumPy exact search provider
```

This gives the best balance between technical depth and practical flexibility.

---

## 11. Similarity Metric Choice

The recommended similarity metric is **cosine similarity**.

Reason:

```text
For semantic text retrieval, vector direction is usually more important than vector magnitude.
```

Implementation detail:

```text
Normalize all embeddings to length 1.
Use inner product in FAISS.
Cosine similarity over normalized vectors is equivalent to inner product.
```

Example FAISS setup:

```python
import faiss
import numpy as np

vectors = np.asarray(vectors).astype("float32")
faiss.normalize_L2(vectors)

index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 64
index.add(vectors)
```

---

## 12. Benchmark Output Format

The benchmark should produce both JSON and Markdown.

### JSON Example

```json
{
  "query": "How does the system handle peak load?",
  "strategy_a": {
    "search_query": "How does the system handle peak load?",
    "top_results": [
      {
        "rank": 1,
        "chunk_id": "doc_002_chunk_001",
        "score": 0.82,
        "text_preview": "The system uses load balancing to distribute traffic..."
      }
    ]
  },
  "strategy_b": {
    "expanded_query": "peak traffic autoscaling load balancing queueing rate limiting high concurrency",
    "top_results": [
      {
        "rank": 1,
        "chunk_id": "doc_004_chunk_002",
        "score": 0.89,
        "text_preview": "During peak load, autoscaling adds workers..."
      }
    ]
  },
  "comparison": {
    "overlap_count": 1,
    "winner": "strategy_b",
    "reason": "Expanded query retrieved a more direct peak-load handling chunk."
  }
}
```

### Markdown Example

```markdown
# Retrieval Benchmark: Strategy A vs Strategy B

## Query 1: How does the system handle peak load?

### Strategy A: Raw Vector Search

| Rank | Chunk ID | Score | Preview |
|---:|---|---:|---|
| 1 | doc_001_chunk_002 | 0.812 | The system distributes traffic... |
| 2 | doc_003_chunk_001 | 0.774 | Request queues prevent overload... |
| 3 | doc_002_chunk_004 | 0.721 | Workers process events asynchronously... |

### Strategy B: AI-Enhanced Retrieval

Expanded query:

`peak traffic autoscaling load balancing queueing rate limiting high concurrency`

| Rank | Chunk ID | Score | Preview |
|---:|---|---:|---|
| 1 | doc_004_chunk_001 | 0.891 | During peak load, the system scales workers... |
| 2 | doc_001_chunk_002 | 0.846 | The system distributes traffic... |
| 3 | doc_003_chunk_001 | 0.803 | Request queues prevent overload... |

### Comparison

- Overlap: 2/3
- Winner: Strategy B
- Reason: Query expansion added technical terms like autoscaling and queueing, which retrieved more specific chunks.
```

---

## 13. Testing Plan

### `test_chunking.py`

Verify:

```text
- recursive chunker creates chunks under max size
- overlap is applied
- metadata is preserved
```

### `test_embeddings.py`

Verify:

```text
- local embedder returns expected vector shape
- normalized vectors have length close to 1
- mock Vertex embedding model behaves deterministically
```

### `test_faiss_store.py`

Verify:

```text
- vectors can be added
- search returns top_k results
- metadata mapping works
- selected FAISS index type is respected
```

### `test_chroma_store.py`

Verify:

```text
- Chroma collection is created
- documents and metadata are inserted
- search returns expected number of results
```

### `test_query_expansion.py`

Verify:

```text
- mock GenerativeModel rewrites known query
- query expansion is deterministic
- fallback returns original query when no mock rule exists
```

### `test_retrieval_pipeline.py`

Verify:

```text
- Strategy A returns top 3 results
- Strategy B returns top 3 results
- Strategy B uses expanded query
- results include score, text, and metadata
```

### `test_benchmark_output.py`

Verify:

```text
- benchmark writes JSON output
- benchmark writes retrieval_benchmark.md
- each query contains both strategy results
- overlap and winner fields are present
```

---

## 14. Production Migration to Vertex AI Vector Search

Local architecture:

```text
SentenceTransformerEmbedder
    ↓
FAISS / Chroma
    ↓
Local metadata store
```

Production architecture:

```text
VertexAIEmbedder
    ↓
Vertex AI Vector Search
    ↓
Cloud Storage / Firestore / BigQuery metadata store
```

Migration steps:

```text
1. Replace SentenceTransformerEmbedder with VertexAIEmbedder.
2. Keep chunking logic unchanged.
3. Generate embeddings using Vertex AI embedding model.
4. Export embeddings and metadata to the format required by Vertex AI Vector Search.
5. Create a Vertex AI Vector Search index.
6. Deploy the index to an endpoint.
7. Query the endpoint using the embedded user query.
8. Fetch full chunk text and metadata by returned IDs.
9. Keep Strategy B query expansion using Gemini / GenerativeModel.
10. Continue producing the same benchmark report format.
```

The important production design point is that the local implementation and the cloud implementation use the same interfaces:

```text
BaseEmbedder
BaseVectorStore
BaseRetriever
BenchmarkRunner
```

Only the implementation changes.

---

## 15. How to Explain This to the Tech Lead

Use this explanation:

```text
I designed the system around pluggable interfaces: Embedder, Chunker, VectorStore, Retriever, QueryExpander, and Evaluator.

For the assessment, the default path uses sentence-transformers, recursive chunking, cosine similarity, and FAISS HNSW. But the config supports switching to Chroma, NumPy exact search, FAISS Flat, IVF Flat, or IVF+PQ.

I kept Strategy A and Strategy B in the same retrieval pipeline so their results are directly comparable. The benchmark outputs both JSON and Markdown, making it easy to inspect whether query expansion improves retrieval quality.

For production, I would keep the same chunking and retrieval abstractions, replace the local embedder with Vertex AI embeddings, and replace FAISS with Vertex AI Vector Search.
```

---

## 16. Final Recommendation

For this assignment, implement:

```text
FAISS + HNSW + normalized embeddings + cosine similarity + recursive chunking + mocked query expansion benchmark
```

Keep these as optional config choices:

```text
Vector stores:
- faiss
- chroma
- numpy

FAISS indexes:
- flat
- hnsw
- ivf_flat
- ivf_pq
- auto

Chunking algorithms:
- fixed_character
- fixed_token
- sentence
- recursive
- semantic

Embedding providers:
- sentence_transformers
- mock_vertex

Similarity metrics:
- cosine
- euclidean
- inner_product
```

This design satisfies the assessment, demonstrates strong retrieval engineering knowledge, and shows a realistic path to production on GCP.
