# TELEPORT — Design & Operations

This document is the **engineering decision log** for the context-aware retrieval engine in this repository: what we built, **why** key choices were made, how Strategy A vs Strategy B is evaluated **honestly**, and how the same shapes migrate to **Vertex AI**. For day-to-day commands, see **`README.md`**. For the long-form technical specification (interfaces, full YAML commentary, index policy tables), see **`context.md`**.

---

## 1. What this project is

A **local**, **config-driven** retrieval stack used to compare:

| Strategy | Flow |
|----------|------|
| **A — Raw vector** | User query → embed → vector search → top‑k chunks. |
| **B — Query expansion** | Same query → **mock** generative rewrite → embed expanded text → search → top‑k chunks. |

The bundled corpus is a small JSON list of technical paragraphs (`data/technical_paragraphs.json`). The code also supports **plaintext corpora** and an **interactive CLI** (`query_cli.py`) for longer documents; the **orchestrator** (`orchestrator.py`) runs tests, benchmarks, smoke, and markdown export with per-run artifacts under `output/<RUN_ID>/`.

**Non-goals for this doc:** line-by-line duplication of every `config.yaml` key. Use `config/config.yaml` (annotated) and `context.md` for that depth.

---

## 2. Architecture (high level)

```text
config/config.yaml
        │
        ▼
┌───────────────┐    ┌────────────┐    ┌─────────────┐    ┌──────────────┐
│ Load JSON or │───▶│ Chunking   │───▶│ Embedder   │───▶│ Vector store │
│ plaintext    │    │ (optional) │    │ ST or mock │    │ FAISS/Chroma │
└───────────────┘    └────────────┘    └──────┬──────┘    │ / NumPy      │
                                              │           └──────┬───────┘
                                              │                  │
                    Query ────────────────────┴──────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
       Strategy A                      Strategy B
   (embed raw query)              (expand then embed)
              │                               │
              └───────────────┬───────────────┘
                              ▼
                    Benchmark / CLI output
                    (JSON, Markdown, run logs)
```

**Module map (`src/rag_engine/`):**

| Area | Responsibility |
|------|------------------|
| `config/` | Pydantic schema + YAML loader; single source of tunables. |
| `documents/` | JSON and plaintext → `Document` / `Chunk` models. |
| `chunking/` | Pluggable splitters (recursive default); token counts respect embedder limits where configured. |
| `embeddings/` | `SentenceTransformerEmbedder`, `MockVertexEmbedder`; tokenizer hooks for token chunkers. |
| `vectorstores/` | FAISS (Flat / HNSW / IVF / auto policy), Chroma, NumPy baseline. |
| `retrieval/` | `Retriever`, `QueryExpander`, result types. |
| `evaluation/` | Benchmark runner, metrics, JSON + Markdown reporters. |
| `app.py` | `RAGEngine` — ingest, `search_raw`, `search_expanded`. |
| `gcp_mocks.py` | Vertex-**shaped** `TextEmbeddingModel` / `GenerativeModel` for local runs. |
| `vertex_stubs.py` | When the real `google-cloud-aiplatform` SDK is absent, registers minimal `vertexai.language_models` so imports match the brief’s literal paths. |
| `canned_query_expansions.py` | Deterministic expansion strings for the mock generative path. |

---

## 3. Design principles

1. **Config over code** — Prefer changing `config/config.yaml` to editing Python when tuning chunking, store, retrieval, or benchmark queries.

2. **Pluggable boundaries** — Factories select embedder, chunker, and vector store implementations from config so adapters (e.g. Vertex) can swap behind the same interfaces.

3. **Local-first** — Default path uses `sentence-transformers` + on-disk or in-memory indexes; mocks keep CI fast and deterministic.

4. **Honest evaluation** — Benchmarks report **structured comparison** (hits, overlap, score delta with caveats). They do **not** declare a single “winner” from incomparable cosine magnitudes between two different embedded query strings (see §7).

5. **Production path** — The same abstractions are described for Vertex embeddings, generative expansion, and **Vertex AI Vector Search** (see §8).

---

## 4. Embeddings and chunking

### 4.1 Default embedder

**`sentence-transformers/all-MiniLM-L6-v2`** is the default: widely available, good baseline for short technical text, 384 dimensions. The wrapper aligns **`max_seq_length`** with the backbone’s **`max_position_embeddings`** (512) where the library had defaulted lower, so chunk budgets in config remain consistent with what the model can encode without silent truncation.

**Mock path:** `MockVertexEmbedder` routes through the **same API shape** as Vertex `TextEmbeddingModel.from_pretrained(...).get_embeddings([TextEmbeddingInput(...)])` implemented in `gcp_mocks.py`, so tests exercise the SDK surface without calling Google.

### 4.2 Chunking

Recursive token-aware chunking is the default. For **very long** single files (e.g. Gutenberg texts), token counting avoids sending megabyte strings through the HF tokenizer in one call (see `chunking/factory.py` guard). For the bundled **paragraph JSON** corpus, chunking often collapses to roughly one chunk per paragraph — that is acceptable; the machinery stays useful for larger corpora and for `query_cli.py`.

---

## 5. Similarity metric: cosine vs Euclidean

**Why cosine (or inner product on L2-normalized vectors) for text**

Sentence-style encoders are trained with **cosine-like** contrastive objectives: semantic information lives primarily in **vector direction**, not raw length (length is influenced by document length and tokenization). **Cosine similarity** measures alignment of directions. **Euclidean distance** on the same unconstrained vectors mixes direction and magnitude and is usually a weaker default for general-purpose text embeddings.

**Implementation**

With **`similarity.metric`** in `cosine` or `inner_product`, embeddings are **L2-normalized** and FAISS uses an **inner product** index — equivalent to cosine on the unit sphere and efficient.

**When Euclidean is appropriate**

Use **L2 / Euclidean** when the model or training pipeline is explicitly **metric-learned for L2** (some image, audio, or multimodal spaces). Do not flip to Euclidean for text MiniLM/BGE-style models unless you have a measured reason.

**Reading scores**

Strategy A and Strategy B embed **different strings** (original vs expanded). **Top-1 cosine values are not directly comparable** as a one-dimensional “quality” score between strategies. Interpret **chunk id overlap**, **rank changes**, and **text previews**; treat signed **`top1_score_delta_b_minus_a`** in JSON as **informational only**, with the caveat in `comparison.notes`.

---

## 6. Vertex AI naming and mocks (assessment alignment)

The assessment brief references:

- `vertexai.language_models.TextEmbeddingModel`
- `vertexai.language_models.GenerativeModel`

**Behavior in this repo**

- **`vertex_stubs.ensure_vertexai_stub_modules()`** installs lightweight modules on `sys.modules` when the real SDK is not importable, so application code can use **`from vertexai.language_models import …`** locally and in CI.
- **`gcp_mocks.py`** defines classes with the **method shapes** used here (`from_pretrained`, `get_embeddings`, `generate_content`, response `.text`).
- **`tests/test_gcp_sdk_mocking.py`** includes **`unittest.mock.patch("vertexai.language_models.GenerativeModel", …)`** to demonstrate literal-path patching.

This is **API-shape compliance** for a local take-home; production would call real credentials and quotas.

---

## 7. Benchmark methodology (Strategy A vs B)

### 7.1 What we report

- Per query: top‑k hits for A and B (chunk id, score, preview, metadata).
- **`overlap_count`** — how many of the top‑3 chunk ids appear in both result sets.
- **`expansion_changed`** — `false` when the mock returned no expansion and the engine fell back to the original query (so A and B may coincide).
- **`top1_score_delta_b_minus_a`** — signed difference of rank‑1 scores; **not** a recall metric.
- **`notes`** — neutral commentary (low-score warnings, rank‑1 match vs differ), explicitly stating that cross-strategy cosine magnitudes are not comparable as “winner/loser.”

### 7.2 What we deliberately do **not** claim

There is **no `winner` field** in benchmark JSON. We do **not** claim “Strategy B improved recall” from higher cosine alone: B’s query string often has more lexical overlap with the corpus by construction, which inflates similarity **without** labeled relevance.

For rigorous evaluation you would need judged (query, relevant_chunk) pairs and metrics such as MRR/nDCG — out of scope for the bundled corpus size.

---

## 8. Production migration (sketch)

Keep **`BaseEmbedder`**, **`BaseVectorStore`**, **`Retriever`**, and expansion behind a **`generate_content`-like** interface; replace implementations.

### 8.1 Embeddings

Use **`TextEmbeddingModel.from_pretrained("textembedding-gecko@003")`** (or current id) with **`TextEmbeddingInput`** and **`task_type`**: e.g. **`RETRIEVAL_DOCUMENT`** for corpus batches and **`RETRIEVAL_QUERY`** for user queries (asymmetric behavior analogous to “passage vs query” locally). Batch to respect **per-request input limits**.

### 8.2 Vector store → Vertex AI Vector Search

- **Ingest:** export vectors + ids (and optional sparse features) to the format your index expects (often **JSONL** in **GCS**), create/update an index with a distance measure consistent with your normalization (**`COSINE_DISTANCE`** / **`DOT_PRODUCT_DISTANCE`**).
- **Serve:** deploy an **`IndexEndpoint`**, query with **`MatchServiceClient.find_neighbors`** (or REST).
- **Payload:** the service returns **ids and distances**; store full text and business metadata in **Firestore**, **BigQuery**, or **GCS** and hydrate in the retriever after search.

### 8.3 Query expansion

Call **`vertexai.generative_models.GenerativeModel`** (e.g. Gemini) with timeouts and **fallback to the original query** on failure so retrieval remains available.

### 8.4 Operations

Plan for **IAM** (Workload Identity), **quotas**, **Logging/Monitoring**, and optional **VPC-SC** / **CMEK** for regulated environments.

Further narrative lives in **`context.md` §14** and the **Production migration** section of **`README.md`**.

---

## 9. Operational artifacts

| Artifact | Meaning |
|----------|---------|
| `storage/` | Default persistence for benchmark JSON and vector index paths from config (often gitignored). |
| `output/<RUN_ID>/` | Orchestrator and `query_cli.py` runs: `manifest.json`, `config.snapshot.yaml`, per-step logs/JSON, `summary.json`; interactive runs add `queries.jsonl`. |
| `retrieval_benchmark.md` | Human-readable benchmark tables; regenerate via orchestrator `--write-benchmark-md`. |
| `data/cache/` | Optional Gutenberg caches for `query_cli.py` (gitignored). |

---

## 10. Limitations and known tradeoffs

- **Corpus size** — Eleven paragraphs are enough to demonstrate the pipeline, not to draw strong conclusions about rank‑2/3 noise or tail recall.
- **Mock expansion** — Deterministic string rules plus fallback to the original query; flagged via **`expansion_changed`** when unchanged.
- **No reranking / hybrid** — Production systems often add cross-encoder rerank or BM25+dense fusion; not required for the baseline assessment.
- **Multiple backends** — FAISS variants, Chroma, and NumPy increase surface area for tests and demos; default config should stay the path you actually run for submission.

---

## 11. Where to read next

| Document | Use when you need… |
|----------|-------------------|
| **`README.md`** | Setup, CLI commands, repository layout, quick links. |
| **`context.md`** | Full specification: chunking variants, FAISS index policy, extended YAML examples, interface sketches. |
| **`config/config.yaml`** | The live tunables with inline comments. |

---

*Last rewritten: 2026-05-15 — decision-log style; replaces the previous spec-length duplicate of `context.md`.*
