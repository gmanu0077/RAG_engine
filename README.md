# Semantic RAG assessment (see `DOCUMENTATION.md` for design decisions, `context.md` for full spec)

Local **context-aware retrieval engine**: everything is driven from **`config/config.yaml`**. The stack includes config-driven **chunking**, **embeddings** (`sentence_transformers` or **`mock_vertex`** for fast tests), **vector stores** (**FAISS** with Flat / **HNSW** / IVF / IVF-PQ + **auto** policy, **Chroma**, or **NumPy** exact baseline), and two retrieval strategies:

| Strategy | Flow |
|----------|------|
| **A — raw vector** | User query → embed → vector search → `top_k` chunks. |
| **B — query expansion** | Mock generative model expands query → embed → search → `top_k` chunks. |

Benchmarks emit **JSON** (per-strategy hits, overlap, score deltas with caveats, per-hit previews) and optional **Markdown** reports.

### Entrypoints (what to run)

| What | Role |
|------|------|
| **`src/rag_engine/app.py`** (`RAGEngine`) | **Library** — load config, ingest, `search_raw` / `search_expanded`. Import from tests or scripts; it does not run the multi-step shell pipeline. |
| **`orchestrator.py`** | **Assessment runner** — pytest, benchmark, smoke, markdown export. Each real run creates **`output/<RUN_ID>/`** with `manifest.json`, `config.snapshot.yaml`, per-step logs/JSON, and **`summary.json`**. |
| **`query_cli.py`** | **Interactive demo** — ingest one large **`.txt` / `.md`** (or a **Gutenberg** download), then prompt only for **queries**. Same style of artifacts under **`output/<RUN_ID>/`** (plus **`queries.jsonl`**); session index lives under that folder so **`storage/`** is left alone. |
| **Vertex stubs** | When `google-cloud-aiplatform` is not installed, **`rag_engine.vertex_stubs`** registers minimal **`vertexai.language_models`** modules so code and tests use the **literal SDK import path** named in the brief (`TextEmbeddingModel`, `GenerativeModel`). |

---

## Repository layout

```text
context.md                         # full architecture + config spec (long reference)
DOCUMENTATION.md                   # design log: decisions, methodology, Vertex migration sketch
config/config.yaml              # tunables (comments describe options + good defaults)
data/technical_paragraphs.json  # sample JSON list (objects with `text`, optional `id`)
data/cache/                     # Gutenberg samples for query_cli.py (gitignored)
storage/                        # default benchmark JSON + vector persistence (gitignored)
output/                         # per-run artifacts: orchestrator + query_cli (gitignored)
orchestrator.py                  # CLI: pytest, benchmark, smoke, markdown export
query_cli.py                     # interactive queries on a real long .txt/.md or Gutenberg sample
scripts/run_benchmark.py        # ingest + benchmark JSON (+ Rich table on stderr)
src/rag_engine/
  app.py                        # RAGEngine: ingest, search_raw, search_expanded
  gcp_mocks.py                  # Vertex-shaped TextEmbeddingModel / GenerativeModel (local)
  vertex_stubs.py               # installs vertexai.language_models when SDK absent
  canned_query_expansions.py    # deterministic mock expansions (no import cycle via retrieval/)
  config/                       # Pydantic schema + YAML loader
  documents/                    # JSON → Document / Chunk models
  chunking/                     # recursive | fixed_character | fixed_token | sentence | semantic
  embeddings/                   # SentenceTransformerEmbedder, MockVertexEmbedder
  vectorstores/                 # FaissVectorStore, ChromaVectorStore, NumpyVectorStore
  retrieval/                    # QueryExpander, Retriever, result types
  evaluation/                   # benchmark runner, metrics, JSON/Markdown reporter
tests/                          # chunking, embeddings, stores, retrieval, benchmark I/O
pyproject.toml                  # dependencies; pytest `pythonpath = ["src"]`
```

---

## Prerequisites

- **Python 3.11+**
- A virtualenv is strongly recommended (sentence-transformers / torch are heavy on first install).

---

## Setup

```bash
cd /path/to/TELEPORT
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional editable install + dev tools:
pip install -e ".[dev]"
pytest
```

Ensure `src` is on `PYTHONPATH` when running modules by hand (or use `pip install -e .`). **`pyproject.toml`** already sets `pythonpath = ["src"]` for **pytest**.

---

## Run everything (recommended order)

From the repository root, after **Setup** (venv + `pip install`):

```bash
cd /path/to/TELEPORT
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 1) Discover orchestrator steps (optional)
python3 orchestrator.py --list-steps

# 2) Fast smoke: mock embeddings + NumPy store + ingest counts
python3 orchestrator.py --smoke

# 3) Unit tests
python3 orchestrator.py --pytest
# equivalent: pytest

# 4) Full benchmark (downloads sentence-transformers model on first run; uses config/config.yaml)
python3 orchestrator.py --benchmark
# JSON on stdout — save with:  python3 orchestrator.py --benchmark > storage/benchmark_latest.json 2>run.log

# 5) Regenerate markdown report + write JSON snapshot (see config benchmark.output_* paths)
python3 orchestrator.py --write-benchmark-md --output retrieval_benchmark.md

# Or run steps 3–4 in one go:
python3 orchestrator.py --steps pytest,benchmark

# Or the default “CI-style” pair (pytest then benchmark only):
python3 orchestrator.py --all

# All registered orchestrator steps in one run (pytest + benchmark + smoke + markdown):
python3 orchestrator.py --steps pytest,benchmark,smoke,write-benchmark-md
```

### Orchestrator steps

| Step name | Command | What it does |
|-----------|---------|----------------|
| `list-steps` | `python3 orchestrator.py --list-steps` | Prints registered step names. |
| `smoke` | `python3 orchestrator.py --smoke` | Ingest sample corpus with **mock** embeddings + **NumPy** index. |
| `pytest` | `python3 orchestrator.py --pytest` | Runs `tests/` (quiet `-q` unless you pass args after `--`). |
| `benchmark` | `python3 orchestrator.py --benchmark` | Ingest + Strategy A vs B; JSON to **stdout**, logs + optional Rich table to **stderr**. |
| `write-benchmark-md` | `python3 orchestrator.py --write-benchmark-md --output retrieval_benchmark.md` | Writes markdown to `--output` and JSON to `benchmark.output_json` in config. |

### Interactive retrieval (`query_cli.py`)

Use this when you want a **single large plaintext corpus** (your file or a **Project Gutenberg** mirror) and the shell should only ask for **queries** after ingest. It reads **`config/config.yaml`** (real **sentence-transformers** + **FAISS** by default). Indexes for that session are under **`output/<RUN_ID>/vector_index/`** so the repo default **`storage/`** directory is not overwritten.

**Run artifacts** (same pattern as the orchestrator): **`manifest.json`**, **`config.snapshot.yaml`**, **`step_resolve_corpus.json`**, **`step_engine_init.json`**, **`step_ingest.json`**, **`queries.jsonl`** (one JSON record per query with Strategy A and B hits), **`summary.json`**.

```bash
# Large public-domain book (cached under data/cache/, gitignored); random ID if omitted
python3 query_cli.py --fetch-sample
python3 query_cli.py --fetch-sample 2701

# Your UTF-8 file (.txt / .md — whole file = one document)
python3 query_cli.py --doc ./notes.md

# Optional: alternate engine YAML, one-shot query (no interactive loop)
python3 query_cli.py --config config/config.yaml --doc README.md --single "What is FAISS?"
```

---

## CLI reference

### Orchestrator (`orchestrator.py`)

Logs go to **stderr**; benchmark JSON goes to **stdout** (pipe-friendly).

```bash
python3 orchestrator.py --list-steps
python3 orchestrator.py --smoke
python3 orchestrator.py --benchmark              # may download MiniLM on first run
python3 orchestrator.py --benchmark --no-rich    # skip Rich table on stderr
python3 orchestrator.py --write-benchmark-md --output retrieval_benchmark.md
python3 orchestrator.py --all                     # pytest then benchmark only
python3 orchestrator.py --steps pytest,benchmark,smoke,write-benchmark-md
python3 orchestrator.py --pytest -- -k chroma -x   # extra pytest args after --
```

### Interactive CLI (`query_cli.py`)

```bash
python3 query_cli.py --doc PATH              # mutually exclusive with --fetch-sample
python3 query_cli.py --fetch-sample [ID]     # Gutenberg: 11, 1342, 2701; omit ID for random
python3 query_cli.py --doc notes.md --config config/config.yaml --single "your question"
```

### Standalone benchmark script

```bash
python3 scripts/run_benchmark.py
```

Runs ingest + Strategy A vs B and prints JSON to stdout; a Rich summary table is printed to stderr.

---

## Create a GitHub repository and push

Replace **`YOUR_GITHUB_USERNAME`** and **`YOUR_REPO_NAME`** (and adjust `--public` / `--private`).

### Option A — GitHub CLI (`gh`, recommended)

Install the CLI: [https://cli.github.com/](https://cli.github.com/)

```bash
gh auth login
cd /path/to/TELEPORT

# If this folder is not a git repo yet:
git init
git add -A
git status   # confirm .venv, storage, __pycache__ are NOT staged (they should be gitignored)
git commit -m "Initial commit: context-aware RAG engine"

# Create repo on GitHub and push in one step (pick one):
gh repo create YOUR_GITHUB_USERNAME/YOUR_REPO_NAME --public --source=. --remote=origin --push
# gh repo create YOUR_GITHUB_USERNAME/YOUR_REPO_NAME --private --source=. --remote=origin --push
```

If the repo **already exists** on GitHub (empty) and you only need to connect and push:

```bash
cd /path/to/TELEPORT
git init   # skip if already a repo
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
# or SSH:  git remote add origin git@github.com:YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
git add -A
git commit -m "Initial commit"   # skip if you already have commits
git push -u origin main
```

### Option B — Web UI only (no `gh`)

1. On GitHub: **New repository** → name **`YOUR_REPO_NAME`** → create (no README if you already have one locally).
2. Locally:

```bash
cd /path/to/TELEPORT
git init
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
# or SSH:  git remote add origin git@github.com:YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
git add -A
git commit -m "Initial commit: context-aware RAG engine"
git push -u origin main
```

### If Git reports “nothing to commit”

You may already be synced, or everything is ignored. Check with `git status`. To amend and push later: `git add -A && git commit -m "..." && git push`.

---

## Configuration

Edit **`config/config.yaml`**. Inline comments explain each block and point to sensible defaults aligned with **`context.md`** (e.g. **recursive** chunking, **cosine** similarity, **`faiss` + `hnsw`**, `sentence-transformers/all-MiniLM-L6-v2`).

Notable keys:

- **`data.input_path`** — JSON list of records for **`RAGEngine.ingest()`** / orchestrator benchmark; each record must include **`data.text_field`** (default `text`). **`query_cli.py`** does not use this path; it loads a **plaintext** file via **`load_documents_plaintext()`** or a cached Gutenberg file.
- **`vector_store.provider`** — `faiss` | `chroma` | `numpy`.
- **`vector_store.faiss`** — `index_type` (`flat`, `hnsw`, `ivf_flat`, `ivf_pq`, `auto`) and `index_selection_policy` (`manual` | `auto`); see `context.md` §9.
- **`retrieval.top_k` / `retrieval.fetch_k`** — over-fetch then trim to `top_k`.
- **`benchmark.output_json`** / **`benchmark.output_markdown`** — default `storage/benchmark_results.json` and `retrieval_benchmark.md` (override `--output` only affects the markdown step in the orchestrator).

---

## Programmatic use

```python
from pathlib import Path
from rag_engine.app import RAGEngine
from rag_engine.config.loader import load_engine_config

cfg = load_engine_config(Path("config/config.yaml"))
engine = RAGEngine(config=cfg)
n = engine.ingest(Path(cfg.data.input_path))
hits = engine.search_raw("How does the system handle peak load?")
print(n, [h.chunk_id for h in hits])
```

Ingest a plaintext file (one document) without temp JSON:

```python
from rag_engine.documents.loader import load_documents_plaintext

docs = load_documents_plaintext("notes.md")
n = engine.ingest_documents(docs)
```

For fast local experiments without downloading models, set in YAML (or mutate the loaded config before constructing `RAGEngine`):

- `embedding.provider: mock_vertex`
- `vector_store.provider: numpy`

The orchestrator **smoke** step does exactly that.

---

## Similarity & FAISS

### Why cosine (or inner product on normalized vectors) for text

Semantic text models are trained with **cosine-style contrastive objectives**: what matters is the **direction** of the embedding (which topics / phrases it aligns with), not the raw **magnitude** (length can grow with text length and surface form). **Cosine similarity** measures alignment of directions. **Euclidean distance on the same vectors** mixes direction and length, so it is usually a poor default for general text retrieval with models like MiniLM unless the model was explicitly trained for L2 geometry.

### When Euclidean can make sense

**Euclidean** is appropriate when the embedding space was trained or calibrated for **L2 metric learning** (some image, audio, or multimodal encoders). For off-the-shelf sentence transformers used as text embedders, prefer cosine / normalized inner product unless you have evidence the checkpoint favors L2.

### How this repo maps metrics to FAISS

For **`similarity.metric`** **`cosine`** or **`inner_product`**, chunk and query vectors are **L2-normalized** and FAISS uses an **inner product** index (`IndexFlatIP` / `IndexHNSWFlat` with inner product). That is **numerically cosine similarity** on the unit sphere and is efficient in FAISS. For **`euclidean`**, FAISS uses an **L2** index and the normalization path described in `config.yaml` / `context.md` is disabled so geometry stays consistent.

### Reading scores

Similarities are typically in **[-1, 1]** for cosine-like scores on this stack. The config’s **`no_match_cosine_threshold`** is a design guard for very low top scores (e.g. surfacing “no good hit” in a product). **Strategy A and Strategy B embed different query strings**, so **raw top-1 scores are not directly comparable** as a “quality” measure between strategies — compare **chunk ids**, **overlap**, and **qualitative previews** instead (see benchmark JSON `comparison.notes` and `top1_score_delta_b_minus_a`, which is informational only).

---

## Outputs

### Repo-root files (config defaults)

- **`retrieval_benchmark.md`** — human-readable tables (or path passed to `--output` on `--write-benchmark-md`).
- **`storage/benchmark_results.json`** — machine-readable rows (strategy A/B blocks + `comparison`: `overlap_count`, signed **`top1_score_delta_b_minus_a`**, **`notes`**; strategy B includes **`expansion_changed`** when the mock returned text different from the original query). There is **no `winner` field** — the brief asks for a structured comparison, not a declared winner from incomparable cosines.

### Per-run directory: `output/<RUN_ID>/`

**`RUN_ID`** is a UTC timestamp plus a short hex suffix (example: `20260514T183023Z-c50fd910`). Both **`orchestrator.py`** and **`query_cli.py`** create this folder at the **repository root** and write:

| Artifact | Purpose |
|----------|---------|
| **`manifest.json`** | `argv`, planned steps, paths, `started_at`. |
| **`config.snapshot.yaml`** | Copy of `config/config.yaml` at run start. |
| **`summary.json`** | Final status and per-step list (exit codes, durations, artifact filenames). |
| **`step_*.json`** / **`.log`** / **`.md`** | Step outputs (e.g. `step_pytest.log`, `step_benchmark.json`, `step_smoke.json`, or query-cli `step_ingest.json`). |
| **`queries.jsonl`** | **`query_cli.py` only** — one JSON line per user query (Strategy A hits + Strategy B expanded query and hits). |

The orchestrator also prints the resolved artifact path on **stderr** when a run finishes successfully.

---

## Production migration

Keep the same **interfaces** (`BaseEmbedder`, `BaseVectorStore`, `Retriever`, query expansion); swap implementations for Vertex.

### 1. Embeddings (Gecko-style)

Use **`vertexai.language_models.TextEmbeddingModel.from_pretrained("textembedding-gecko@003")`** (or current model id). Call **`get_embeddings`** with **`TextEmbeddingInput`** and **`task_type`**: e.g. **`RETRIEVAL_DOCUMENT`** when embedding corpus chunks, **`RETRIEVAL_QUERY`** when embedding the user query (asymmetric encoding, similar in spirit to this repo’s Strategy A vs expanded Strategy B). Respect batch limits (e.g. 250 inputs per request) by batching in the adapter.

### 2. Vector store → Vertex AI Vector Search

- **Ingest:** export chunk ids + vectors + optional sparse vectors to the format your index expects (often **JSONL** in **Cloud Storage**), then create/update an index (`gcloud ai indexes` / client libraries) with **`COSINE_DISTANCE`** or **`DOT_PRODUCT_DISTANCE`** aligned to your normalization choice.
- **Serve:** deploy an **`IndexEndpoint`**, then at query time call **`MatchServiceClient.find_neighbors`** (or the REST equivalent) with the query vector and `neighbor_count`.
- **Payload:** Vector Search returns **ids and distances** — full chunk text and business metadata live elsewhere (**Firestore**, **BigQuery**, **GCS**). The retriever hydrates text from your metadata store using returned ids (same pattern as local FAISS + in-memory chunk list, but with a network hop).

### 3. Query expansion

Wrap **`vertexai.generative_models.GenerativeModel`** (e.g. Gemini) behind the same **`generate_content(prompt)`** surface your expander already expects; add **timeouts** and **fallback to the original query** on failure so retrieval stays available.

### 4. Operations (short checklist)

**IAM** (service account / Workload Identity), **quotas** (embedding batch size, QPS on the endpoint), **Cloud Logging / Monitoring** on the index and endpoint, and optional **VPC-SC** / **CMEK** for regulated environments.

More background: **`context.md` §14** (architecture narrative).

---

## Limitations

- Small sample corpus; not a substitute for labeled relevance sets.
- Strategy **B** uses **deterministic mock** expansions (`vertexai.language_models`-shaped stubs when the SDK is not installed); unknown queries **fall back** to the original text — benchmark JSON marks this with **`expansion_changed: false`** when it happens.
- **Auto** FAISS index selection follows **`context.md` §9** when `faiss.index_selection_policy` is **`auto`**.
- Benchmark output **does not** pick a “winner” from cosine scores: Strategy A and B use **different embedded strings**, so magnitudes are not directly comparable as recall.
