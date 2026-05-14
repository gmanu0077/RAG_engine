# Semantic RAG assessment (see `context.md`)

Local **context-aware retrieval engine**: everything is driven from **`config/config.yaml`**. The stack includes config-driven **chunking**, **embeddings** (`sentence_transformers` or **`mock_vertex`** for fast tests), **vector stores** (**FAISS** with Flat / **HNSW** / IVF / IVF-PQ + **auto** policy, **Chroma**, or **NumPy** exact baseline), and two retrieval strategies:

| Strategy | Flow |
|----------|------|
| **A — raw vector** | User query → embed → vector search → `top_k` chunks. |
| **B — query expansion** | Mock generative model expands query → embed → search → `top_k` chunks. |

Benchmarks emit **JSON** (overlap, winner heuristic, per-hit previews) and optional **Markdown** reports.

---

## Repository layout

```text
context.md                         # full architecture + config spec
config/config.yaml              # tunables (comments describe options + good defaults)
data/technical_paragraphs.json  # sample JSON list (objects with `text`, optional `id`)
storage/                        # default benchmark JSON + vector persistence (gitignored)
orchestrator.py                  # CLI: pytest, benchmark, smoke, markdown export
scripts/run_benchmark.py        # ingest + benchmark JSON (+ Rich table on stderr)
src/rag_engine/
  app.py                        # RAGEngine: ingest, search_raw, search_expanded
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

# Or the default “CI-style” pair (pytest then benchmark):
python3 orchestrator.py --all
```

| Step name | Command | What it does |
|-----------|---------|----------------|
| `list-steps` | `python3 orchestrator.py --list-steps` | Prints registered step names. |
| `smoke` | `python3 orchestrator.py --smoke` | Ingest sample corpus with **mock** embeddings + **NumPy** index. |
| `pytest` | `python3 orchestrator.py --pytest` | Runs `tests/` (quiet `-q` unless you pass args after `--`). |
| `benchmark` | `python3 orchestrator.py --benchmark` | Ingest + Strategy A vs B; JSON to **stdout**, logs + optional Rich table to **stderr**. |
| `write-benchmark-md` | `python3 orchestrator.py --write-benchmark-md --output retrieval_benchmark.md` | Writes markdown to `--output` and JSON to `benchmark.output_json` in config. |

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
python3 orchestrator.py --all                     # pytest then benchmark
python3 orchestrator.py --steps pytest,benchmark
python3 orchestrator.py --pytest -- -k chroma -x   # extra pytest args after --
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

- **`data.input_path`** — JSON list of records; each must include **`data.text_field`** (default `text`).
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

For fast local experiments without downloading models, set in YAML (or mutate the loaded config before constructing `RAGEngine`):

- `embedding.provider: mock_vertex`
- `vector_store.provider: numpy`

The orchestrator **smoke** step does exactly that.

---

## Similarity & FAISS

For **`similarity.metric`** `cosine` or `inner_product`, embeddings used with FAISS are **L2-normalized** and indexed with **inner product** (equivalent to cosine on the sphere). For **`euclidean`**, FAISS uses an **L2** index without that cosine-style normalization path.

---

## Outputs

- **`retrieval_benchmark.md`** — human-readable tables (or path passed to `--output` on `--write-benchmark-md`).
- **`storage/benchmark_results.json`** — machine-readable rows (strategy A/B blocks + `comparison`: `overlap_count`, `winner`, `reason`), written when the markdown export step runs.

---

## Production migration

See **`context.md` §14**: keep chunking and retrieval abstractions; swap **`SentenceTransformerEmbedder`** for Vertex embeddings and the vector store for **Vertex AI Vector Search**; hydrate chunk text by returned IDs from your metadata store.

---

## Limitations

- Small sample corpus; not a substitute for labeled relevance sets.
- Strategy **B** uses **deterministic mock** expansions, not a live Gemini / GPT.
- **Auto** FAISS index selection follows **`context.md` §9** when `faiss.index_selection_policy` is **`auto`**.
