#!/usr/bin/env python3
"""
Interactive RAG on a **real** long document: ``sentence_transformers`` embeddings + configured
vector store (default FAISS HNSW). Prompts only for **queries** on stdin; logs go to stderr.

Each run writes **``output/<RUN_ID>/``** with the same style of artifacts as ``orchestrator.py``:
``manifest.json``, ``config.snapshot.yaml``, per-step JSON, ``queries.jsonl``, ``summary.json``.

**Corpus** (pick one):

- ``--doc PATH`` — one UTF-8 ``.txt`` / ``.md`` file (whole file = one logical document).
- ``--fetch-sample [BOOK_ID]`` — download a large **Project Gutenberg** plain text (no argument =
  random among built-in IDs: 11, 1342, 2701).

Examples::

  python3 query_cli.py --fetch-sample
  python3 query_cli.py --doc ./my_notes.md
  python3 query_cli.py --fetch-sample 2701 --single "What is Moby Dick?"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# Plain UTF-8 mirrors on Project Gutenberg (IDs match filenames).
GUTENBERG_TXT: dict[int, str] = {
    11: "https://www.gutenberg.org/files/11/11-0.txt",
    1342: "https://www.gutenberg.org/files/1342/1342-0.txt",
    2701: "https://www.gutenberg.org/files/2701/2701-0.txt",
}


def _log(run_id: str, message: str) -> None:
    print(f"[{run_id}] {message}", file=sys.stderr, flush=True)


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def _prepare_run_dir(run_id: str, argv_repr: str, plan: list[str]) -> Path:
    d = ROOT / "output" / run_id
    d.mkdir(parents=True, exist_ok=True)
    cfg = ROOT / "config" / "config.yaml"
    if cfg.is_file():
        shutil.copy2(cfg, d / "config.snapshot.yaml")
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "mode": "interactive_query_cli",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "argv": argv_repr,
                "planned_steps": plan,
                "output_dir": str(d.resolve()),
                "repo_root": str(ROOT.resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return d


def _write_summary(run_dir: Path, run_id: str, status: str, steps: list[dict]) -> None:
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": status,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "steps": steps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _ensure_src_path() -> None:
    s = str(SRC)
    if s not in sys.path:
        sys.path.insert(0, s)


def _isolated_config_for_run(cfg: Any, run_dir: Path) -> Any:
    """Avoid clobbering repo ``storage/`` — keep indexes under this run directory."""
    n = cfg.model_copy(deep=True)
    n.vector_store.persist_path = run_dir / "vector_index"
    n.vector_store.chroma.persist_directory = run_dir / "chroma"
    return n


def _file_sha256_head(path: Path, max_bytes: int = 1_000_000) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()


def _hits_to_json(hits: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for h in hits:
        out.append(
            {
                "rank": h.rank,
                "chunk_id": h.chunk_id,
                "score": h.score,
                "text_preview": h.text[:500],
                "metadata": dict(h.metadata),
            }
        )
    return out


def _fetch_gutenberg(book_id: int, dest: Path, run_id: str) -> None:
    url = GUTENBERG_TXT.get(book_id)
    if not url:
        known = ", ".join(str(k) for k in sorted(GUTENBERG_TXT))
        raise SystemExit(f"Unknown Gutenberg id {book_id}. Use one of: {known}")
    if dest.is_file() and dest.stat().st_size > 4096:
        _log(run_id, f"reusing cached sample ({dest.stat().st_size} bytes): {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    _log(run_id, f"downloading book {book_id}: {url}")
    req = Request(
        url,
        headers={"User-Agent": "TELEPORT-query-cli/1.0 (local RAG educational demo)"},
    )
    try:
        with urlopen(req, timeout=240) as resp:
            data = resp.read()
    except HTTPError as e:
        raise SystemExit(f"HTTP error fetching Gutenberg text: {e}") from e
    except URLError as e:
        raise SystemExit(f"Network error fetching Gutenberg text: {e}") from e
    dest.write_bytes(data)
    _log(run_id, f"saved {len(data)} bytes -> {dest}")


def _resolve_corpus_path(args: argparse.Namespace, run_id: str) -> Path:
    if args.doc is not None:
        p = args.doc.expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"Not a file: {p}")
        return p
    # --fetch-sample
    raw = args.fetch_sample
    if raw in (None, "random"):
        book_id = random.choice(list(GUTENBERG_TXT))
    else:
        try:
            book_id = int(raw)
        except ValueError as e:
            raise SystemExit(f"BOOK_ID must be an integer, got {raw!r}") from e
    dest = ROOT / "data" / "cache" / f"gutenberg_{book_id}.txt"
    _fetch_gutenberg(book_id, dest, run_id)
    return dest.resolve()


def _print_results(query: str, raw: list[Any], expanded: Any) -> None:
    print("\n" + "=" * 72)
    print(f"Query: {query}")
    print("-" * 72)
    print("Strategy A (raw embedding)")
    for h in raw:
        prev = h.text.replace("\n", " ")[:220]
        print(f"  [{h.rank}] {h.chunk_id}  score={h.score:.4f}")
        print(f"       {prev}{'…' if len(h.text) > 220 else ''}")
    print("-" * 72)
    print("Strategy B (expanded query)")
    print(f"  Expanded: {expanded.expanded_query[:300]}{'…' if len(expanded.expanded_query) > 300 else ''}")
    for h in expanded.results:
        prev = h.text.replace("\n", " ")[:220]
        print(f"  [{h.rank}] {h.chunk_id}  score={h.score:.4f}")
        print(f"       {prev}{'…' if len(h.text) > 220 else ''}")
    print("=" * 72 + "\n")


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    run_id = _new_run_id()
    argv_repr = " ".join(argv) if argv else "(none)"

    parser = argparse.ArgumentParser(
        description="Interactive Strategy A/B retrieval on a real long document.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--doc",
        type=Path,
        metavar="PATH",
        help="UTF-8 text or markdown file to ingest as one document.",
    )
    src.add_argument(
        "--fetch-sample",
        nargs="?",
        const="random",
        metavar="BOOK_ID",
        help="Download a large Gutenberg .txt (optional id; default = random among 11, 1342, 2701).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "config.yaml",
        help="Engine YAML (default: config/config.yaml).",
    )
    parser.add_argument(
        "--single",
        metavar="QUERY",
        help="Run one query non-interactively, then exit.",
    )
    args = parser.parse_args(argv)
    if args.single is not None and not args.single.strip():
        raise SystemExit("--single requires a non-empty query string.")

    planned = ["resolve_corpus", "engine_init", "ingest", "interactive_queries"]
    run_dir = _prepare_run_dir(run_id, argv_repr, planned)
    step_records: list[dict[str, Any]] = []

    _log(run_id, f"artifacts -> {run_dir.resolve()}")

    try:
        corpus_path = _resolve_corpus_path(args, run_id)
        doc_meta = {
            "corpus_path": str(corpus_path),
            "byte_length": corpus_path.stat().st_size,
            "sha256_first_1mb": _file_sha256_head(corpus_path),
        }
        (run_dir / "step_resolve_corpus.json").write_text(
            json.dumps(doc_meta, indent=2),
            encoding="utf-8",
        )
        step_records.append(
            {
                "step": "resolve_corpus",
                "index": 1,
                "exit_code": 0,
                "duration_ms": 0.0,
                "artifacts": ["step_resolve_corpus.json"],
            }
        )

        _ensure_src_path()
        from rag_engine.app import RAGEngine
        from rag_engine.config.loader import load_engine_config
        from rag_engine.documents.loader import load_documents_plaintext

        cfg = load_engine_config(args.config)
        cfg = _isolated_config_for_run(cfg, run_dir)

        t_engine = time.perf_counter()
        engine = RAGEngine(config=cfg)
        load_ms = (time.perf_counter() - t_engine) * 1000
        engine_info = {
            "embedding_provider": cfg.embedding.provider,
            "embedding_model": cfg.embedding.model_name,
            "vector_store": cfg.vector_store.provider,
            "load_engine_ms": round(load_ms, 2),
        }
        (run_dir / "step_engine_init.json").write_text(
            json.dumps(engine_info, indent=2),
            encoding="utf-8",
        )
        step_records.append(
            {
                "step": "engine_init",
                "index": 2,
                "exit_code": 0,
                "duration_ms": round(load_ms, 2),
                "artifacts": ["step_engine_init.json"],
            }
        )

        _log(run_id, f"ingesting {corpus_path} (real embeddings; may take a while on first model load)...")
        t0 = time.perf_counter()
        docs = load_documents_plaintext(corpus_path)
        n_chunks = engine.ingest_documents(docs)
        ingest_ms = (time.perf_counter() - t0) * 1000
        ingest_payload = {
            "documents": len(docs),
            "chunks_indexed": n_chunks,
            "duration_ms": round(ingest_ms, 2),
            "document_ids": [d.document_id for d in docs],
        }
        (run_dir / "step_ingest.json").write_text(
            json.dumps(ingest_payload, indent=2),
            encoding="utf-8",
        )
        step_records.append(
            {
                "step": "ingest",
                "index": 3,
                "exit_code": 0,
                "duration_ms": round(ingest_ms, 2),
                "artifacts": ["step_ingest.json"],
            }
        )
        _log(run_id, f"ingest done: {n_chunks} chunks indexed in {ingest_ms:.0f} ms")

        queries_path = run_dir / "queries.jsonl"
        queries_path.write_text("", encoding="utf-8")
        query_count = 0

        def run_one_query(q: str) -> None:
            nonlocal query_count
            raw = engine.search_raw(q)
            exp = engine.search_expanded(q)
            _print_results(q, raw, exp)
            rec = {
                "query": q,
                "strategy_a": _hits_to_json(raw),
                "strategy_b": {
                    "expanded_query": exp.expanded_query,
                    "hits": _hits_to_json(exp.results),
                },
                "queried_at": datetime.now(timezone.utc).isoformat(),
            }
            with queries_path.open("a", encoding="utf-8") as jf:
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            query_count += 1

        if args.single:
            run_one_query(args.single.strip())
        else:
            print(
                f"Corpus ready ({n_chunks} chunks). Enter a query per line; empty line exits.\n"
                f"Artifacts: {run_dir.resolve()}",
                flush=True,
            )
            while True:
                try:
                    line = input("query> ")
                except EOFError:
                    break
                q = line.strip()
                if not q:
                    break
                run_one_query(q)

        step_records.append(
            {
                "step": "interactive_queries",
                "index": 4,
                "exit_code": 0,
                "duration_ms": 0.0,
                "artifacts": ["queries.jsonl"],
                "query_count": query_count,
            }
        )
        _write_summary(run_dir, run_id, "success", step_records)
        _log(run_id, f"RUN COMPLETE  queries={query_count}  dir={run_dir.resolve()}")
        return 0
    except SystemExit as e:
        ec = e.code
        code = ec if isinstance(ec, int) else 1
        step_records.append(
            {
                "step": "fatal",
                "index": len(step_records) + 1,
                "exit_code": code,
                "message": str(e),
            }
        )
        _write_summary(run_dir, run_id, "error", step_records)
        raise
    except Exception as e:
        step_records.append(
            {
                "step": "fatal",
                "index": len(step_records) + 1,
                "exit_code": 1,
                "message": repr(e),
            }
        )
        _write_summary(run_dir, run_id, "error", step_records)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
