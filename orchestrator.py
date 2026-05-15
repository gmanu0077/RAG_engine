#!/usr/bin/env python3
"""
Orchestrator for the RAG assessment repo: run tests, benchmarks, smoke checks,
and optional benchmark markdown export in a single entrypoint.

Each run gets a ``RUN_ID`` (UTC timestamp + short hex). Artifacts for every executed
step are written under **``output/<RUN_ID>/``** at the repo root (logs, JSON, config
snapshot, final ``summary.json``). **Orchestrator logs always go to stderr** (no flag
to disable them). ``--no-rich`` only skips the optional Rich table after a benchmark.

Benchmark JSON is still printed to **stdout** for piping; a copy is saved under
``output/<RUN_ID>/step_benchmark.json``.

Examples (use ``python3`` on macOS if ``python`` is not installed):
  python3 orchestrator.py --list-steps
  python3 orchestrator.py --all
  python3 orchestrator.py --pytest
  python3 orchestrator.py --benchmark --no-rich
  python3 orchestrator.py --steps pytest,benchmark
  python3 orchestrator.py --write-benchmark-md --output retrieval_benchmark.md
  python3 orchestrator.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# Ordered registry: name -> one-line description
STEP_DESCRIPTIONS: dict[str, str] = {
    "pytest": "Run pytest (uses pyproject.toml / pytest config).",
    "benchmark": "Ingest default corpus, run Strategy A vs B benchmark, print JSON to stdout.",
    "smoke": "Load default JSON, chunk, embed (mock vectors), ingest; print counts.",
    "write-benchmark-md": "Run benchmark and rewrite markdown report (see --output).",
}


def _log(run_id: str, message: str, *, out: bool = False) -> None:
    """Orchestrator logs on stderr so stdout stays clean for piped JSON."""
    stream = sys.stdout if out else sys.stderr
    print(f"[{run_id}] {message}", file=stream, flush=True)


def _banner(char: str = "-", width: int = 76) -> str:
    return char * width


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def _emit_orchestrator_header(run_id: str, argv_repr: str) -> None:
    """Always on stderr; unconditional (no CLI flag disables this)."""
    print(_banner("="), file=sys.stderr, flush=True)
    _log(run_id, "ORCHESTRATOR START")
    _log(run_id, f"argv: {argv_repr}")
    _log(run_id, f"root: {ROOT}")
    _log(run_id, f"python: {sys.executable}")
    print(_banner("="), file=sys.stderr, flush=True)


def _emit_orchestrator_footer(run_id: str, message: str) -> None:
    print(_banner("="), file=sys.stderr, flush=True)
    _log(run_id, message)
    print(_banner("="), file=sys.stderr, flush=True)


def _prepare_output_run_dir(run_id: str, argv_repr: str, plan: list[str]) -> Path:
    """Create ``output/<RUN_ID>/`` with manifest + config snapshot."""
    d = ROOT / "output" / run_id
    d.mkdir(parents=True, exist_ok=True)
    cfg = ROOT / "config" / "config.yaml"
    if cfg.is_file():
        shutil.copy2(cfg, d / "config.snapshot.yaml")
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
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


def _write_run_summary(run_dir: Path, run_id: str, status: str, steps: list[dict]) -> None:
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


def step_pytest(extra_pytest_args: list[str], run_dir: Path) -> int:
    cmd = [sys.executable, "-m", "pytest", "tests/"]
    if extra_pytest_args:
        cmd.extend(extra_pytest_args)
    else:
        cmd.append("-q")
    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC) if not prev else f"{SRC}{os.pathsep}{prev}"
    log_path = run_dir / "step_pytest.log"
    with log_path.open("w", encoding="utf-8") as logf:
        return subprocess.call(cmd, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)


def step_benchmark(no_rich: bool, run_id: str, run_dir: Path) -> int:
    _ensure_src_path()
    import json as json_module

    from rich.console import Console
    from rich.table import Table

    from rag_engine.app import RAGEngine
    from rag_engine.config.loader import load_engine_config
    from rag_engine.evaluation.benchmark import benchmark_results_to_jsonable, run_strategy_benchmark

    cfg_path = ROOT / "config" / "config.yaml"
    _log(run_id, f"benchmark: loading config {cfg_path}")
    cfg = load_engine_config(cfg_path)
    _log(run_id, "benchmark: building RAGEngine (may load embedding model)...")
    t0 = time.perf_counter()
    engine = RAGEngine(config=cfg)
    _log(run_id, f"benchmark: ingest started ({engine.config.data.input_path})")
    n = engine.ingest(ROOT / engine.config.data.input_path)
    _log(run_id, f"benchmark: ingest done chunks_indexed={n} elapsed_ms={1000*(time.perf_counter()-t0):.0f}")
    t1 = time.perf_counter()
    results = run_strategy_benchmark(engine.retriever, cfg, queries=cfg.benchmark.queries)
    _log(run_id, f"benchmark: queries finished elapsed_ms={1000*(time.perf_counter()-t1):.0f}")
    payload = {"chunks_indexed": n, "strategy_a_vs_b": benchmark_results_to_jsonable(results)}
    out_json = json_module.dumps(payload, indent=2)
    (run_dir / "step_benchmark.json").write_text(out_json + "\n", encoding="utf-8")
    _log(run_id, "benchmark: JSON OUTPUT (stdout, pipe-safe) -----")
    print(out_json, flush=True)
    _log(run_id, "benchmark: end JSON OUTPUT -----")

    if not no_rich:
        _log(run_id, "benchmark: Rich table OUTPUT (stderr + output folder) -----")
        table = Table(title="Strategy A vs B (top-1 preview)", show_lines=True)
        table.add_column("Query", max_width=36)
        table.add_column("A chunk", max_width=14)
        table.add_column("A score")
        table.add_column("B chunk", max_width=14)
        table.add_column("B score")
        for br in results:
            a0 = br.raw_results[0] if br.raw_results else None
            b0 = br.expanded_results[0] if br.expanded_results else None
            table.add_row(
                br.query[:80] + ("…" if len(br.query) > 80 else ""),
                a0.chunk_id if a0 else "",
                f"{a0.score:.4f}" if a0 else "",
                b0.chunk_id if b0 else "",
                f"{b0.score:.4f}" if b0 else "",
            )
        Console(stderr=True).print(table)
        with (run_dir / "step_benchmark_rich.txt").open("w", encoding="utf-8") as tf:
            Console(file=tf, width=120).print(table)
        _log(run_id, "benchmark: end Rich table OUTPUT -----")
    return 0


def step_smoke(run_id: str, run_dir: Path) -> int:
    _ensure_src_path()
    from rag_engine.app import RAGEngine
    from rag_engine.config.loader import load_engine_config
    from rag_engine.documents.loader import load_documents_json

    cfg_path = ROOT / "config" / "config.yaml"
    cfg = load_engine_config(cfg_path)
    cfg.embedding.provider = "mock_vertex"
    cfg.embedding.mock_vertex = cfg.embedding.mock_vertex.model_copy(update={"dimensions": 16})
    cfg.vector_store.provider = "numpy"

    _log(run_id, "smoke: load default dataset + chunk + mock embed + ingest (rag_engine)")
    engine = RAGEngine(config=cfg)
    path = ROOT / engine.config.data.input_path
    _log(run_id, f"smoke: loading {path}")
    docs = load_documents_json(path, cfg.data.text_field, cfg.data.id_field)
    n = engine.ingest(path)
    _log(run_id, f"smoke: documents={len(docs)} chunks_indexed={n}")
    summary = {
        "run_id": run_id,
        "dataset": str(path.resolve()),
        "document_count": len(docs),
        "chunks_indexed": n,
    }
    (run_dir / "step_smoke.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(run_id, "smoke: OUTPUT (summary)", out=True)
    print(f"  run_id:              {run_id}", flush=True)
    print(f"  dataset:             {path}", flush=True)
    print(f"  documents:           {len(docs)}", flush=True)
    print(f"  chunks indexed:      {n}", flush=True)
    print(f"  artifacts:           {run_dir / 'step_smoke.json'}", flush=True)
    return 0


def step_write_benchmark_md(output: Path, run_id: str, run_dir: Path) -> int:
    _ensure_src_path()
    from rag_engine.app import RAGEngine
    from rag_engine.config.loader import load_engine_config
    from rag_engine.evaluation.benchmark import benchmark_results_to_jsonable, run_strategy_benchmark
    from rag_engine.evaluation.reporter import write_benchmark_json, write_benchmark_markdown

    cfg_path = ROOT / "config" / "config.yaml"
    cfg = load_engine_config(cfg_path)
    _log(run_id, f"write-benchmark-md: writing {output.resolve()}")
    engine = RAGEngine(config=cfg)
    _log(run_id, "write-benchmark-md: ingesting...")
    n = engine.ingest(ROOT / engine.config.data.input_path)
    results = run_strategy_benchmark(engine.retriever, cfg, queries=cfg.benchmark.queries)
    rows = benchmark_results_to_jsonable(results)
    write_benchmark_json(cfg.benchmark.output_json, rows)
    write_benchmark_markdown(results, output, cfg)
    js = json.dumps({"chunks_indexed": n, "strategy_a_vs_b": rows}, indent=2)

    header = """# Retrieval benchmark: Strategy A vs Strategy B

## Objective

Compare raw vector retrieval (Strategy A) against mocked query expansion (Strategy B). Default embedding model is **BAAI/bge-small-en-v1.5** with an asymmetric **query instruction** on Strategy A/B queries (see ``config/config.yaml``). See ``plan.md`` / ``context.md`` and ``DOCUMENTATION.md`` for architecture and methodology.

## Machine-readable output

Generated via ``python3 orchestrator.py --write-benchmark-md``. JSON snapshot (also written to ``storage/benchmark_results.json`` per config):

```json
"""
    footer = """
```

## Notes

Regenerate after corpus or **embedding** changes. The embedded JSON uses neutral ``comparison`` fields (overlap, informational score delta, ``expansion_changed``); it does **not** declare a cross-strategy ``winner`` from cosine magnitudes alone.
"""
    output.write_text(header + js + footer, encoding="utf-8")
    shutil.copy2(output, run_dir / "step_write_benchmark_md.md")
    if cfg.benchmark.output_json.is_file():
        shutil.copy2(cfg.benchmark.output_json, run_dir / "step_write_benchmark_rows.json")
    meta = {
        "run_id": run_id,
        "chunks_indexed": n,
        "markdown_root": str(output.resolve()),
        "markdown_copy": str((run_dir / "step_write_benchmark_md.md").resolve()),
        "json_storage_path": str(cfg.benchmark.output_json.resolve()),
    }
    (run_dir / "step_write_benchmark_md.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log(run_id, f"write-benchmark-md: wrote {len(header + js + footer)} bytes")
    _log(run_id, "write-benchmark-md: OUTPUT", out=True)
    print(f"  path:   {output.resolve()}", flush=True)
    print(f"  bytes:  {output.stat().st_size}", flush=True)
    print(f"  copy:   {run_dir / 'step_write_benchmark_md.md'}", flush=True)
    return 0


def _parse_steps_arg(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def _run_steps(
    names: list[str],
    *,
    run_id: str,
    run_dir: Path,
    benchmark_no_rich: bool,
    md_output: Path,
    pytest_extra: list[str],
) -> int:
    total = len(names)
    _log(run_id, f"planned steps ({total}): {', '.join(names)}")
    step_records: list[dict] = []

    for i, name in enumerate(names, start=1):
        if name not in STEP_DESCRIPTIONS:
            _log(run_id, f"unknown step: {name!r} (use --list-steps)")
            step_records.append({"step": name, "index": i, "exit_code": 2, "error": "unknown_step"})
            _write_run_summary(run_dir, run_id, "failed", step_records)
            _emit_orchestrator_footer(run_id, f"RUN FAILED  unknown_step={name!r}")
            return 2

        print(_banner("="), file=sys.stderr, flush=True)
        _log(run_id, f"STEP {i}/{total} START  name={name}")
        _log(run_id, f"         about: {STEP_DESCRIPTIONS[name]}")
        print(_banner("-"), file=sys.stderr, flush=True)

        t0 = time.perf_counter()
        if name == "pytest":
            _log(run_id, "pytest: command " + " ".join(_pytest_cmd(pytest_extra)))
            _log(run_id, f"pytest: log -> {run_dir / 'step_pytest.log'}")
            code = step_pytest(pytest_extra, run_dir)
            rec: dict = {
                "step": name,
                "index": i,
                "exit_code": code,
                "duration_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "artifacts": ["step_pytest.log"],
            }
        elif name == "benchmark":
            code = step_benchmark(benchmark_no_rich, run_id, run_dir)
            arts = ["step_benchmark.json"]
            if not benchmark_no_rich:
                arts.append("step_benchmark_rich.txt")
            rec = {
                "step": name,
                "index": i,
                "exit_code": code,
                "duration_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "artifacts": arts,
            }
        elif name == "smoke":
            code = step_smoke(run_id, run_dir)
            rec = {
                "step": name,
                "index": i,
                "exit_code": code,
                "duration_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "artifacts": ["step_smoke.json"],
            }
        elif name == "write-benchmark-md":
            code = step_write_benchmark_md(md_output, run_id, run_dir)
            rec = {
                "step": name,
                "index": i,
                "exit_code": code,
                "duration_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "artifacts": [
                    "step_write_benchmark_md.md",
                    "step_write_benchmark_md.json",
                    "step_write_benchmark_rows.json",
                ],
            }
        else:
            code = 2
            rec = {"step": name, "index": i, "exit_code": code, "duration_ms": 0.0, "error": "internal"}
        elapsed_ms = rec["duration_ms"]
        step_records.append(rec)

        print(_banner("-"), file=sys.stderr, flush=True)
        _log(run_id, f"STEP {i}/{total} END    name={name}  exit_code={code}  duration_ms={elapsed_ms:.1f}")
        print(_banner("="), file=sys.stderr, flush=True)

        if code != 0:
            _log(run_id, f"aborting run: step {name!r} failed")
            _write_run_summary(run_dir, run_id, "failed", step_records)
            _emit_orchestrator_footer(run_id, f"RUN FAILED  step={name}  exit_code={code}")
            return code

    _write_run_summary(run_dir, run_id, "success", step_records)
    _log(run_id, f"Artifacts directory: {run_dir.resolve()}", out=True)
    _emit_orchestrator_footer(run_id, f"RUN COMPLETE  all {total} step(s) succeeded")
    return 0


def _pytest_cmd(extra_pytest_args: list[str]) -> list[str]:
    cmd = [sys.executable, "-m", "pytest", "tests/"]
    if extra_pytest_args:
        cmd.extend(extra_pytest_args)
    else:
        cmd.append("-q")
    return cmd


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    pytest_tail: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        pytest_tail = argv[i + 1 :]
        argv = argv[:i]

    parser = argparse.ArgumentParser(
        description="Orchestrator: run pytest, benchmarks, smoke ingest, and markdown export.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="Print available step names and descriptions, then exit.",
    )
    parser.add_argument(
        "--steps",
        metavar="NAMES",
        help="Comma-separated steps to run in order (e.g. pytest,benchmark).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Shorthand for: pytest then benchmark (same as --steps pytest,benchmark).",
    )
    parser.add_argument("--pytest", action="store_true", help="Run only the pytest step.")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run only the benchmark step (ingest + JSON + optional Rich table).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the smoke step (load/chunk/embed stats).",
    )
    parser.add_argument(
        "--write-benchmark-md",
        action="store_true",
        help="Run benchmark and write markdown report to --output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "retrieval_benchmark.md",
        help="Target path for --write-benchmark-md (default: retrieval_benchmark.md).",
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="With --benchmark: skip only the Rich summary table (orchestrator logs always print to stderr).",
    )
    parser.epilog = (
        (parser.epilog or "")
        + "\nPass extra pytest arguments after -- , e.g.\n  python3 orchestrator.py --pytest -- -k chroma -x\n"
    )

    args = parser.parse_args(argv)

    run_id = _new_run_id()
    argv_repr = " ".join(argv) if argv else "(none)"

    if args.list_steps:
        _emit_orchestrator_header(run_id, argv_repr)
        _log(run_id, "mode=list-steps (listing only, no steps executed)")
        print(_banner("-"), file=sys.stderr, flush=True)
        for name, desc in STEP_DESCRIPTIONS.items():
            _log(run_id, f"  {name:22} {desc}")
        _emit_orchestrator_footer(run_id, "RUN COMPLETE (list-steps)")
        return 0

    selected: list[str] = []
    if args.all:
        selected.extend(["pytest", "benchmark"])
    if args.steps:
        selected.extend(_parse_steps_arg(args.steps))
    if args.pytest:
        selected.append("pytest")
    if args.benchmark:
        selected.append("benchmark")
    if args.smoke:
        selected.append("smoke")
    if args.write_benchmark_md:
        selected.append("write-benchmark-md")

    # Deduplicate preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for s in selected:
        if s not in seen:
            seen.add(s)
            ordered.append(s)

    if not ordered:
        _emit_orchestrator_header(run_id, argv_repr)
        _log(run_id, "mode=help (no steps selected; printing argparse help to stdout)")
        print(_banner("-"), file=sys.stderr, flush=True)
        parser.print_help()
        _log(run_id, "ERROR: pass --all, --steps ..., or a step flag (see stdout help)")
        _emit_orchestrator_footer(run_id, "RUN ABORTED (no steps)")
        return 1

    run_dir = _prepare_output_run_dir(run_id, argv_repr, ordered)
    _emit_orchestrator_header(run_id, argv_repr)
    _log(run_id, f"artifacts -> {run_dir.resolve()}")

    return _run_steps(
        ordered,
        run_id=run_id,
        run_dir=run_dir,
        benchmark_no_rich=args.no_rich,
        md_output=args.output,
        pytest_extra=pytest_tail,
    )


if __name__ == "__main__":
    raise SystemExit(main())
