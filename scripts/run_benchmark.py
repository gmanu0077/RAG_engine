#!/usr/bin/env python3
"""Run Strategy A vs Strategy B benchmark and print JSON (see retrieval_benchmark.md)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rich.console import Console
from rich.table import Table

from rag_engine.app import RAGEngine
from rag_engine.config.loader import load_engine_config
from rag_engine.evaluation.benchmark import benchmark_results_to_jsonable, run_strategy_benchmark


def main() -> None:
    cfg = load_engine_config(ROOT / "config" / "config.yaml")
    engine = RAGEngine(config=cfg)
    n = engine.ingest(ROOT / engine.config.data.input_path)
    results = run_strategy_benchmark(engine.retriever, cfg, queries=cfg.benchmark.queries)
    payload = {"chunks_indexed": n, "strategy_a_vs_b": benchmark_results_to_jsonable(results)}
    print(json.dumps(payload, indent=2))

    console = Console(stderr=True)
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
    console.print(table)


if __name__ == "__main__":
    main()
