"""JSON + Markdown benchmark reports (plan §12)."""

from __future__ import annotations

import json
from pathlib import Path

from rag_engine.config.schema import EngineConfig
from rag_engine.evaluation.metrics import overlap_count, pick_winner
from rag_engine.retrieval.result_models import BenchmarkResult


def write_benchmark_json(path: Path, jsonable_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable_rows, indent=2, ensure_ascii=False), encoding="utf-8")


def write_benchmark_markdown(results: list[BenchmarkResult], path: Path, cfg: EngineConfig) -> None:
    lines: list[str] = [
        "# Retrieval Benchmark: Strategy A vs Strategy B",
        "",
        f"Embedding: `{cfg.embedding.model_name}` · Vector store: `{cfg.vector_store.provider}`",
        "",
    ]
    for i, br in enumerate(results, start=1):
        ov = overlap_count(br.raw_results, br.expanded_results, top_n=3)
        winner = pick_winner(br.raw_results, br.expanded_results)
        lines.extend(
            [
                f"## Query {i}: {br.query}",
                "",
                "### Strategy A: Raw Vector Search",
                "",
                "| Rank | Chunk ID | Score | Preview |",
                "|---:|---|---:|---|",
            ],
        )
        for h in br.raw_results:
            prev = h.text.replace("\n", " ")[:120]
            lines.append(f"| {h.rank} | {h.chunk_id} | {h.score:.4f} | {prev}… |")
        lines.extend(["", "### Strategy B: AI-Enhanced Retrieval", "", f"Expanded query: `{br.expanded_query}`", ""])
        lines.extend(["| Rank | Chunk ID | Score | Preview |", "|---:|---|---:|---|"])
        for h in br.expanded_results:
            prev = h.text.replace("\n", " ")[:120]
            lines.append(f"| {h.rank} | {h.chunk_id} | {h.score:.4f} | {prev}… |")
        lines.extend(
            [
                "",
                "### Comparison",
                "",
                f"- Overlap (top-3 chunk ids): **{ov}**",
                f"- Winner (heuristic on top-1 score): **{winner}**",
                f"- Notes: {br.notes}",
                "",
            ],
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
