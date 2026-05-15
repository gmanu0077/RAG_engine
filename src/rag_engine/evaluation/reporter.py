"""JSON + Markdown benchmark reports (plan §12)."""

from __future__ import annotations

import json
from pathlib import Path

from rag_engine.config.schema import EngineConfig
from rag_engine.evaluation.metrics import overlap_count, top1_score_delta_b_minus_a
from rag_engine.retrieval.result_models import BenchmarkResult


def write_benchmark_json(path: Path, jsonable_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable_rows, indent=2, ensure_ascii=False), encoding="utf-8")


def write_benchmark_markdown(results: list[BenchmarkResult], path: Path, cfg: EngineConfig) -> None:
    lines: list[str] = [
        "# Retrieval Benchmark: Strategy A vs Strategy B",
        "",
        "This report compares **two retrieval paths** on the same corpus. It does **not** declare a "
        "\"winner\" from cosine scores: Strategy A and Strategy B embed **different query strings**, "
        "so absolute similarities are not directly comparable. Use **chunk-id overlap**, **rank "
        "shifts**, and **qualitative previews** — not raw score magnitude — to interpret results.",
        "",
        f"Embedding: `{cfg.embedding.model_name}` · Vector store: `{cfg.vector_store.provider}`",
        "",
    ]
    for i, br in enumerate(results, start=1):
        ov = overlap_count(br.raw_results, br.expanded_results, top_n=3)
        delta = top1_score_delta_b_minus_a(br.raw_results, br.expanded_results)
        expansion_changed = br.expanded_query.strip() != br.query.strip()
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
        if not expansion_changed:
            lines.append("*Query expansion returned the original text (no canned rule matched).*")
            lines.append("")
        lines.extend(["| Rank | Chunk ID | Score | Preview |", "|---:|---|---:|---|"])
        for h in br.expanded_results:
            prev = h.text.replace("\n", " ")[:120]
            lines.append(f"| {h.rank} | {h.chunk_id} | {h.score:.4f} | {prev}… |")
        delta_s = "n/a" if delta is None else f"{delta:+.4f}"
        lines.extend(
            [
                "",
                "### Comparison",
                "",
                f"- Overlap (top-3 chunk ids): **{ov}**",
                f"- Rank-1 score delta (B − A, informational only): **{delta_s}**",
                f"- Notes: {br.notes}",
                "",
            ],
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
