"""Strategy A vs B benchmark runner (plan §4.4)."""

from __future__ import annotations

from rag_engine.config.schema import EngineConfig
from rag_engine.evaluation.metrics import overlap_count, pick_winner, summarize_strategy_shift
from rag_engine.retrieval.result_models import BenchmarkResult
from rag_engine.retrieval.retriever import Retriever


def default_benchmark_queries() -> list[str]:
    return [
        "How does the system handle peak load without increasing latency?",
        "What happens when downstream services are unavailable?",
        "How is data kept consistent during concurrent updates?",
        "How does the platform recover from infrastructure failure?",
        "What mechanisms prevent repeated processing of the same request?",
    ]


def run_strategy_benchmark(
    retriever: Retriever,
    engine_cfg: EngineConfig,
    queries: list[str] | None = None,
    top_k: int | None = None,
) -> list[BenchmarkResult]:
    k = top_k if top_k is not None else engine_cfg.retrieval.top_k
    qs = queries if queries is not None else default_benchmark_queries()
    out: list[BenchmarkResult] = []
    for q in qs:
        raw = retriever.retrieve_raw(q, top_k=k)
        expanded_bundle = retriever.retrieve_with_expansion(q, top_k=k)
        note = summarize_strategy_shift(
            raw,
            expanded_bundle.results,
            engine_cfg.retrieval.no_match_cosine_threshold,
        )
        out.append(
            BenchmarkResult(
                query=q,
                expanded_query=expanded_bundle.expanded_query,
                raw_results=list(raw),
                expanded_results=list(expanded_bundle.results),
                notes=note,
            ),
        )
    return out


def benchmark_results_to_jsonable(results: list[BenchmarkResult]) -> list[dict]:
    def pack_hit(r):
        return {
            "rank": r.rank,
            "chunk_id": r.chunk_id,
            "score": round(r.score, 6),
            "text_preview": r.text[:220],
            "metadata": r.metadata,
        }

    rows = []
    for br in results:
        ov = overlap_count(br.raw_results, br.expanded_results, top_n=3)
        winner = pick_winner(br.raw_results, br.expanded_results)
        rows.append(
            {
                "query": br.query,
                "strategy_a": {
                    "search_query": br.query,
                    "top_results": [pack_hit(x) for x in br.raw_results],
                },
                "strategy_b": {
                    "expanded_query": br.expanded_query,
                    "top_results": [pack_hit(x) for x in br.expanded_results],
                },
                "comparison": {
                    "overlap_count": ov,
                    "winner": winner,
                    "reason": br.notes,
                },
            },
        )
    return rows
