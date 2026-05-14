"""Retrieval quality notes (plan §12 honest observations)."""

from __future__ import annotations

from rag_engine.retrieval.result_models import SearchResult


def scores_descending(results: list[SearchResult]) -> bool:
    scores = [r.score for r in results]
    return scores == sorted(scores, reverse=True)


def top_score(results: list[SearchResult]) -> float | None:
    if not results:
        return None
    return results[0].score


def summarize_strategy_shift(
    raw: list[SearchResult],
    expanded: list[SearchResult],
    no_match_threshold: float,
) -> str:
    notes: list[str] = []
    ts_raw = top_score(raw)
    ts_exp = top_score(expanded)
    if ts_raw is not None and ts_raw < no_match_threshold:
        notes.append(
            "Top Strategy A score is below the configured no-match design threshold; "
            "production might return no context instead of forcing a chunk."
        )
    if ts_exp is not None and ts_exp < no_match_threshold:
        notes.append("Top Strategy B score is similarly low after expansion.")

    rid = raw[0].chunk_id if raw else None
    eid = expanded[0].chunk_id if expanded else None
    if rid and eid and rid != eid:
        notes.append("Top-1 chunk differs: expansion shifted lexical overlap.")
    elif rid and eid:
        notes.append("Top-1 chunk matches; compare deeper ranks and score deltas.")

    if ts_raw is not None and ts_exp is not None and ts_exp + 1e-6 < ts_raw:
        notes.append(
            "Strategy B rank-1 score is lower than Strategy A - expansion is not universally better "
            "and can reduce precision when extra terms misalign with the corpus."
        )
    elif ts_raw is not None and ts_exp is not None and ts_exp > ts_raw + 1e-6:
        notes.append("Strategy B increased the top similarity - likely improved recall for this query.")

    if not notes:
        return "Review expanded query text alongside scores; ambiguous queries benefit most from expansion."
    return " ".join(notes)


def overlap_count(a: list[SearchResult], b: list[SearchResult], top_n: int = 3) -> int:
    sa = {x.chunk_id for x in a[:top_n]}
    sb = {x.chunk_id for x in b[:top_n]}
    return len(sa & sb)


def pick_winner(a: list[SearchResult], b: list[SearchResult]) -> str:
    ta = top_score(a)
    tb = top_score(b)
    if ta is None and tb is None:
        return "tie"
    if ta is None:
        return "strategy_b"
    if tb is None:
        return "strategy_a"
    if tb > ta + 1e-6:
        return "strategy_b"
    if ta > tb + 1e-6:
        return "strategy_a"
    return "tie"
