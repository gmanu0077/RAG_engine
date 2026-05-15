"""Retrieval comparison notes (plan §12 — honest observations, no spurious 'winner')."""

from __future__ import annotations

from rag_engine.retrieval.result_models import SearchResult


def scores_descending(results: list[SearchResult]) -> bool:
    scores = [r.score for r in results]
    return scores == sorted(scores, reverse=True)


def top_score(results: list[SearchResult]) -> float | None:
    if not results:
        return None
    return results[0].score


def top1_score_delta_b_minus_a(a: list[SearchResult], b: list[SearchResult]) -> float | None:
    """Signed difference of rank-1 scores (B minus A). Not a recall metric — different queries are embedded."""
    ta = top_score(a)
    tb = top_score(b)
    if ta is None or tb is None:
        return None
    return float(tb - ta)


def summarize_strategy_shift(
    raw: list[SearchResult],
    expanded: list[SearchResult],
    no_match_threshold: float,
) -> str:
    notes: list[str] = [
        "Strategy A and Strategy B embed different query strings; rank-1 cosine values are not "
        "directly comparable as a quality score (higher B does not imply better retrieval)."
    ]
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
        notes.append("Top-1 chunk id differs between strategies; compare tables for rank/shift detail.")
    elif rid and eid:
        notes.append("Top-1 chunk id matches; deeper ranks may still differ.")

    if not notes:
        return "Compare chunk ids, overlap, and qualitative previews; no labelled relevance on this corpus."
    return " ".join(notes)


def overlap_count(a: list[SearchResult], b: list[SearchResult], top_n: int = 3) -> int:
    sa = {x.chunk_id for x in a[:top_n]}
    sb = {x.chunk_id for x in b[:top_n]}
    return len(sa & sb)
