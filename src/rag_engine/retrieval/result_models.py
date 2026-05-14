"""Retrieval and benchmark result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SearchResult:
    rank: int
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ExpandedSearchResult:
    original_query: str
    expanded_query: str
    results: list[SearchResult]


@dataclass(frozen=True)
class BenchmarkResult:
    query: str
    expanded_query: str
    raw_results: list[SearchResult]
    expanded_results: list[SearchResult]
    notes: str
