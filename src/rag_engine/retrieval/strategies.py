"""Strategy A vs B identifiers (plan §4)."""

from __future__ import annotations

from enum import Enum


class RetrievalStrategy(str, Enum):
    strategy_a_raw_vector = "strategy_a_raw_vector"
    strategy_b_query_expansion = "strategy_b_query_expansion"
