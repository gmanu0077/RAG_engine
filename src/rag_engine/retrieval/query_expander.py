"""Mock Vertex GenerativeModel + query expansion (plan §4.3)."""

from __future__ import annotations

import re

from rag_engine.config.schema import QueryExpansionConfig

MOCK_EXPANSIONS: dict[str, str] = {
    "How does the system handle peak load?": (
        "system scalability during peak traffic, high concurrency, autoscaling, "
        "load balancing, throughput, latency"
    ),
    "What happens when downstream services fail?": (
        "downstream dependency failure handling, retries, circuit breakers, "
        "fallback behavior, timeout handling"
    ),
    "How is data consistency maintained during concurrent updates?": (
        "data consistency, concurrent writes, transactions, locking, "
        "optimistic concurrency control, idempotency"
    ),
    "peak load without increasing latency": (
        "horizontal autoscaling, HPA, queue depth, connection draining, "
        "latency SLOs, throughput under peak traffic"
    ),
    "downstream services are unavailable": (
        "downstream outage, circuit breaker, retries with backoff, "
        "degraded mode, timeout budgets"
    ),
    "data kept consistent during concurrent updates": (
        "transactions, optimistic locking, CRDTs, vector clocks, "
        "read-your-writes, replica lag"
    ),
    "recover from infrastructure failure": (
        "regional failover, control plane recovery, runbook automation, "
        "disaster recovery, replica promotion"
    ),
    "repeated processing of the same request": (
        "idempotency keys, deduplication, exactly-once semantics, "
        "consumer offsets, poison message quarantine"
    ),
    "How are user requests protected from unauthorized access?": (
        "authentication, authorization, tenant isolation, mutual TLS, "
        "row-level security, access control boundaries"
    ),
}


class MockResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class MockGenerativeModel:
    """Maps prompts to canned expansions; empty response falls back to original query."""

    def __init__(self, expansions: dict[str, str] | None = None) -> None:
        self._expansions = expansions or dict(MOCK_EXPANSIONS)

    def generate_content(self, prompt: str) -> MockResponse:
        for needle, expanded in self._expansions.items():
            if needle in prompt:
                return MockResponse(expanded)
        m = re.search(r"User query:\s*(.+?)(?:\n|$)", prompt, flags=re.DOTALL)
        if m:
            q = m.group(1).strip()
            if q in self._expansions:
                return MockResponse(self._expansions[q])
        return MockResponse("")


class QueryExpander:
    def __init__(self, generative_model: object, cfg: QueryExpansionConfig) -> None:
        self._model = generative_model
        self._cfg = cfg

    def expand(self, query: str) -> str:
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        if not self._cfg.enabled or self._cfg.provider == "none":
            return query.strip()
        prompt = self._cfg.prompt_template.format(query=query.strip())
        response = self._model.generate_content(prompt)
        expanded = getattr(response, "text", str(response)).strip()
        if not expanded:
            return query.strip()
        max_c = self._cfg.expansion_max_chars
        if len(expanded) <= max_c:
            return expanded
        cut = expanded[:max_c].rsplit(" ", 1)[0]
        return cut if cut else expanded[:max_c]
