"""Deterministic canned query expansions for mocks (brief §2 / §4.3).

Lives outside ``retrieval/`` so ``gcp_mocks`` can import it without loading
``rag_engine.retrieval`` (avoids circular imports with ``vertex_stubs``).
"""

from __future__ import annotations

import re

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


def expansion_text_for_prompt(prompt: str, expansions: dict[str, str] | None = None) -> str:
    """Return expanded text, or empty string if no canned rule matches."""
    d = expansions if expansions is not None else MOCK_EXPANSIONS
    for needle, expanded in d.items():
        if needle in prompt:
            return expanded
    m = re.search(r"User query:\s*(.+?)(?:\n|$)", prompt, flags=re.DOTALL)
    if m:
        q = m.group(1).strip()
        if q in d:
            return d[q]
    return ""
