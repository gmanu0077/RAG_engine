# Retrieval benchmark: Strategy A vs Strategy B

## Objective

Compare raw vector retrieval (Strategy A) against mocked query expansion (Strategy B). See ``plan.md`` and ``config/config.yaml`` for architecture and tunables.

## Machine-readable output

Generated via ``python3 orchestrator.py --write-benchmark-md``. JSON snapshot (also written to ``storage/benchmark_results.json`` per config):

```json
{
  "chunks_indexed": 11,
  "strategy_a_vs_b": [
    {
      "query": "How does the system handle peak load?",
      "strategy_a": {
        "search_query": "How does the system handle peak load?",
        "top_results": [
          {
            "rank": 1,
            "chunk_id": "doc_000_chunk_000",
            "score": 0.451218,
            "text_preview": "During peak load the API tier scales horizontally using Kubernetes HPA targets based on CPU and custom request-queue depth signals. New pods join the service mesh within seconds; connection draining ensures in-flight wor",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_000"
            }
          },
          {
            "rank": 2,
            "chunk_id": "doc_002_chunk_000",
            "score": 0.342084,
            "text_preview": "When the primary Redis tier becomes unhealthy the application degrades gracefully by bypassing cache reads and writes, falling back to authoritative database queries while emitting elevated latency alerts to SRE dashboar",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_002"
            }
          },
          {
            "rank": 3,
            "chunk_id": "doc_003_chunk_000",
            "score": 0.319287,
            "text_preview": "Read replicas serve analytical queries and offload the primary writer. Replica lag is monitored continuously; queries that require strongly consistent reads are pinned to the primary connection pool. Transactions on the ",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_003"
            }
          }
        ]
      },
      "strategy_b": {
        "expanded_query": "system scalability during peak traffic, high concurrency, autoscaling, load balancing, throughput, latency",
        "top_results": [
          {
            "rank": 1,
            "chunk_id": "doc_000_chunk_000",
            "score": 0.53761,
            "text_preview": "During peak load the API tier scales horizontally using Kubernetes HPA targets based on CPU and custom request-queue depth signals. New pods join the service mesh within seconds; connection draining ensures in-flight wor",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_000"
            }
          },
          {
            "rank": 2,
            "chunk_id": "doc_001_chunk_000",
            "score": 0.383281,
            "text_preview": "The edge cache uses a two-tier design: an in-memory LRU for hot keys and a regional Redis cluster for shared objects. TTLs are staggered to prevent thundering herds when popular keys expire at the same instant.",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_001"
            }
          },
          {
            "rank": 3,
            "chunk_id": "doc_003_chunk_000",
            "score": 0.366839,
            "text_preview": "Read replicas serve analytical queries and offload the primary writer. Replica lag is monitored continuously; queries that require strongly consistent reads are pinned to the primary connection pool. Transactions on the ",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_003"
            }
          }
        ]
      },
      "comparison": {
        "overlap_count": 2,
        "winner": "strategy_b",
        "reason": "Top-1 chunk matches; compare deeper ranks and score deltas. Strategy B increased the top similarity - likely improved recall for this query."
      }
    },
    {
      "query": "What happens when downstream services fail?",
      "strategy_a": {
        "search_query": "What happens when downstream services fail?",
        "top_results": [
          {
            "rank": 1,
            "chunk_id": "doc_008_chunk_000",
            "score": 0.58265,
            "text_preview": "When downstream services are unavailable the edge gateway sheds optional traffic, applies circuit breakers per dependency, and uses exponential backoff retries only for idempotent verbs. Timeouts are budgeted end-to-end ",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_008"
            }
          },
          {
            "rank": 2,
            "chunk_id": "doc_005_chunk_000",
            "score": 0.424505,
            "text_preview": "Event ingestion pipelines batch acknowledgements and apply backpressure when downstream sinks lag. Poison messages are quarantined after configurable retry budgets to protect overall throughput. Consumer offsets are chec",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_005"
            }
          },
          {
            "rank": 3,
            "chunk_id": "doc_009_chunk_000",
            "score": 0.334732,
            "text_preview": "The platform recovers from infrastructure failure by promoting warm replicas in a secondary region, replaying durable logs, and automating runbooks that validate data checksums before shifting user traffic back to the he",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_009"
            }
          }
        ]
      },
      "strategy_b": {
        "expanded_query": "downstream dependency failure handling, retries, circuit breakers, fallback behavior, timeout handling",
        "top_results": [
          {
            "rank": 1,
            "chunk_id": "doc_008_chunk_000",
            "score": 0.618051,
            "text_preview": "When downstream services are unavailable the edge gateway sheds optional traffic, applies circuit breakers per dependency, and uses exponential backoff retries only for idempotent verbs. Timeouts are budgeted end-to-end ",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_008"
            }
          },
          {
            "rank": 2,
            "chunk_id": "doc_005_chunk_000",
            "score": 0.456833,
            "text_preview": "Event ingestion pipelines batch acknowledgements and apply backpressure when downstream sinks lag. Poison messages are quarantined after configurable retry budgets to protect overall throughput. Consumer offsets are chec",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_005"
            }
          },
          {
            "rank": 3,
            "chunk_id": "doc_009_chunk_000",
            "score": 0.363646,
            "text_preview": "The platform recovers from infrastructure failure by promoting warm replicas in a secondary region, replaying durable logs, and automating runbooks that validate data checksums before shifting user traffic back to the he",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_009"
            }
          }
        ]
      },
      "comparison": {
        "overlap_count": 3,
        "winner": "strategy_b",
        "reason": "Top-1 chunk matches; compare deeper ranks and score deltas. Strategy B increased the top similarity - likely improved recall for this query."
      }
    },
    {
      "query": "How are user requests protected from unauthorized access?",
      "strategy_a": {
        "search_query": "How are user requests protected from unauthorized access?",
        "top_results": [
          {
            "rank": 1,
            "chunk_id": "doc_006_chunk_000",
            "score": 0.295643,
            "text_preview": "Tenant isolation enforces row-level security at the database layer and mutual TLS between microservices. Secrets never appear in logs\u2014only opaque identifiers are printed for correlation.",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_006"
            }
          },
          {
            "rank": 2,
            "chunk_id": "doc_010_chunk_000",
            "score": 0.22097,
            "text_preview": "Repeated processing of the same request is prevented by idempotency keys stored in a short-lived deduplication table; clients replaying a submission receive the original response without double-charging inventory or re-e",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_010"
            }
          },
          {
            "rank": 3,
            "chunk_id": "doc_003_chunk_000",
            "score": 0.127738,
            "text_preview": "Read replicas serve analytical queries and offload the primary writer. Replica lag is monitored continuously; queries that require strongly consistent reads are pinned to the primary connection pool. Transactions on the ",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_003"
            }
          }
        ]
      },
      "strategy_b": {
        "expanded_query": "authentication, authorization, tenant isolation, mutual TLS, row-level security, access control boundaries",
        "top_results": [
          {
            "rank": 1,
            "chunk_id": "doc_006_chunk_000",
            "score": 0.686261,
            "text_preview": "Tenant isolation enforces row-level security at the database layer and mutual TLS between microservices. Secrets never appear in logs\u2014only opaque identifiers are printed for correlation.",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_006"
            }
          },
          {
            "rank": 2,
            "chunk_id": "doc_001_chunk_000",
            "score": 0.245331,
            "text_preview": "The edge cache uses a two-tier design: an in-memory LRU for hot keys and a regional Redis cluster for shared objects. TTLs are staggered to prevent thundering herds when popular keys expire at the same instant.",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_001"
            }
          },
          {
            "rank": 3,
            "chunk_id": "doc_004_chunk_000",
            "score": 0.226724,
            "text_preview": "Cross-region writes flow through a conflict-free replicated data type for inventory counters while user profile updates use last-write-wins with vector clocks to survive partial network partitions between regions.",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_004"
            }
          }
        ]
      },
      "comparison": {
        "overlap_count": 1,
        "winner": "strategy_b",
        "reason": "Top-1 chunk matches; compare deeper ranks and score deltas. Strategy B increased the top similarity - likely improved recall for this query."
      }
    }
  ]
}
```

## Conclusion

Regenerate this file after corpus or retrieval changes. Strategy B can improve recall on ambiguous queries but may reduce top-1 precision when expansions misalign with the corpus; production needs score thresholds, filters, and monitoring.
