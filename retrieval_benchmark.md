# Retrieval benchmark: Strategy A vs Strategy B

## Objective

Compare raw vector retrieval (Strategy A) against mocked query expansion (Strategy B). Default embedding model is **BAAI/bge-small-en-v1.5** with an asymmetric **query instruction** on Strategy A/B queries (see ``config/config.yaml``). See ``plan.md`` / ``context.md`` and ``DOCUMENTATION.md`` for architecture and methodology.

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
            "score": 0.686012,
            "text_preview": "During peak load the API tier scales horizontally using Kubernetes HPA targets based on CPU and custom request-queue depth signals. New pods join the service mesh within seconds; connection draining ensures in-flight wor",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_000"
            }
          },
          {
            "rank": 2,
            "chunk_id": "doc_003_chunk_000",
            "score": 0.596352,
            "text_preview": "Read replicas serve analytical queries and offload the primary writer. Replica lag is monitored continuously; queries that require strongly consistent reads are pinned to the primary connection pool. Transactions on the ",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_003"
            }
          },
          {
            "rank": 3,
            "chunk_id": "doc_001_chunk_000",
            "score": 0.595259,
            "text_preview": "The edge cache uses a two-tier design: an in-memory LRU for hot keys and a regional Redis cluster for shared objects. TTLs are staggered to prevent thundering herds when popular keys expire at the same instant.",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_001"
            }
          }
        ]
      },
      "strategy_b": {
        "expanded_query": "system scalability during peak traffic, high concurrency, autoscaling, load balancing, throughput, latency",
        "expansion_changed": true,
        "top_results": [
          {
            "rank": 1,
            "chunk_id": "doc_000_chunk_000",
            "score": 0.741233,
            "text_preview": "During peak load the API tier scales horizontally using Kubernetes HPA targets based on CPU and custom request-queue depth signals. New pods join the service mesh within seconds; connection draining ensures in-flight wor",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_000"
            }
          },
          {
            "rank": 2,
            "chunk_id": "doc_001_chunk_000",
            "score": 0.702441,
            "text_preview": "The edge cache uses a two-tier design: an in-memory LRU for hot keys and a regional Redis cluster for shared objects. TTLs are staggered to prevent thundering herds when popular keys expire at the same instant.",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_001"
            }
          },
          {
            "rank": 3,
            "chunk_id": "doc_004_chunk_000",
            "score": 0.696497,
            "text_preview": "Cross-region writes flow through a conflict-free replicated data type for inventory counters while user profile updates use last-write-wins with vector clocks to survive partial network partitions between regions.",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_004"
            }
          }
        ]
      },
      "comparison": {
        "overlap_count": 2,
        "top1_score_delta_b_minus_a": 0.055222,
        "notes": "Strategy A and Strategy B embed different query strings; rank-1 cosine values are not directly comparable as a quality score (higher B does not imply better retrieval). Top-1 chunk id matches; deeper ranks may still differ."
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
            "score": 0.674185,
            "text_preview": "When downstream services are unavailable the edge gateway sheds optional traffic, applies circuit breakers per dependency, and uses exponential backoff retries only for idempotent verbs. Timeouts are budgeted end-to-end ",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_008"
            }
          },
          {
            "rank": 2,
            "chunk_id": "doc_005_chunk_000",
            "score": 0.636236,
            "text_preview": "Event ingestion pipelines batch acknowledgements and apply backpressure when downstream sinks lag. Poison messages are quarantined after configurable retry budgets to protect overall throughput. Consumer offsets are chec",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_005"
            }
          },
          {
            "rank": 3,
            "chunk_id": "doc_007_chunk_000",
            "score": 0.585857,
            "text_preview": "Distributed traces stitch HTTP spans with queue consumers and database calls. SLO burn alerts trigger paging when error budgets drain faster than the weekly rolling window allows.",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_007"
            }
          }
        ]
      },
      "strategy_b": {
        "expanded_query": "downstream dependency failure handling, retries, circuit breakers, fallback behavior, timeout handling",
        "expansion_changed": true,
        "top_results": [
          {
            "rank": 1,
            "chunk_id": "doc_008_chunk_000",
            "score": 0.751168,
            "text_preview": "When downstream services are unavailable the edge gateway sheds optional traffic, applies circuit breakers per dependency, and uses exponential backoff retries only for idempotent verbs. Timeouts are budgeted end-to-end ",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_008"
            }
          },
          {
            "rank": 2,
            "chunk_id": "doc_005_chunk_000",
            "score": 0.698801,
            "text_preview": "Event ingestion pipelines batch acknowledgements and apply backpressure when downstream sinks lag. Poison messages are quarantined after configurable retry budgets to protect overall throughput. Consumer offsets are chec",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_005"
            }
          },
          {
            "rank": 3,
            "chunk_id": "doc_007_chunk_000",
            "score": 0.624171,
            "text_preview": "Distributed traces stitch HTTP spans with queue consumers and database calls. SLO burn alerts trigger paging when error budgets drain faster than the weekly rolling window allows.",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_007"
            }
          }
        ]
      },
      "comparison": {
        "overlap_count": 3,
        "top1_score_delta_b_minus_a": 0.076983,
        "notes": "Strategy A and Strategy B embed different query strings; rank-1 cosine values are not directly comparable as a quality score (higher B does not imply better retrieval). Top-1 chunk id matches; deeper ranks may still differ."
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
            "score": 0.688081,
            "text_preview": "Tenant isolation enforces row-level security at the database layer and mutual TLS between microservices. Secrets never appear in logs\u2014only opaque identifiers are printed for correlation.",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_006"
            }
          },
          {
            "rank": 2,
            "chunk_id": "doc_003_chunk_000",
            "score": 0.602094,
            "text_preview": "Read replicas serve analytical queries and offload the primary writer. Replica lag is monitored continuously; queries that require strongly consistent reads are pinned to the primary connection pool. Transactions on the ",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_003"
            }
          },
          {
            "rank": 3,
            "chunk_id": "doc_008_chunk_000",
            "score": 0.575678,
            "text_preview": "When downstream services are unavailable the edge gateway sheds optional traffic, applies circuit breakers per dependency, and uses exponential backoff retries only for idempotent verbs. Timeouts are budgeted end-to-end ",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_008"
            }
          }
        ]
      },
      "strategy_b": {
        "expanded_query": "authentication, authorization, tenant isolation, mutual TLS, row-level security, access control boundaries",
        "expansion_changed": true,
        "top_results": [
          {
            "rank": 1,
            "chunk_id": "doc_006_chunk_000",
            "score": 0.820476,
            "text_preview": "Tenant isolation enforces row-level security at the database layer and mutual TLS between microservices. Secrets never appear in logs\u2014only opaque identifiers are printed for correlation.",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_006"
            }
          },
          {
            "rank": 2,
            "chunk_id": "doc_008_chunk_000",
            "score": 0.621699,
            "text_preview": "When downstream services are unavailable the edge gateway sheds optional traffic, applies circuit breakers per dependency, and uses exponential backoff retries only for idempotent verbs. Timeouts are budgeted end-to-end ",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_008"
            }
          },
          {
            "rank": 3,
            "chunk_id": "doc_001_chunk_000",
            "score": 0.617088,
            "text_preview": "The edge cache uses a two-tier design: an in-memory LRU for hot keys and a regional Redis cluster for shared objects. TTLs are staggered to prevent thundering herds when popular keys expire at the same instant.",
            "metadata": {
              "source": "dataset",
              "document_id": "doc_001"
            }
          }
        ]
      },
      "comparison": {
        "overlap_count": 2,
        "top1_score_delta_b_minus_a": 0.132395,
        "notes": "Strategy A and Strategy B embed different query strings; rank-1 cosine values are not directly comparable as a quality score (higher B does not imply better retrieval). Top-1 chunk id matches; deeper ranks may still differ."
      }
    }
  ]
}
```

## Notes

Regenerate after corpus or **embedding** changes. The embedded JSON uses neutral ``comparison`` fields (overlap, informational score delta, ``expansion_changed``); it does **not** declare a cross-strategy ``winner`` from cosine magnitudes alone.
