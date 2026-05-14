# Retrieval benchmark: Strategy A vs Strategy B

## Objective

Compare raw cosine retrieval (Strategy A) against mocked query expansion plus retrieval (Strategy B). Each row includes similarity scores, chunk IDs, expanded query text, and honest observations (Strategy B is not assumed to always win).

## Dataset

`data/technical_paragraphs.json` — eleven technical paragraphs (autoscaling, cache tiers, downstream outages, consistency, idempotency, recovery, observability, security).

## Embedding model

Local `sentence-transformers` model `all-MiniLM-L6-v2`, applied to both corpus and queries (same model end-to-end).

## Vector store

`NumpyVectorStore`: L2-normalized vectors, cosine similarity as dot product, deterministic tie-break `(-score, chunk_id, index)`, guards for empty index, invalid `top_k`, dimension mismatch, NaN/Inf, and zero vectors.

## Strategy A

Embed the user query as provided.

## Strategy B

`MockGenerativeModel` rewrites using the assessment-style expansion prompt; empty model output falls back to the original query; expansion length is capped via `Settings.expansion_max_chars`.

## Machine-readable output

Regenerate with:

```bash
python3 main.py --write-benchmark-md
# or: python3 scripts/run_benchmark.py
```

```json
{
  "chunks_indexed": 11,
  "strategy_a_vs_b": [
    {
      "query": "How does the system handle peak load without increasing latency?",
      "expanded_query": "horizontal autoscaling, HPA, queue depth, connection draining, latency SLOs, throughput under peak traffic",
      "strategy_a": [
        {
          "rank": 1,
          "chunk_id": "chunk_001",
          "score": 0.543202,
          "text_preview": "Peak load and autoscaling\n\nDuring peak load the API tier scales horizontally using Kubernetes HPA targets based on CPU and custom request-queue depth signals. New pods join the service mesh within seconds; connection dra",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "9ab8bb38c16055ba",
            "section": "Peak load and autoscaling"
          }
        },
        {
          "rank": 2,
          "chunk_id": "chunk_004",
          "score": 0.447115,
          "text_preview": "Database read scaling\n\nRead replicas serve analytical queries and offload the primary writer. Replica lag is monitored continuously; queries that require strongly consistent reads are pinned to the primary connection poo",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "f8816f1c9a071b04",
            "section": "Database read scaling"
          }
        },
        {
          "rank": 3,
          "chunk_id": "chunk_003",
          "score": 0.370816,
          "text_preview": "Cache failure modes\n\nWhen the primary Redis tier becomes unhealthy the application degrades gracefully by bypassing cache reads and writes, falling back to authoritative database queries while emitting elevated latency a",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "8d47c1d734d11b4c",
            "section": "Cache failure modes"
          }
        }
      ],
      "strategy_b": [
        {
          "rank": 1,
          "chunk_id": "chunk_001",
          "score": 0.714625,
          "text_preview": "Peak load and autoscaling\n\nDuring peak load the API tier scales horizontally using Kubernetes HPA targets based on CPU and custom request-queue depth signals. New pods join the service mesh within seconds; connection dra",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "9ab8bb38c16055ba",
            "section": "Peak load and autoscaling"
          }
        },
        {
          "rank": 2,
          "chunk_id": "chunk_008",
          "score": 0.415413,
          "text_preview": "Observability\n\nDistributed traces stitch HTTP spans with queue consumers and database calls. SLO burn alerts trigger paging when error budgets drain faster than the weekly rolling window allows.",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "55a64b401b92b912",
            "section": "Observability"
          }
        },
        {
          "rank": 3,
          "chunk_id": "chunk_004",
          "score": 0.392135,
          "text_preview": "Database read scaling\n\nRead replicas serve analytical queries and offload the primary writer. Replica lag is monitored continuously; queries that require strongly consistent reads are pinned to the primary connection poo",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "f8816f1c9a071b04",
            "section": "Database read scaling"
          }
        }
      ],
      "observation": "Top-1 chunk matches; compare deeper ranks and score deltas. Strategy B increased the top similarity - likely improved recall for this query."
    },
    {
      "query": "What happens when downstream services are unavailable?",
      "expanded_query": "downstream outage, circuit breaker, retries with backoff, degraded mode, timeout budgets",
      "strategy_a": [
        {
          "rank": 1,
          "chunk_id": "chunk_009",
          "score": 0.68741,
          "text_preview": "Downstream resilience\n\nWhen downstream services are unavailable the edge gateway sheds optional traffic, applies circuit breakers per dependency, and uses exponential backoff retries only for idempotent verbs. Timeouts a",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "1059140ff33491d7",
            "section": "Downstream resilience"
          }
        },
        {
          "rank": 2,
          "chunk_id": "chunk_006",
          "score": 0.418357,
          "text_preview": "Stream processing\n\nEvent ingestion pipelines batch acknowledgements and apply backpressure when downstream sinks lag. Poison messages are quarantined after configurable retry budgets to protect overall throughput. Consum",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "b2a8cfb0673d6c98",
            "section": "Stream processing"
          }
        },
        {
          "rank": 3,
          "chunk_id": "chunk_010",
          "score": 0.348225,
          "text_preview": "Infrastructure recovery\n\nThe platform recovers from infrastructure failure by promoting warm replicas in a secondary region, replaying durable logs, and automating runbooks that validate data checksums before shifting us",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "aeb06c5f4b6cdd00",
            "section": "Infrastructure recovery"
          }
        }
      ],
      "strategy_b": [
        {
          "rank": 1,
          "chunk_id": "chunk_009",
          "score": 0.634108,
          "text_preview": "Downstream resilience\n\nWhen downstream services are unavailable the edge gateway sheds optional traffic, applies circuit breakers per dependency, and uses exponential backoff retries only for idempotent verbs. Timeouts a",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "1059140ff33491d7",
            "section": "Downstream resilience"
          }
        },
        {
          "rank": 2,
          "chunk_id": "chunk_010",
          "score": 0.358874,
          "text_preview": "Infrastructure recovery\n\nThe platform recovers from infrastructure failure by promoting warm replicas in a secondary region, replaying durable logs, and automating runbooks that validate data checksums before shifting us",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "aeb06c5f4b6cdd00",
            "section": "Infrastructure recovery"
          }
        },
        {
          "rank": 3,
          "chunk_id": "chunk_006",
          "score": 0.356416,
          "text_preview": "Stream processing\n\nEvent ingestion pipelines batch acknowledgements and apply backpressure when downstream sinks lag. Poison messages are quarantined after configurable retry budgets to protect overall throughput. Consum",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "b2a8cfb0673d6c98",
            "section": "Stream processing"
          }
        }
      ],
      "observation": "Top-1 chunk matches; compare deeper ranks and score deltas. Strategy B rank-1 score is lower than Strategy A - expansion is not universally better and can reduce precision when extra terms misalign with the corpus."
    },
    {
      "query": "How is data kept consistent during concurrent updates?",
      "expanded_query": "transactions, optimistic locking, CRDTs, vector clocks, read-your-writes, replica lag",
      "strategy_a": [
        {
          "rank": 1,
          "chunk_id": "chunk_005",
          "score": 0.535641,
          "text_preview": "Cross-region consistency\n\nCross-region writes flow through a conflict-free replicated data type for inventory counters while user profile updates use last-write-wins with vector clocks to survive partial network partitio",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "3df7f3b7dac47137",
            "section": "Cross-region consistency"
          }
        },
        {
          "rank": 2,
          "chunk_id": "chunk_004",
          "score": 0.455455,
          "text_preview": "Database read scaling\n\nRead replicas serve analytical queries and offload the primary writer. Replica lag is monitored continuously; queries that require strongly consistent reads are pinned to the primary connection poo",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "f8816f1c9a071b04",
            "section": "Database read scaling"
          }
        },
        {
          "rank": 3,
          "chunk_id": "chunk_011",
          "score": 0.412445,
          "text_preview": "Idempotency and deduplication\n\nRepeated processing of the same request is prevented by idempotency keys stored in a short-lived deduplication table; clients replaying a submission receive the original response without do",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "a28fd4935c3cc218",
            "section": "Idempotency and deduplication"
          }
        }
      ],
      "strategy_b": [
        {
          "rank": 1,
          "chunk_id": "chunk_004",
          "score": 0.623559,
          "text_preview": "Database read scaling\n\nRead replicas serve analytical queries and offload the primary writer. Replica lag is monitored continuously; queries that require strongly consistent reads are pinned to the primary connection poo",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "f8816f1c9a071b04",
            "section": "Database read scaling"
          }
        },
        {
          "rank": 2,
          "chunk_id": "chunk_005",
          "score": 0.59871,
          "text_preview": "Cross-region consistency\n\nCross-region writes flow through a conflict-free replicated data type for inventory counters while user profile updates use last-write-wins with vector clocks to survive partial network partitio",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "3df7f3b7dac47137",
            "section": "Cross-region consistency"
          }
        },
        {
          "rank": 3,
          "chunk_id": "chunk_002",
          "score": 0.447692,
          "text_preview": "Cache architecture\n\nThe edge cache uses a two-tier design: an in-memory LRU for hot keys and a regional Redis cluster for shared objects. TTLs are staggered to prevent thundering herds when popular keys expire at the sam",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "847284773a3264c5",
            "section": "Cache architecture"
          }
        }
      ],
      "observation": "Top-1 chunk differs: expansion shifted lexical overlap. Strategy B increased the top similarity - likely improved recall for this query."
    },
    {
      "query": "How does the platform recover from infrastructure failure?",
      "expanded_query": "regional failover, control plane recovery, runbook automation, disaster recovery, replica promotion",
      "strategy_a": [
        {
          "rank": 1,
          "chunk_id": "chunk_010",
          "score": 0.762936,
          "text_preview": "Infrastructure recovery\n\nThe platform recovers from infrastructure failure by promoting warm replicas in a secondary region, replaying durable logs, and automating runbooks that validate data checksums before shifting us",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "aeb06c5f4b6cdd00",
            "section": "Infrastructure recovery"
          }
        },
        {
          "rank": 2,
          "chunk_id": "chunk_009",
          "score": 0.418876,
          "text_preview": "Downstream resilience\n\nWhen downstream services are unavailable the edge gateway sheds optional traffic, applies circuit breakers per dependency, and uses exponential backoff retries only for idempotent verbs. Timeouts a",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "1059140ff33491d7",
            "section": "Downstream resilience"
          }
        },
        {
          "rank": 3,
          "chunk_id": "chunk_003",
          "score": 0.335139,
          "text_preview": "Cache failure modes\n\nWhen the primary Redis tier becomes unhealthy the application degrades gracefully by bypassing cache reads and writes, falling back to authoritative database queries while emitting elevated latency a",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "8d47c1d734d11b4c",
            "section": "Cache failure modes"
          }
        }
      ],
      "strategy_b": [
        {
          "rank": 1,
          "chunk_id": "chunk_010",
          "score": 0.663188,
          "text_preview": "Infrastructure recovery\n\nThe platform recovers from infrastructure failure by promoting warm replicas in a secondary region, replaying durable logs, and automating runbooks that validate data checksums before shifting us",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "aeb06c5f4b6cdd00",
            "section": "Infrastructure recovery"
          }
        },
        {
          "rank": 2,
          "chunk_id": "chunk_005",
          "score": 0.391332,
          "text_preview": "Cross-region consistency\n\nCross-region writes flow through a conflict-free replicated data type for inventory counters while user profile updates use last-write-wins with vector clocks to survive partial network partitio",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "3df7f3b7dac47137",
            "section": "Cross-region consistency"
          }
        },
        {
          "rank": 3,
          "chunk_id": "chunk_009",
          "score": 0.337371,
          "text_preview": "Downstream resilience\n\nWhen downstream services are unavailable the edge gateway sheds optional traffic, applies circuit breakers per dependency, and uses exponential backoff retries only for idempotent verbs. Timeouts a",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "1059140ff33491d7",
            "section": "Downstream resilience"
          }
        }
      ],
      "observation": "Top-1 chunk matches; compare deeper ranks and score deltas. Strategy B rank-1 score is lower than Strategy A - expansion is not universally better and can reduce precision when extra terms misalign with the corpus."
    },
    {
      "query": "What mechanisms prevent repeated processing of the same request?",
      "expanded_query": "idempotency keys, deduplication, exactly-once semantics, consumer offsets, poison message quarantine",
      "strategy_a": [
        {
          "rank": 1,
          "chunk_id": "chunk_011",
          "score": 0.605786,
          "text_preview": "Idempotency and deduplication\n\nRepeated processing of the same request is prevented by idempotency keys stored in a short-lived deduplication table; clients replaying a submission receive the original response without do",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "a28fd4935c3cc218",
            "section": "Idempotency and deduplication"
          }
        },
        {
          "rank": 2,
          "chunk_id": "chunk_006",
          "score": 0.404064,
          "text_preview": "Stream processing\n\nEvent ingestion pipelines batch acknowledgements and apply backpressure when downstream sinks lag. Poison messages are quarantined after configurable retry budgets to protect overall throughput. Consum",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "b2a8cfb0673d6c98",
            "section": "Stream processing"
          }
        },
        {
          "rank": 3,
          "chunk_id": "chunk_004",
          "score": 0.366109,
          "text_preview": "Database read scaling\n\nRead replicas serve analytical queries and offload the primary writer. Replica lag is monitored continuously; queries that require strongly consistent reads are pinned to the primary connection poo",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "f8816f1c9a071b04",
            "section": "Database read scaling"
          }
        }
      ],
      "strategy_b": [
        {
          "rank": 1,
          "chunk_id": "chunk_011",
          "score": 0.636525,
          "text_preview": "Idempotency and deduplication\n\nRepeated processing of the same request is prevented by idempotency keys stored in a short-lived deduplication table; clients replaying a submission receive the original response without do",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "a28fd4935c3cc218",
            "section": "Idempotency and deduplication"
          }
        },
        {
          "rank": 2,
          "chunk_id": "chunk_006",
          "score": 0.456365,
          "text_preview": "Stream processing\n\nEvent ingestion pipelines batch acknowledgements and apply backpressure when downstream sinks lag. Poison messages are quarantined after configurable retry budgets to protect overall throughput. Consum",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "b2a8cfb0673d6c98",
            "section": "Stream processing"
          }
        },
        {
          "rank": 3,
          "chunk_id": "chunk_005",
          "score": 0.350232,
          "text_preview": "Cross-region consistency\n\nCross-region writes flow through a conflict-free replicated data type for inventory counters while user profile updates use last-write-wins with vector clocks to survive partial network partitio",
          "metadata": {
            "source": "technical_paragraphs.json",
            "content_hash": "3df7f3b7dac47137",
            "section": "Cross-region consistency"
          }
        }
      ],
      "observation": "Top-1 chunk matches; compare deeper ranks and score deltas. Strategy B increased the top similarity - likely improved recall for this query."
    }
  ]
}

```

## Conclusion

Strategy B improved several ambiguous operational queries because expansions added domain terms (HPA, circuit breaker, CRDTs, idempotency). In other cases rank-1 stayed the same while deeper ranks or scores shifted; in at least one query Strategy B's top similarity was lower than Strategy A - expansion can reduce precision when extra terms misalign with the corpus. In production I would monitor Recall@k, MRR, top-score distributions, no-result rates, and expansion quality; use metadata filters and ACLs; and apply score thresholds before passing context to an LLM.
