"""Benchmark JSON + Markdown (plan §13)."""

from __future__ import annotations
import json
from pathlib import Path

from rag_engine.app import RAGEngine
from rag_engine.config.loader import load_engine_config
from rag_engine.evaluation.benchmark import benchmark_results_to_jsonable, run_strategy_benchmark
from rag_engine.evaluation.reporter import write_benchmark_json, write_benchmark_markdown


def test_benchmark_writes_json_and_md(tmp_path: Path) -> None:
    cfg = load_engine_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    cfg.embedding.provider = "mock_vertex"
    cfg.embedding.mock_vertex = cfg.embedding.mock_vertex.model_copy(update={"dimensions": 16})
    cfg.vector_store.provider = "numpy"
    cfg.benchmark.queries = ["How does the system handle peak load?"]
    cfg.benchmark.output_json = tmp_path / "bench.json"
    md_out = tmp_path / "bench.md"

    eng = RAGEngine(config=cfg)
    eng.ingest(Path(__file__).resolve().parents[1] / "data" / "technical_paragraphs.json")
    results = run_strategy_benchmark(eng.retriever, cfg, queries=cfg.benchmark.queries)
    rows = benchmark_results_to_jsonable(results)
    write_benchmark_json(cfg.benchmark.output_json, rows)
    write_benchmark_markdown(results, md_out, cfg)

    data = json.loads(cfg.benchmark.output_json.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert "strategy_a" in data[0]
    assert "comparison" in data[0]
    assert "overlap_count" in data[0]["comparison"]
    body = md_out.read_text(encoding="utf-8")
    assert "Strategy A" in body and "Strategy B" in body
