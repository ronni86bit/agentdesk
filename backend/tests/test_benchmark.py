"""Tests for the benchmark harness. The pipeline is patched — no live calls."""

import json

import pytest

import eval.benchmark as benchmark
from agents.schemas import (
    CritiqueResult,
    EvidenceBundle,
    EvidenceSource,
    Report,
    RetrievalSummary,
    ResearchPlan,
    ResearchRunResult,
    RunTiming,
)


def _fake_result(query, mode="both"):
    source = EvidenceSource(
        source_type="web",
        title="S1",
        url="https://example.com/1",
        content="Evidence text. " * 5,
        score=0.9,
        source_id="https://example.com/1",
    )
    return ResearchRunResult(
        query=query,
        plan=ResearchPlan(original_query=query, sub_questions=["a?", "b?", "c?"]),
        evidence=EvidenceBundle(
            original_query=query,
            sources=[source],
            retrieval_summary=RetrievalSummary(
                total_sources=1, web_sources=1, kb_sources=0,
                deduplicated_sources=0, kb_status="empty", web_status="ok",
            ),
        ),
        report=Report(
            title="Report",
            markdown="# R\nA fully cited claim sentence right here [1].\n\n## References\n[1] S1",
            references=[],
        ),
        critique=CritiqueResult(status="approved", iteration=0),
        iteration_count=0,
        final_status="approved",
        retrieval_mode=mode,
        timing=RunTiming(total_seconds=1.25),
    )


def test_run_benchmark_collects_metrics(monkeypatch):
    calls = []

    def fake_pipeline(query, event_callback=None, retrieval_mode="both"):
        calls.append((query, retrieval_mode))
        return _fake_result(query, retrieval_mode)

    monkeypatch.setattr(benchmark, "run_research_pipeline", fake_pipeline)
    records = benchmark.run_benchmark(
        queries=["query one", "query two"], retrieval_mode="web"
    )
    assert len(records) == 2
    assert calls == [("query one", "web"), ("query two", "web")]
    record = records[0]
    assert record["final_status"] == "approved"
    assert record["grounding_score"] == pytest.approx(1.0)
    assert record["citation_coverage"] == 1.0
    assert record["total_seconds"] == pytest.approx(1.25)
    assert record["error"] is None


def test_run_benchmark_records_failure_without_aborting(monkeypatch):
    def fake_pipeline(query, event_callback=None, retrieval_mode="both"):
        if query == "bad":
            raise RuntimeError("LLM request returned HTTP 429")
        return _fake_result(query)

    monkeypatch.setattr(benchmark, "run_research_pipeline", fake_pipeline)
    records = benchmark.run_benchmark(queries=["good", "bad"])
    assert len(records) == 2
    failed = [r for r in records if r["error"]]
    assert len(failed) == 1
    assert "429" in failed[0]["error"]

    summary = benchmark.summarize(records)
    assert summary["queries"] == 2
    assert summary["completed"] == 1
    assert summary["failed"] == 1
    assert summary["approved"] == 1
    assert summary["avg_grounding_score"] == pytest.approx(1.0)


def test_save_results_writes_json(tmp_path, monkeypatch):
    monkeypatch.setattr(
        benchmark, "run_research_pipeline",
        lambda query, event_callback=None, retrieval_mode="both": _fake_result(query),
    )
    records = benchmark.run_benchmark(queries=["saved query"])
    summary = benchmark.summarize(records)
    path = benchmark.save_results(records, summary, out_dir=tmp_path)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["summary"]["completed"] == 1
    assert data["records"][0]["query"] == "saved query"
