"""Tests for retrieval modes (both/web/kb) across the retrieval layer,
orchestrator, and API. All external services are mocked."""

import pytest

import tools.retrieval as retrieval
from agents import orchestrator
from agents.orchestrator import run_research_pipeline
from agents.schemas import (
    CritiqueResult,
    EvidenceBundle,
    EvidenceSource,
    RetrievalSummary,
    ResearchPlan,
)
from tools.retrieval import RetrievalError, retrieve_evidence


def _web_source(**overrides):
    base = dict(
        source_type="web",
        title="Web result",
        url="https://example.com/a",
        content="web content",
        score=0.9,
    )
    return EvidenceSource(**{**base, **overrides})


def _kb_source(**overrides):
    base = dict(
        source_type="kb",
        title="KB chunk",
        document_name="doc.txt",
        content="kb content",
        score=0.8,
        source_id="doc1:0",
    )
    return EvidenceSource(**{**base, **overrides})


@pytest.fixture
def searches(monkeypatch):
    calls = {"kb": 0, "web": 0}

    def kb(query, limit=5):
        calls["kb"] += 1
        return [_kb_source()]

    def web(query, limit=5):
        calls["web"] += 1
        return [_web_source()]

    monkeypatch.setattr(retrieval, "search_qdrant", kb)
    monkeypatch.setattr(retrieval, "search_tavily", web)
    return calls


def test_mode_web_only_queries_tavily(searches):
    bundle = retrieve_evidence(["q"], mode="web")
    assert searches["web"] == 1
    assert searches["kb"] == 0
    assert bundle.retrieval_summary.web_status == "ok"
    assert bundle.retrieval_summary.kb_status == "skipped"
    assert all(s.source_type == "web" for s in bundle.sources)


def test_mode_kb_only_queries_qdrant(searches):
    bundle = retrieve_evidence(["q"], mode="kb")
    assert searches["kb"] == 1
    assert searches["web"] == 0
    assert bundle.retrieval_summary.kb_status == "ok"
    assert bundle.retrieval_summary.web_status == "skipped"
    assert all(s.source_type == "kb" for s in bundle.sources)


def test_mode_both_queries_both(searches):
    bundle = retrieve_evidence(["q"], mode="both")
    assert searches["kb"] == 1
    assert searches["web"] == 1
    assert bundle.retrieval_summary.kb_status == "ok"
    assert bundle.retrieval_summary.web_status == "ok"


def test_default_mode_is_both(searches):
    retrieve_evidence(["q"])
    assert searches["kb"] == 1 and searches["web"] == 1


def test_invalid_mode_rejected(searches):
    with pytest.raises(ValueError, match="mode"):
        retrieve_evidence(["q"], mode="twitter")


def test_mode_web_failure_is_controlled_failure(monkeypatch):
    def web_fail(query, limit=5):
        raise retrieval.TavilyUnavailableError("down")

    monkeypatch.setattr(retrieval, "search_tavily", web_fail)
    with pytest.raises(RetrievalError, match="[Ww]eb"):
        retrieve_evidence(["q"], mode="web")


def test_mode_kb_failure_is_controlled_failure(monkeypatch):
    def kb_fail(query, limit=5):
        raise retrieval.QdrantUnavailableError("down")

    monkeypatch.setattr(retrieval, "search_qdrant", kb_fail)
    with pytest.raises(RetrievalError, match="[Kk]nowledge"):
        retrieve_evidence(["q"], mode="kb")


def test_mode_both_still_requires_both_to_fail(monkeypatch):
    def kb_fail(query, limit=5):
        raise retrieval.QdrantUnavailableError("down")

    monkeypatch.setattr(retrieval, "search_qdrant", kb_fail)
    monkeypatch.setattr(
        retrieval, "search_tavily", lambda query, limit=5: [_web_source()]
    )
    # KB down but web works: BOTH mode continues with web only.
    bundle = retrieve_evidence(["q"], mode="both")
    assert bundle.retrieval_summary.kb_status == "unavailable"
    assert bundle.retrieval_summary.web_status == "ok"


# ---------------------------------------------------------------------------
# Orchestrator mode propagation (all agents mocked)
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline_mocks(monkeypatch):
    captured = {"modes": [], "result_modes": []}

    def plan_fn(q):
        return ResearchPlan(
            original_query=q, sub_questions=["q1?", "q2?", "q3?"]
        )

    def retrieve_fn(plan, mode="both"):
        captured["modes"].append(mode)
        return EvidenceBundle(
            original_query=plan.original_query,
            sources=[_web_source(), _kb_source()],
            retrieval_summary=RetrievalSummary(
                total_sources=2, web_sources=1, kb_sources=1,
                deduplicated_sources=0, kb_status="ok", web_status="ok",
            ),
        )

    def writer_fn(plan, evidence, previous_report=None, critique=None):
        from agents.schemas import Report

        return Report(
            title="T",
            markdown="# T\nClaim [1].\n\n## References\n[1] x",
            references=[],
        )

    def critic_fn(plan, evidence, report, iteration=0):
        return CritiqueResult(status="approved", iteration=iteration)

    monkeypatch.setattr(orchestrator, "create_research_plan", plan_fn)
    monkeypatch.setattr(
        orchestrator, "retrieve_for_plan",
        lambda p, mode="both": retrieve_fn(p, mode=mode),
    )
    monkeypatch.setattr(orchestrator, "write_report", writer_fn)
    monkeypatch.setattr(orchestrator, "evaluate_report", critic_fn)
    captured["run"] = run_research_pipeline
    return captured


@pytest.mark.parametrize("mode", ["both", "web", "kb"])
def test_mode_flows_to_retriever_and_result(pipeline_mocks, mode):
    result = pipeline_mocks["run"]("some query", retrieval_mode=mode)
    assert pipeline_mocks["modes"] == [mode]
    assert result.retrieval_mode == mode
    assert result.final_status == "approved"


def test_default_pipeline_mode_is_both(pipeline_mocks):
    result = pipeline_mocks["run"]("some query")
    assert pipeline_mocks["modes"] == ["both"]
    assert result.retrieval_mode == "both"


def test_pipeline_rejects_invalid_mode(pipeline_mocks):
    with pytest.raises(ValueError, match="mode"):
        pipeline_mocks["run"]("q", retrieval_mode="everything")


def test_result_with_mode_is_serializable(pipeline_mocks):
    import json

    result = pipeline_mocks["run"]("q", retrieval_mode="kb")
    payload = json.loads(result.model_dump_json())
    assert payload["retrieval_mode"] == "kb"
