"""Tests for /query, /runtime, /report/pdf endpoints and backward
compatibility of the existing endpoints. Pipeline and stores are mocked."""

import pytest
from fastapi.testclient import TestClient

from agents import orchestrator
from agents.schemas import (
    CritiqueResult,
    EvidenceBundle,
    EvidenceSource,
    Report,
    RetrievalSummary,
    ResearchPlan,
)
from main import app


def _source(n, source_type="web"):
    return EvidenceSource(
        source_type=source_type,
        title=f"Source {n}",
        url=f"https://example.com/{n}" if source_type == "web" else None,
        document_name=f"doc{n}.txt" if source_type == "kb" else None,
        content=f"Evidence text {n}. " * 4,
        score=0.9,
        source_id=f"https://example.com/{n}" if source_type == "web" else f"doc{n}:0",
    )


def _report():
    md = (
        "# Findings\n\n## Executive summary\nAI matters [1].\n\n"
        "## References\n[1] Source 1"
    )
    return Report(title="AI in Healthcare", markdown=md, references=[])


@pytest.fixture
def pipeline(monkeypatch):
    captured = {}

    def plan_fn(q):
        return ResearchPlan(original_query=q, sub_questions=["a?", "b?", "c?"])

    def retrieve_fn(plan, mode="both"):
        captured["mode"] = mode
        return EvidenceBundle(
            original_query=plan.original_query,
            sources=[_source(1), _source(2)],
            retrieval_summary=RetrievalSummary(
                total_sources=2, web_sources=2, kb_sources=0,
                deduplicated_sources=0, kb_status="empty", web_status="ok",
            ),
        )

    def writer_fn(plan, evidence, previous_report=None, critique=None):
        return _report()

    def critic_fn(plan, evidence, report, iteration=0):
        return CritiqueResult(status="approved", iteration=iteration)

    monkeypatch.setattr(orchestrator, "create_research_plan", plan_fn)
    monkeypatch.setattr(
        orchestrator, "retrieve_for_plan",
        lambda p, mode="both": retrieve_fn(p, mode=mode),
    )
    monkeypatch.setattr(orchestrator, "write_report", writer_fn)
    monkeypatch.setattr(orchestrator, "evaluate_report", critic_fn)
    return captured


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

def test_query_returns_full_result_with_evaluation_and_timing(client, pipeline):
    resp = client.post("/query", json={"query": "Research AI healthcare"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["final_status"] == "approved"
    assert body["report"]["title"] == "AI in Healthcare"
    # Phase 6/23: evaluation and timing ride along on the result.
    assert body["evaluation"] is not None
    assert 0.0 <= body["evaluation"]["grounding_score"] <= 1.0
    assert body["evaluation"]["total_sources"] == 2
    assert body["timing"] is not None
    assert body["timing"]["total_seconds"] >= 0.0
    assert body["retrieval_mode"] == "both"


def test_query_passes_retrieval_mode(client, pipeline):
    resp = client.post(
        "/query", json={"query": "q", "retrieval_mode": "kb"}
    )
    assert resp.status_code == 200
    assert resp.json()["retrieval_mode"] == "kb"
    assert pipeline["mode"] == "kb"


def test_query_rejects_invalid_mode(client, pipeline):
    resp = client.post(
        "/query", json={"query": "q", "retrieval_mode": "all"}
    )
    assert resp.status_code == 422


def test_query_rejects_empty_query(client, pipeline):
    resp = client.post("/query", json={"query": "   "})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

def test_research_still_works_with_default_mode(client, pipeline):
    resp = client.post("/research", json={"query": "Research AI"})
    assert resp.status_code == 200
    assert resp.json()["final_status"] == "approved"


def test_research_accepts_retrieval_mode(client, pipeline):
    resp = client.post(
        "/research", json={"query": "Research AI", "retrieval_mode": "web"}
    )
    assert resp.status_code == 200
    assert pipeline["mode"] == "web"


def test_existing_endpoints_still_registered():
    paths = {r.path for r in app.routes if hasattr(r, "methods")}
    assert {
        "/health", "/plan", "/retrieve", "/write", "/research",
        "/research/stream", "/query", "/runtime", "/report/pdf",
        "/kb/upload", "/kb/documents", "/kb/status",
        "/kb/documents/{document_id}",
    } <= paths


def test_health_unchanged(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "AgentDesk", "running": True}


def test_runtime_exposes_no_secrets(client):
    resp = client.get("/runtime")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"]
    assert isinstance(body["groq_configured"], bool)
    assert isinstance(body["tavily_configured"], bool)
    assert "both" in body["retrieval_modes"]
    text = str(body).lower()
    for secret_marker in ("gsk_", "tvly-", "api_key", "authorization", "bearer"):
        assert secret_marker not in text


# ---------------------------------------------------------------------------
# POST /report/pdf
# ---------------------------------------------------------------------------

def test_pdf_generation_valid(client):
    resp = client.post(
        "/report/pdf",
        json={
            "report": {
                "title": "AI in Healthcare",
                "markdown": (
                    "# AI in Healthcare\n\n## Executive summary\n"
                    "AI adoption is accelerating across India [1].\n\n"
                    "## Findings\n- Diagnostics lead adoption [1].\n\n"
                    "## References\n[1] Source 1 — https://example.com"
                ),
                "references": [],
            },
            "query": "Research the impact of AI on healthcare in India",
            "retrieval_mode": "both",
            "final_status": "approved",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert "attachment" in resp.headers["content-disposition"]
    data = resp.content
    assert data.startswith(b"%PDF-")
    assert len(data) > 500


def test_pdf_empty_report_fails_safely(client):
    resp = client.post(
        "/report/pdf",
        json={"report": {"title": "", "markdown": "", "references": []}},
    )
    assert resp.status_code == 422


def test_pdf_unsafe_chars_do_not_break_rendering(client):
    resp = client.post(
        "/report/pdf",
        json={
            "report": {
                "title": "Résumé of AI — 東京 emoji 🚀 report",
                "markdown": "# Título\n\nEm—dash … and “quotes” [1].\n",
                "references": [],
            },
        },
    )
    assert resp.status_code == 200
    assert resp.content.startswith(b"%PDF-")
