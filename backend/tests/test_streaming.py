"""Tests for SSE streaming: event schema, orchestrator callback, endpoint.

All LLM/network calls are mocked. The real pipeline is only exercised by the
separate manual integration run.
"""

import json
import re

import pytest
from fastapi.testclient import TestClient

from agents import orchestrator
from agents.events import EventNames, ResearchEvent, format_sse, safe_failure
from agents.orchestrator import run_research_pipeline
from agents.schemas import (
    CritiqueResult,
    EvidenceBundle,
    EvidenceSource,
    Report,
    ReportReference,
    RetrievalSummary,
    ResearchPlan,
)
from agents.writer import WriterError
from main import app


# ---------------------------------------------------------------------------
# Helpers / fixtures (same shape as test_orchestrator)
# ---------------------------------------------------------------------------

def _source(n: int):
    return EvidenceSource(
        source_type="web",
        title=f"Source {n}",
        url=f"https://example.com/{n}",
        content=f"Evidence text {n}. " * 5,
        score=0.9 - n * 0.1,
        source_id=f"https://example.com/{n}",
    )


def _bundle():
    sources = [_source(1), _source(2)]
    return EvidenceBundle(
        original_query="goal",
        sources=sources,
        retrieval_summary=RetrievalSummary(
            total_sources=2,
            web_sources=2,
            kb_sources=0,
            deduplicated_sources=1,
            kb_status="empty",
            web_status="ok",
        ),
    )


def _plan():
    return ResearchPlan(original_query="goal", sub_questions=["q1?", "q2?", "q3?"])


def _report(tag="r0"):
    md = f"# Report {tag}\nClaim [1].\n\n## References\n[1] Source 1"
    return Report(
        title=f"Report {tag}",
        markdown=md,
        references=[ReportReference(citation_id=1, source_type="web", title="Source 1")],
    )


def _approved(iteration=0):
    return CritiqueResult(
        status="approved", iteration=iteration, overall_assessment="OK."
    )


def _revision_needed(iteration=0):
    return CritiqueResult(
        status="revision_needed",
        iteration=iteration,
        unsupported_claims=["Claim X is unsupported."],
        revision_instructions=["Remove claim X."],
        overall_assessment="Needs fixes.",
    )


@pytest.fixture
def pipeline_mocks(monkeypatch):
    class Recorder:
        def __init__(self):
            self.plan = _plan()
            self.evidence = _bundle()
            self.writer_calls = 0
            self.critic_verdicts = []
            self.retriever_modes = []

        def record_plan(self, query):
            return self.plan

        def retrieve(self, plan, mode="both"):
            self.retriever_modes.append(mode)
            return self.evidence

        def writer(self, plan, evidence, previous_report=None, critique=None):
            report = _report(f"r{self.writer_calls}")
            self.writer_calls += 1
            return report

        def critic(self, plan, evidence, report, iteration=0):
            return self.critic_verdicts.pop(0)

    rec = Recorder()
    monkeypatch.setattr(orchestrator, "create_research_plan", lambda q: rec.record_plan(q))
    monkeypatch.setattr(
        orchestrator, "retrieve_for_plan",
        lambda p, mode="both": rec.retrieve(p, mode=mode),
    )
    monkeypatch.setattr(orchestrator, "write_report", rec.writer)
    monkeypatch.setattr(orchestrator, "evaluate_report", rec.critic)
    yield rec


def collect_events(callback_list):
    def cb(event):
        callback_list.append(event)
    return cb


def names(events):
    return [e.event for e in events]


# ---------------------------------------------------------------------------
# Event schema
# ---------------------------------------------------------------------------

def test_event_valid_with_all_fields():
    evt = ResearchEvent(
        event=EventNames.PLANNER_STARTED,
        stage="planner",
        status="started",
        message="Planning research questions",
        data={"sub_questions": 3},
        iteration=0,
    )
    assert evt.event == "planner_started"
    assert evt.data == {"sub_questions": 3}


def test_event_optional_fields_default_none():
    evt = ResearchEvent(event=EventNames.RESEARCH_STARTED)
    assert evt.stage is None
    assert evt.status is None
    assert evt.message is None
    assert evt.data is None
    assert evt.iteration is None


def test_event_timestamp_generated_and_parseable():
    from datetime import datetime

    evt = ResearchEvent(event="x")
    parsed = datetime.fromisoformat(evt.timestamp)
    assert parsed.tzinfo is not None


def test_event_json_serialization():
    evt = ResearchEvent(
        event="writer_completed",
        stage="writer",
        data={"citation_count": 5},
    )
    payload = json.loads(evt.model_dump_json())
    assert payload["event"] == "writer_completed"
    assert payload["stage"] == "writer"
    assert payload["data"]["citation_count"] == 5
    assert "timestamp" in payload


def test_format_sse_framing():
    evt = ResearchEvent(event="planner_started", message="hi")
    frame = format_sse(evt)
    assert frame.startswith("event: planner_started\n")
    assert "\ndata: " in frame
    assert frame.endswith("\n\n")
    data_line = frame.split("\ndata: ", 1)[1].rsplit("\n\n", 1)[0]
    parsed = json.loads(data_line)
    assert parsed["event"] == "planner_started"


# ---------------------------------------------------------------------------
# Orchestrator callback behavior
# ---------------------------------------------------------------------------

EXPECTED_BASE_SEQUENCE = [
    "research_started",
    "planner_started",
    "planner_completed",
    "retrieval_started",
    "retrieval_completed",
    "writer_started",
    "writer_completed",
    "critic_started",
    "critic_completed",
    "research_completed",
]


def test_case1_first_pass_approval_event_sequence(pipeline_mocks):
    pipeline_mocks.critic_verdicts = [_approved()]
    events = []
    result = run_research_pipeline("some query", event_callback=collect_events(events))

    assert names(events) == EXPECTED_BASE_SEQUENCE
    assert result.final_status == "approved"

    started = events[0]
    assert started.data == {"query": "some query", "retrieval_mode": "both"}

    writer_done = events[6]
    assert writer_done.data["citation_count"] >= 1
    assert writer_done.data["report_title"] == "Report r0"

    critic_done = events[8]
    assert critic_done.iteration == 0
    assert critic_done.data["status"] == "approved"
    assert critic_done.data["issue_count"] == 0

    completed = events[-1]
    assert completed.data["final_status"] == "approved"
    assert completed.data["iteration_count"] == 0
    assert completed.data["result"]["report"]["title"] == "Report r0"


def test_case2_revision_events_in_correct_positions(pipeline_mocks):
    pipeline_mocks.critic_verdicts = [_revision_needed(), _approved()]
    events = []
    result = run_research_pipeline("some query", event_callback=collect_events(events))

    order = names(events)
    assert order.index("revision_started") > order.index("critic_completed")
    # Exactly one revision cycle before completion.
    assert order.count("revision_started") == 1
    assert order.count("revision_completed") == 1
    assert order.index("revision_started") < order.index("revision_completed")
    assert order.index("revision_completed") < order.index("critic_started", order.index("revision_completed"))
    assert order[-1] == "research_completed"

    revision_start = [e for e in events if e.event == "revision_started"][0]
    revision_end = [e for e in events if e.event == "revision_completed"][0]
    assert revision_start.iteration == 1
    assert revision_end.iteration == 1
    assert result.iteration_count == 1
    assert result.final_status == "approved"


def test_case2b_two_revisions_event_sequence(pipeline_mocks):
    pipeline_mocks.critic_verdicts = [
        _revision_needed(0),
        _revision_needed(1),
        _approved(2),
    ]
    events = []
    run_research_pipeline("some query", event_callback=collect_events(events))

    order = names(events)
    assert order.count("revision_started") == 2
    starts = [i for i, n in enumerate(order) if n == "revision_started"]
    ends = [i for i, n in enumerate(order) if n == "revision_completed"]
    critic_completes = [i for i, n in enumerate(order) if n == "critic_completed"]
    # Both revision cycles sit between their triggering critic and next critic.
    assert starts[0] < ends[0] < critic_completes[1]
    assert starts[1] < ends[1] < critic_completes[2]
    assert order[-1] == "research_completed"


def test_case3_failure_emits_research_failed(pipeline_mocks):
    def boom(plan, evidence, previous_report=None, critique=None):
        raise WriterError("Writer execution failed")

    events = []
    original = orchestrator.write_report
    orchestrator.write_report = boom
    try:
        with pytest.raises(WriterError):
            run_research_pipeline("some query", event_callback=collect_events(events))
    finally:
        orchestrator.write_report = original

    order = names(events)
    assert order[-1] == "research_failed"
    failed = events[-1]
    assert failed.data["error_code"] == "writer_failed"
    # The internal exception detail must never appear in the event payload.
    assert "Writer execution failed" not in failed.data["message"]


def test_no_callback_still_works_silently(pipeline_mocks):
    pipeline_mocks.critic_verdicts = [_approved()]
    result = run_research_pipeline("some query")
    assert result.final_status == "approved"


def test_safe_failure_mapping_is_user_safe():
    from agents.planner import MissingGroqKeyError as PlannerMissingKey

    code, msg = safe_failure(PlannerMissingKey("LLM provider is not configured."))
    assert code == "llm_not_configured"

    code, msg = safe_failure(ValueError("query must be a non-empty string"))
    assert code == "invalid_request"

    code, _ = safe_failure(RuntimeError("secret internal detail xyz"))
    assert code == "internal_error"


# ---------------------------------------------------------------------------
# SSE endpoint (TestClient; pipeline mocked at the main-module boundary)
# ---------------------------------------------------------------------------

def parse_sse_frames(text: str):
    """Parse an SSE byte stream into [(event_name, payload_dict), ...]."""
    frames = []
    for block in re.split(r"\n\n+", text):
        block = block.strip()
        if not block or block.startswith(":"):
            continue
        m_event = re.search(r"^event: (.+)$", block, re.MULTILINE)
        m_data = re.search(r"^data: (.+)$", block, re.MULTILINE)
        if m_event and m_data:
            frames.append((m_event.group(1), json.loads(m_data.group(1))))
    return frames


@pytest.fixture
def sse_client(monkeypatch, pipeline_mocks):
    """TestClient whose /research/stream uses the already-mocked pipeline.

    `run_research_pipeline` is imported into main's namespace at call time via
    the orchestrator patching above (main calls the orchestrator function).
    """
    return TestClient(app)


def test_stream_endpoint_success(sse_client, pipeline_mocks):
    pipeline_mocks.critic_verdicts = [_approved()]
    with sse_client.stream("GET", "/research/stream", params={"query": "some query"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(chunk for chunk in resp.iter_text())

    frames = parse_sse_frames(body)
    got_names = [name for name, _ in frames]
    assert got_names == EXPECTED_BASE_SEQUENCE

    final_name, final_payload = frames[-1]
    assert final_name == "research_completed"
    assert final_payload["data"]["final_status"] == "approved"
    assert final_payload["data"]["result"]["final_status"] == "approved"
    assert final_payload["data"]["result"]["report"]["title"] == "Report r0"


def test_stream_endpoint_with_revision(sse_client, pipeline_mocks):
    pipeline_mocks.critic_verdicts = [_revision_needed(), _approved()]
    with sse_client.stream("GET", "/research/stream", params={"query": "q"}) as resp:
        body = "".join(chunk for chunk in resp.iter_text())

    got_names = [name for name, _ in parse_sse_frames(body)]
    assert "revision_started" in got_names
    assert "revision_completed" in got_names
    assert got_names[-1] == "research_completed"
    assert got_names.index("revision_started") > got_names.index("critic_completed")


def test_stream_endpoint_pipeline_failure_emits_failed_event(
    sse_client, pipeline_mocks, monkeypatch
):
    from agents import orchestrator as orch_mod

    def boom(query, event_callback=None, retrieval_mode="both"):
        def cb(event):
            if event_callback:
                event_callback(event)

        from agents.events import ResearchEvent

        cb(ResearchEvent(event=EventNames.RESEARCH_STARTED, data={"query": query}))
        cb(
            ResearchEvent(
                event=EventNames.RESEARCH_FAILED,
                status="failed",
                data={"error_code": "internal_error", "message": "Research failed unexpectedly."},
            )
        )
        raise RuntimeError("hidden details")

    monkeypatch.setattr(orch_mod, "run_research_pipeline", boom, raising=True)
    # main.py imported run_research_pipeline directly; patch that binding too.
    import main as main_mod

    monkeypatch.setattr(main_mod, "run_research_pipeline", boom, raising=True)

    with sse_client.stream("GET", "/research/stream", params={"query": "q"}) as resp:
        assert resp.status_code == 200
        body = "".join(chunk for chunk in resp.iter_text())

    frames = parse_sse_frames(body)
    assert frames[-1][0] == "research_failed"
    assert frames[-1][1]["data"]["error_code"] == "internal_error"
    assert "hidden details" not in body


def test_stream_endpoint_empty_query_rejected():
    client = TestClient(app)
    resp = client.get("/research/stream", params={"query": "   "})
    assert resp.status_code == 422


def test_existing_endpoints_still_registered():
    paths = {r.path for r in app.routes if hasattr(r, "methods")}
    assert {
        "/health", "/plan", "/retrieve", "/write", "/research",
        "/research/stream",
    } <= paths
