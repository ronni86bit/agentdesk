"""Tests for the research pipeline orchestrator. All agents are mocked."""

import pytest

from agents import orchestrator
from agents.orchestrator import MAX_REVISIONS, run_research_pipeline
from agents.schemas import (
    CritiqueResult,
    EvidenceBundle,
    EvidenceSource,
    Report,
    RetrievalSummary,
    ResearchPlan,
)
from agents.writer import WriterError


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _source(n: int, **overrides):
    base = dict(
        source_type="web",
        title=f"Source {n}",
        url=f"https://example.com/{n}",
        content=f"Evidence text {n}. " * 5,
        score=0.9 - n * 0.1,
        source_id=f"https://example.com/{n}",
    )
    return EvidenceSource(**{**base, **overrides})


def _bundle():
    sources = [_source(1), _source(2)]
    return EvidenceBundle(
        original_query="goal",
        sources=sources,
        retrieval_summary=RetrievalSummary(
            total_sources=2,
            web_sources=2,
            kb_sources=0,
            deduplicated_sources=0,
            kb_status="empty",
            web_status="ok",
        ),
    )


def _plan():
    return ResearchPlan(original_query="goal", sub_questions=["q1?", "q2?", "q3?"])


def _report(tag="r0", citations="[1]"):
    md = f"# Report {tag}\nClaim {citations}.\n\n## References\n[1] Source 1"
    return Report(title=f"Report {tag}", markdown=md, references=[])


def _approved(iteration=0):
    return CritiqueResult(
        status="approved", iteration=iteration, overall_assessment="OK."
    )


def _revision_needed(iteration=0):
    return CritiqueResult(
        status="revision_needed",
        iteration=iteration,
        unsupported_claims=["Claim X is unsupported by provided evidence."],
        revision_instructions=["Remove claim X."],
        overall_assessment="Needs fixes.",
    )


@pytest.fixture
def pipeline_mocks(monkeypatch):
    """Patch planner/retriever/writer/critic inside the orchestrator module.

    Returns a recorder exposing call lists and configurable side effects.
    """

    class Recorder:
        def __init__(self):
            self.plan = _plan()
            self.evidence = _bundle()
            self.planner_calls = 0
            self.retriever_calls = 0
            self.retriever_modes = []
            self.writer_args = []
            self.critic_args = []
            self.writer_side_effect = None  # callable or exception
            self.critic_side_effect = []    # list of CritiqueResults

        def record_plan(self, query):
            self.planner_calls += 1
            return self.plan

        def retrieve(self, plan, mode="both"):
            self.retriever_calls += 1
            self.retriever_modes.append(mode)
            return self.evidence

        def writer(self, plan, evidence, previous_report=None, critique=None):
            self.writer_args.append(
                {"plan": plan, "evidence": evidence,
                 "previous_report": previous_report, "critique": critique}
            )
            if isinstance(self.writer_side_effect, Exception):
                raise self.writer_side_effect
            if self.writer_side_effect is not None:
                return self.writer_side_effect(
                    plan, evidence, previous_report=previous_report,
                    critique=critique,
                )
            tag = f"r{len(self.writer_args) - 1}"
            return _report(tag)

        def critic(self, plan, evidence, report, iteration=0):
            self.critic_args.append({"iteration": iteration, "report": report})
            result = self.critic_side_effect.pop(0)
            return result

    rec = Recorder()

    monkeypatch.setattr(
        orchestrator, "create_research_plan", lambda q: rec.record_plan(q)
    )
    monkeypatch.setattr(
        orchestrator, "retrieve_for_plan", lambda p, mode="both": rec.retrieve(p, mode=mode)
    )
    monkeypatch.setattr(orchestrator, "write_report", rec.writer)
    monkeypatch.setattr(orchestrator, "evaluate_report", rec.critic)
    yield rec


def _set_verdicts(rec, *critiques):
    rec.critic_side_effect = list(critiques)


# ---------------------------------------------------------------------------
# Orchestrator loop
# ---------------------------------------------------------------------------

def test_case1_first_pass_approval(pipeline_mocks):
    _set_verdicts(pipeline_mocks, _approved())
    result = run_research_pipeline("some query")

    assert result.iteration_count == 0
    assert result.final_status == "approved"
    assert result.report.title == "Report r0"
    assert len(pipeline_mocks.writer_args) == 1
    assert pipeline_mocks.writer_args[0]["previous_report"] is None
    assert result.critique.status == "approved"


def test_case2_one_revision_then_approval(pipeline_mocks):
    _set_verdicts(pipeline_mocks, _revision_needed(0), _approved(1))
    result = run_research_pipeline("some query")

    assert result.iteration_count == 1
    assert result.final_status == "approved"
    assert len(pipeline_mocks.writer_args) == 2
    # Revision received the previous report and the critique.
    assert pipeline_mocks.writer_args[1]["previous_report"].title == "Report r0"
    assert pipeline_mocks.writer_args[1]["critique"].status == "revision_needed"


def test_case3_two_revisions_then_approval(pipeline_mocks):
    _set_verdicts(
        pipeline_mocks, _revision_needed(0), _revision_needed(1), _approved(2)
    )
    result = run_research_pipeline("some query")

    assert result.iteration_count == 2
    assert result.final_status == "approved"
    assert len(pipeline_mocks.writer_args) == 3
    assert len(pipeline_mocks.critic_args) == 3
    assert [c["iteration"] for c in pipeline_mocks.critic_args] == [0, 1, 2]


def test_case4_maximum_revisions_reached_stops_loop(pipeline_mocks):
    verdicts = [_revision_needed(0), _revision_needed(1), _revision_needed(2)]
    _set_verdicts(pipeline_mocks, *verdicts)
    result = run_research_pipeline("some query")

    assert result.iteration_count == MAX_REVISIONS == 2
    assert result.final_status == "max_revisions_reached"
    # Exactly 3 writer calls (initial + 2 revisions); no third revision.
    assert len(pipeline_mocks.writer_args) == 3
    assert len(pipeline_mocks.critic_args) == 3
    assert result.critique.status == "revision_needed"


def test_evidence_bundle_unchanged_across_revisions(pipeline_mocks):
    _set_verdicts(pipeline_mocks, _revision_needed(0), _revision_needed(1),
                  _approved(2))
    result = run_research_pipeline("some query")

    evidence_ids = {id(a["evidence"]) for a in pipeline_mocks.writer_args}
    assert len(evidence_ids) == 1
    assert evidence_ids == {id(result.evidence)}
    assert result.evidence.sources == pipeline_mocks.evidence.sources


def test_case5_invalid_revision_citation_rejected_safely(pipeline_mocks):
    # Revision attempt produces a Writer failure (invalid citation rejected
    # by the deterministic validator). Pipeline must keep the last valid
    # report and never expose invalid references.
    original_valid = _report("r0")
    calls = {"n": 0}

    def writer_behavior(plan, evidence, previous_report=None, critique=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return original_valid
        bad = Report(
            title="Bad revision",
            markdown="# Bad\nClaim [99].\n\n## References\n",
            references=[],
        )
        # Deterministic validation would reject [99]; simulate that outcome.
        from agents.citations import validate_report_citations, assign_citation_ids

        cmap = assign_citation_ids(evidence)
        if validate_report_citations(bad.markdown, set(cmap)):
            raise WriterError(
                "Writer produced invalid citation IDs not present in evidence: [99]"
            )
        return bad

    pipeline_mocks.writer_side_effect = writer_behavior
    _set_verdicts(pipeline_mocks, _revision_needed(0))

    result = run_research_pipeline("some query")

    assert result.final_status == "revision_failed"
    assert result.report.title == "Report r0"
    assert result.iteration_count == 0
    assert "[99]" not in result.report.markdown
    assert all(1 <= r.citation_id <= len(result.evidence.sources)
               for r in result.report.references)


def test_initial_writer_failure_propagates(pipeline_mocks):
    pipeline_mocks.writer_side_effect = WriterError("Writer execution failed")
    with pytest.raises(WriterError):
        run_research_pipeline("some query")


def test_empty_query_rejected(pipeline_mocks):
    for bad in ("", "   "):
        with pytest.raises(ValueError, match="query"):
            run_research_pipeline(bad)
    assert pipeline_mocks.planner_calls == 0


def test_result_is_serializable(pipeline_mocks):
    _set_verdicts(pipeline_mocks, _approved())
    result = run_research_pipeline("some query")
    payload = result.model_dump_json()
    assert '"final_status":"approved"' in payload.replace(" ", "")
