"""Tests for the Writer Agent. All LLM calls are mocked (no network)."""

import json

import pytest
from pydantic import ValidationError

from agents import citations
from agents import writer as writer_module
from agents.schemas import (
    EvidenceBundle,
    EvidenceSource,
    RetrievalSummary,
    Report,
    ReportDraft,
    ReportReference,
    ResearchPlan,
)
from agents.writer import (
    InsufficientEvidenceError,
    WriterError,
    report_draft_json_schema,
    write_report,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _source(n: int, source_type="web", **overrides):
    base = dict(
        source_type=source_type,
        title=f"Source {n}",
        url=f"https://example.com/{n}" if source_type == "web" else None,
        document_name=f"doc{n}.pdf" if source_type == "kb" else None,
        content=f"Evidence text {n}. " * 5,
        score=0.9 - n * 0.1,
        source_id=str(n) if source_type == "kb" else f"https://example.com/{n}",
    )
    return EvidenceSource(**{**base, **overrides})


def _bundle(*sources):
    return EvidenceBundle(
        original_query="goal",
        sources=list(sources),
        retrieval_summary=RetrievalSummary(
            total_sources=len(sources),
            web_sources=sum(1 for s in sources if s.source_type == "web"),
            kb_sources=sum(1 for s in sources if s.source_type == "kb"),
            deduplicated_sources=0,
            kb_status="ok" if any(s.source_type == "kb" for s in sources) else "empty",
            web_status="ok",
        ),
    )


def _plan():
    return ResearchPlan(
        original_query="goal",
        sub_questions=["q1?", "q2?", "q3?"],
    )


@pytest.fixture
def five_source_bundle():
    return _bundle(*[_source(i) for i in range(1, 6)])


class _LLMMock:
    """Scripted mock for the Writer's direct structured-output call."""

    def __init__(self, monkeypatch):
        self.script = []
        self.calls = []  # captured system/user prompts per call

        def fake_request(system_prompt, user_prompt, cfg):
            self.calls.append({"system": system_prompt, "user": user_prompt})
            outcome = self.script.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            if isinstance(outcome, dict):
                return json.dumps(outcome)
            return outcome

        monkeypatch.setattr(
            writer_module, "_request_structured_draft", fake_request
        )

    def respond(self, markdown, title="Test Report"):
        self.script.append({"title": title, "markdown": markdown})

    def fail(self, exc):
        self.script.append(exc)


@pytest.fixture
def mock_llm(monkeypatch):
    """Patch the direct Groq helper; returns a scripted mock handle."""
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-tests")
    from config import get_config

    get_config.cache_clear()
    yield _LLMMock(monkeypatch)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    get_config.cache_clear()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

def test_valid_report_reference_web():
    ref = ReportReference(
        citation_id=1, source_type="web", title="T", url="https://example.com"
    )
    assert ref.document_name is None


def test_valid_report_reference_kb():
    ref = ReportReference(
        citation_id=2, source_type="kb", title="T", document_name="doc.pdf"
    )
    assert ref.url is None


def test_invalid_report_reference_rejected():
    with pytest.raises(ValidationError):
        ReportReference(citation_id=0, source_type="blog", title="T")


def test_valid_report_roundtrip():
    report = Report(
        title="Title",
        markdown="# Title\nBody [1].",
        references=[
            ReportReference(citation_id=1, source_type="web", title="S1")
        ],
    )
    assert report.references[0].citation_id == 1


# ---------------------------------------------------------------------------
# Citation extraction and validation
# ---------------------------------------------------------------------------

def test_extract_citations_basic():
    assert citations.extract_citations("A [1] and [3][12].") == {1, 3, 12}


def test_extract_citations_ignores_markdown_links():
    assert citations.extract_citations("[1](https://x.com)") == set()


def test_validate_accepts_known_ids(five_source_bundle):
    cmap = citations.assign_citation_ids(five_source_bundle)
    assert citations.validate_report_citations("Claim [1][5].", set(cmap)) == []


def test_validate_rejects_unknown_ids(five_source_bundle):
    cmap = citations.assign_citation_ids(five_source_bundle)
    assert citations.validate_report_citations("Bad [99].", set(cmap)) == [99]


def test_assign_citation_ids_deterministic_order(five_source_bundle):
    cmap = citations.assign_citation_ids(five_source_bundle)
    assert sorted(cmap) == [1, 2, 3, 4, 5]
    assert cmap[1].title == "Source 1"
    assert cmap[5].title == "Source 5"


# ---------------------------------------------------------------------------
# Reference construction (LLM number -> actual source -> authoritative ref)
# ---------------------------------------------------------------------------

def test_build_references_maps_to_real_sources(five_source_bundle):
    cmap = citations.assign_citation_ids(five_source_bundle)
    refs = citations.build_references(
        "Uses [1] and [3] only.", cmap
    )
    assert [r.citation_id for r in refs] == [1, 3]
    assert refs[0].url == "https://example.com/1"
    assert refs[1].url == "https://example.com/3"


def test_build_references_preserves_kb_metadata():
    bundle = _bundle(_source(1), _source(2, source_type="kb"))
    cmap = citations.assign_citation_ids(bundle)
    refs = citations.build_references("Fact [2].", cmap)
    assert refs[0].document_name == "doc2.pdf"
    assert refs[0].url is None
    assert refs[0].source_id == "2"


def test_unused_references_are_omitted(five_source_bundle):
    cmap = citations.assign_citation_ids(five_source_bundle)
    refs = citations.build_references("Only [4].", cmap)
    assert len(refs) == 1 and refs[0].citation_id == 4


# ---------------------------------------------------------------------------
# Provisional coverage heuristic
# ---------------------------------------------------------------------------

def test_citation_coverage_full_when_all_claims_cited():
    md = (
        "# Report\nAI adoption grew quickly according to multiple studies [1]. "
        "Hospitals reported measurable efficiency gains across departments [2].\n"
        "## References\n[1] A — https://a\n[2] B — https://b"
    )
    assert citations.citation_coverage(md) == 1.0


def test_citation_coverage_partial():
    cited = "AI adoption grew quickly according to multiple studies [1]. "
    uncited = "Many hospitals use electronic records across the country widely. "
    assert citations.citation_coverage(cited + uncited) == 0.5


def test_citation_coverage_no_claims_returns_one():
    assert citations.citation_coverage("# Title only") == 1.0


# ---------------------------------------------------------------------------
# Strict JSON-schema generation (native Structured Outputs compatibility)
# ---------------------------------------------------------------------------

def test_report_draft_schema_is_strict_compatible():
    schema = report_draft_json_schema()
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"title", "markdown"}
    assert set(schema["properties"]) == {"title", "markdown"}


def test_report_draft_schema_property_types():
    props = report_draft_json_schema()["properties"]
    assert props["title"]["type"] == "string"
    assert props["markdown"]["type"] == "string"


# ---------------------------------------------------------------------------
# Draft parsing
# ---------------------------------------------------------------------------

def test_parse_draft_valid_json():
    draft = writer_module._parse_draft(
        json.dumps({"title": "T", "markdown": "# T\nBody [1]."})
    )
    assert isinstance(draft, ReportDraft)
    assert draft.title == "T"


def test_parse_draft_malformed_json_raises_writer_error():
    with pytest.raises(WriterError, match="not valid JSON"):
        writer_module._parse_draft("{not json")


def test_parse_draft_non_object_raises_writer_error():
    with pytest.raises(WriterError, match="not a JSON object"):
        writer_module._parse_draft('["a"]')


def test_parse_draft_missing_fields_rejected():
    with pytest.raises(WriterError, match="validation"):
        writer_module._parse_draft(json.dumps({"title": "T"}))


# ---------------------------------------------------------------------------
# write_report service (mocked native structured output)
# ---------------------------------------------------------------------------

GOOD_MD = (
    "# AI in Healthcare\n\n## Executive Summary\nEvidence shows growth [1][2].\n\n"
    "## Findings\nOutcomes improved in trials [3].\n\n"
    "## References\n[1] Source 1\n[2] Source 2\n[3] Source 3"
)


def test_write_report_success(mock_llm, five_source_bundle):
    mock_llm.respond(GOOD_MD, title="AI in Healthcare")
    report = write_report(_plan(), five_source_bundle)
    assert isinstance(report, Report)
    assert report.title == "AI in Healthcare"
    assert "[1]" in report.markdown
    assert [r.citation_id for r in report.references] == [1, 2, 3]
    assert report.references[0].url == "https://example.com/1"


def test_write_report_rejects_hallucinated_citation(mock_llm, five_source_bundle):
    bad = GOOD_MD + "\nAlso [99]."
    mock_llm.respond(bad, title="T")
    with pytest.raises(WriterError, match="invalid citation"):
        write_report(_plan(), five_source_bundle)


def test_write_report_empty_evidence_fails_safely():
    with pytest.raises(InsufficientEvidenceError, match="Insufficient evidence"):
        write_report(_plan(), _bundle())


def test_write_report_empty_plan_rejected(five_source_bundle):
    plan = ResearchPlan(original_query="goal", sub_questions=["a?", "b?", "c?"])
    empty = plan.model_copy(update={"sub_questions": []})
    with pytest.raises(ValueError, match="sub-question"):
        write_report(empty, five_source_bundle)


def test_write_report_malformed_output_rejected(mock_llm, five_source_bundle):
    # Structured output guarantees JSON; an empty body still fails safely.
    mock_llm.respond("   ", title="T")
    with pytest.raises(WriterError, match="malformed"):
        write_report(_plan(), five_source_bundle)


def test_write_report_wraps_transport_failure_and_retries_once(
    mock_llm, five_source_bundle
):
    mock_llm.fail(WriterError("LLM request failed: ConnectError"))
    mock_llm.respond(GOOD_MD, title="T")
    report = write_report(_plan(), five_source_bundle)
    assert report.title == "T"
    assert len(mock_llm.calls) == 2


def test_write_report_all_attempts_exhausted_raises(mock_llm, five_source_bundle):
    for _ in range(writer_module.MAX_LLM_ATTEMPTS):
        mock_llm.fail(WriterError("LLM request returned HTTP 503"))
    with pytest.raises(WriterError):
        write_report(_plan(), five_source_bundle)
    assert len(mock_llm.calls) == writer_module.MAX_LLM_ATTEMPTS


def test_writer_prompt_contains_only_numbered_sources(
    mock_llm, five_source_bundle
):
    mock_llm.respond(GOOD_MD, title="T")
    write_report(_plan(), five_source_bundle)
    prompt = mock_llm.calls[0]["user"]
    assert "SOURCE [1]" in prompt
    assert "SOURCE [5]" in prompt
    assert "https://example.com/5" in prompt
    assert "[99]" not in prompt


def test_writer_prompt_states_explicit_citation_range(
    mock_llm, five_source_bundle
):
    mock_llm.respond(GOOD_MD, title="T")
    write_report(_plan(), five_source_bundle)
    prompt = mock_llm.calls[0]["user"]
    assert "AVAILABLE CITATIONS: [1] through [5]" in prompt
    assert "MUST NOT emit any citation number outside this range" in prompt


def test_writer_revision_prompt_contains_previous_report_and_critique(
    mock_llm, five_source_bundle
):
    from agents.schemas import CritiqueResult

    previous = Report(
        title="Old report",
        markdown="# Old report\nClaim [1].\n\n## References\n[1] Source 1",
        references=[],
    )
    critique = CritiqueResult(
        status="revision_needed",
        iteration=0,
        unsupported_claims=["Claim X is unsupported by provided evidence."],
        revision_instructions=["Remove claim X."],
        overall_assessment="Needs fixes.",
    )
    mock_llm.respond(GOOD_MD, title="Revised")
    report = write_report(
        _plan(),
        five_source_bundle,
        previous_report=previous,
        critique=critique,
    )
    assert report.title == "Revised"
    prompt = mock_llm.calls[0]["user"]
    assert "You are REVISING your previous report" in prompt
    assert "# Old report" in prompt
    assert "- Remove claim X." in prompt
    # Revision still receives the numbered evidence bundle.
    assert "SOURCE [1]" in prompt
    assert "AVAILABLE CITATIONS: [1] through [5]" in prompt


def test_writer_revision_with_valid_citations_accepted(
    mock_llm, five_source_bundle
):
    from agents.schemas import CritiqueResult

    previous = Report(title="Old", markdown="# Old\nClaim [1].", references=[])
    critique = CritiqueResult(
        status="revision_needed",
        iteration=0,
        revision_instructions=["Keep only supported claims."],
    )
    revised_md = (
        "# Revised\nSupported claim [2][4].\n\n## References\n"
        "[2] Source 2\n[4] Source 4"
    )
    mock_llm.respond(revised_md, title="Revised")
    report = write_report(
        _plan(), five_source_bundle,
        previous_report=previous, critique=critique,
    )
    assert [r.citation_id for r in report.references] == [2, 4]


def test_writer_revision_with_invalid_citation_rejected(
    mock_llm, five_source_bundle
):
    from agents.schemas import CritiqueResult

    previous = Report(title="Old", markdown="# Old\nClaim [1].", references=[])
    critique = CritiqueResult(
        status="revision_needed", iteration=0,
        revision_instructions=["Fix something."],
    )
    bad_md = (
        "# Revised\nClaim [99] beyond the bundle.\n\n## References\n"
    )
    mock_llm.respond(bad_md, title="Revised")
    with pytest.raises(WriterError, match="invalid citation"):
        write_report(
            _plan(), five_source_bundle,
            previous_report=previous, critique=critique,
        )


def test_same_citation_mapping_initial_and_revision(
    mock_llm, five_source_bundle
):
    """Initial and revision calls must see identical numbered sources."""
    from agents.schemas import CritiqueResult

    mock_llm.respond(GOOD_MD, title="Initial")
    initial = write_report(_plan(), five_source_bundle)
    critique = CritiqueResult(
        status="revision_needed", iteration=0,
        revision_instructions=["Adjust one claim."],
    )
    mock_llm.respond(GOOD_MD.replace("growth", "expansion"), title="Revised")
    revised = write_report(
        _plan(), five_source_bundle,
        previous_report=initial, critique=critique,
    )

    initial_prompt = mock_llm.calls[0]["user"]
    revision_prompt = mock_llm.calls[1]["user"]

    def _sources_block(prompt: str) -> str:
        start = prompt.index("Numbered sources")
        end = prompt.index("AVAILABLE CITATIONS")
        return prompt[start:end]

    assert _sources_block(initial_prompt) == _sources_block(revision_prompt)
    assert initial.references[-1].url == revised.references[-1].url


def test_writer_prompt_stays_within_tier_budget_with_large_inputs(
    mock_llm,
):
    """17 long sources + a long previous report must still fit one request."""
    from agents.schemas import CritiqueResult

    big_bundle = _bundle(
        *[_source(i, content="Long evidence passage. " * 120) for i in range(1, 18)]
    )
    long_report_md = "# Long\n" + ("Detailed claim [1]. " * 500)
    previous = Report(title="Old", markdown=long_report_md, references=[])
    critique = CritiqueResult(
        status="revision_needed", iteration=0,
        revision_instructions=["Trim unsupported claims."],
    )
    mock_llm.respond(GOOD_MD, title="Revised")
    write_report(
        _plan(), big_bundle, previous_report=previous, critique=critique,
    )
    prompt = mock_llm.calls[0]["user"]
    assert len(prompt) <= writer_module.source_budget_chars(0) + 200
