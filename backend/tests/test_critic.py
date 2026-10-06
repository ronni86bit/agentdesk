"""Tests for the Critic Agent. All LLM calls are mocked (no network)."""

import json

import pytest

from agents import critic as critic_module
from agents.critic import (
    MAX_LLM_ATTEMPTS,
    CriticError,
    MissingGroqKeyError,
    critique_json_schema,
    evaluate_report,
)
from agents.schemas import (
    CritiqueResult,
    EvidenceBundle,
    EvidenceSource,
    RetrievalSummary,
    Report,
    ResearchPlan,
)


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


def _bundle(*sources):
    return EvidenceBundle(
        original_query="goal",
        sources=list(sources),
        retrieval_summary=RetrievalSummary(
            total_sources=len(sources),
            web_sources=len(sources),
            kb_sources=0,
            deduplicated_sources=0,
            kb_status="empty",
            web_status="ok",
        ),
    )


def _plan():
    return ResearchPlan(original_query="goal", sub_questions=["q1?", "q2?", "q3?"])


def _report(markdown="# Report\nClaim [1].\n\n## References\n[1] Source 1"):
    return Report(title="Report", markdown=markdown, references=[])


def _approved_payload(iteration=0, assessment="Fully grounded."):
    return {
        "status": "approved",
        "iteration": iteration,
        "citation_issues": [],
        "unsupported_claims": [],
        "logical_issues": [],
        "missing_evidence": [],
        "revision_instructions": [],
        "overall_assessment": assessment,
    }


def _revision_needed_payload(instructions=None, iteration=0):
    return {
        "status": "revision_needed",
        "iteration": iteration,
        "citation_issues": [],
        "unsupported_claims": ["Claim X is unsupported by provided evidence."],
        "logical_issues": [],
        "missing_evidence": [],
        "revision_instructions": instructions
        or ["Remove claim X; no retrieved source supports it."],
        "overall_assessment": "Needs fixes.",
    }


@pytest.fixture
def fake_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-tests")
    from config import get_config

    get_config.cache_clear()
    yield
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    get_config.cache_clear()


def _patch_llm(monkeypatch, payloads):
    """Patch the direct Groq helper with a scripted sequence of JSON payloads.

    `payloads` items may be dicts (serialized to JSON), raw strings, or
    Exception instances to raise.
    """
    scripted = list(payloads)
    calls = {"system": None, "user": None, "count": 0}

    def fake_request(system_prompt, user_prompt, cfg):
        calls["system"] = system_prompt
        calls["user"] = user_prompt
        calls["count"] += 1
        outcome = scripted.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, dict):
            return json.dumps(outcome)
        return outcome

    monkeypatch.setattr(critic_module, "_request_structured_critique", fake_request)
    return calls


# ---------------------------------------------------------------------------
# Schema generation (strict Structured Outputs compatibility)
# ---------------------------------------------------------------------------

EXPECTED_PROPERTIES = {
    "status",
    "iteration",
    "citation_issues",
    "unsupported_claims",
    "logical_issues",
    "missing_evidence",
    "revision_instructions",
    "overall_assessment",
}


def test_schema_has_expected_properties():
    schema = critique_json_schema()
    assert set(schema["properties"]) == EXPECTED_PROPERTIES


def test_schema_is_strict_compatible():
    schema = critique_json_schema()
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == EXPECTED_PROPERTIES


def test_schema_status_is_explicit_enum():
    status = critique_json_schema()["properties"]["status"]
    assert status["enum"] == ["approved", "revision_needed"]
    assert status["type"] == "string"


def test_schema_array_fields_are_string_arrays():
    props = critique_json_schema()["properties"]
    for name in (
        "citation_issues",
        "unsupported_claims",
        "logical_issues",
        "missing_evidence",
        "revision_instructions",
    ):
        assert props[name]["type"] == "array"
        assert props[name]["items"]["type"] == "string"


def test_schema_iteration_is_non_negative_integer():
    prop = critique_json_schema()["properties"]["iteration"]
    assert prop["type"] == "integer"
    assert prop.get("minimum") == 0


# ---------------------------------------------------------------------------
# Rate-limit backoff parsing
# ---------------------------------------------------------------------------

def test_parse_reset_seconds_handles_seconds():
    assert critic_module._parse_reset_seconds("2.077s") == 2.077


def test_parse_reset_seconds_handles_minutes_and_seconds():
    assert critic_module._parse_reset_seconds("1m30s") == 90.0


def test_parse_reset_seconds_rejects_garbage_and_zero():
    assert critic_module._parse_reset_seconds(None) is None
    assert critic_module._parse_reset_seconds("") is None
    assert critic_module._parse_reset_seconds("soon") is None
    assert critic_module._parse_reset_seconds("0s") is None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_valid_json_becomes_critique_result():
    result = critic_module._parse_critique(json.dumps(_approved_payload()))
    assert isinstance(result, CritiqueResult)
    assert result.status == "approved"


def test_parse_malformed_json_raises_critic_error():
    with pytest.raises(CriticError, match="not valid JSON"):
        critic_module._parse_critique("this is not json at all")


def test_parse_non_object_json_raises_critic_error():
    with pytest.raises(CriticError, match="not a JSON object"):
        critic_module._parse_critique('["approved"]')


def test_parse_missing_required_fields_rejected():
    payload = _revision_needed_payload()
    del payload["status"]
    with pytest.raises(CriticError, match="validation"):
        critic_module._parse_critique(json.dumps(payload))


def test_parse_invalid_status_rejected():
    payload = _approved_payload()
    payload["status"] = "maybe"
    with pytest.raises(CriticError, match="validation"):
        critic_module._parse_critique(json.dumps(payload))


def test_parse_approved_with_instructions_rejected_by_model_validator():
    payload = _approved_payload()
    payload["revision_instructions"] = ["Fix something."]
    with pytest.raises(CriticError, match="validation"):
        critic_module._parse_critique(json.dumps(payload))


def test_parse_revision_needed_without_instructions_rejected():
    payload = _revision_needed_payload()
    payload["revision_instructions"] = []
    with pytest.raises(CriticError, match="validation"):
        critic_module._parse_critique(json.dumps(payload))


# ---------------------------------------------------------------------------
# evaluate_report behavior (mocked direct Groq call)
# ---------------------------------------------------------------------------

def test_evaluate_report_fully_supported_report_approved(fake_env, monkeypatch):
    _patch_llm(monkeypatch, [_approved_payload()])
    critique = evaluate_report(_plan(), _bundle(_source(1)), _report(), 0)
    assert isinstance(critique, CritiqueResult)
    assert critique.status == "approved"


def test_evaluate_report_unsupported_claim_needs_revision(fake_env, monkeypatch):
    payload = _revision_needed_payload(
        instructions=[
            "Remove the claim that AI reduced costs by 40%; no retrieved "
            "source contains this figure."
        ]
    )
    _patch_llm(monkeypatch, [payload])
    critique = evaluate_report(_plan(), _bundle(_source(1)), _report(), 0)
    assert critique.status == "revision_needed"
    assert any("40%" in i for i in critique.revision_instructions)


def test_evaluate_report_missing_citation_needs_revision(fake_env, monkeypatch):
    payload = _revision_needed_payload(instructions=["x"])
    payload["unsupported_claims"] = []
    payload["citation_issues"] = ["Regulatory statement has no supporting citation."]
    payload["revision_instructions"] = [
        "Add a citation to the regulatory statement using source [1]."
    ]
    _patch_llm(monkeypatch, [payload])
    critique = evaluate_report(_plan(), _bundle(_source(1)), _report(), 0)
    assert critique.status == "revision_needed"
    assert critique.citation_issues


def test_evaluate_report_major_plan_gap_needs_revision(fake_env, monkeypatch):
    payload = _revision_needed_payload(instructions=["y"])
    payload["unsupported_claims"] = []
    payload["missing_evidence"] = ["Sub-question q3 about regulation is not addressed."]
    payload["revision_instructions"] = [
        "Add a section addressing q3 using only the provided sources."
    ]
    _patch_llm(monkeypatch, [payload])
    critique = evaluate_report(_plan(), _bundle(_source(1)), _report(), 0)
    assert critique.status == "revision_needed"
    assert critique.missing_evidence


def test_evaluate_report_prompt_contains_plan_sources_and_report(
    fake_env, monkeypatch
):
    calls = _patch_llm(monkeypatch, [_approved_payload()])
    plan = _plan()
    bundle = _bundle(_source(1), _source(2))
    report = _report("Body text [1].")
    evaluate_report(plan, bundle, report, iteration=1)

    user_prompt = calls["user"]
    assert "Report iteration under review: 1" in user_prompt
    assert "- q1?" in user_prompt
    assert "SOURCE [1]" in user_prompt and "SOURCE [2]" in user_prompt
    assert "Body text [1]." in user_prompt
    assert "NEVER propose replacement facts" in user_prompt
    system_prompt = calls["system"]
    assert CRITIC_ROLE_SENTINEL in system_prompt


CRITIC_ROLE_SENTINEL = "Research Quality and Citation Auditor"


def test_evaluate_report_retries_once_on_transient_failure(fake_env, monkeypatch):
    calls = _patch_llm(
        monkeypatch,
        [CriticError("transient HTTP failure"), _approved_payload()],
    )
    critique = evaluate_report(_plan(), _bundle(_source(1)), _report(), 0)
    assert critique.status == "approved"
    assert calls["count"] == 2


def test_evaluate_report_all_attempts_exhausted_raises(fake_env, monkeypatch):
    calls = _patch_llm(
        monkeypatch,
        [CriticError("fail")] * MAX_LLM_ATTEMPTS,
    )
    with pytest.raises(CriticError):
        evaluate_report(_plan(), _bundle(_source(1)), _report(), 0)
    assert calls["count"] == MAX_LLM_ATTEMPTS


def test_evaluate_report_no_unbounded_retry(fake_env, monkeypatch):
    # More failures than attempts: must stop at exactly MAX_LLM_ATTEMPTS.
    calls = _patch_llm(monkeypatch, [CriticError("fail")] * 10)
    with pytest.raises(CriticError):
        evaluate_report(_plan(), _bundle(_source(1)), _report(), 0)
    assert calls["count"] == MAX_LLM_ATTEMPTS == 2


def test_evaluate_report_malformed_output_then_valid_succeeds(fake_env, monkeypatch):
    _patch_llm(
        monkeypatch,
        ["{{not json", json.dumps(_approved_payload())],
    )
    critique = evaluate_report(_plan(), _bundle(_source(1)), _report(), 0)
    assert critique.status == "approved"


def test_evaluate_report_negative_iteration_rejected():
    with pytest.raises(ValueError, match="iteration"):
        evaluate_report(_plan(), _bundle(_source(1)), _report(), -1)


def test_missing_groq_key_raises_clean_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    from config import get_config

    get_config.cache_clear()
    try:
        with pytest.raises(MissingGroqKeyError):
            evaluate_report(_plan(), _bundle(_source(1)), _report(), 0)
    finally:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        get_config.cache_clear()


# ---------------------------------------------------------------------------
# Writer revision interface (contract)
# ---------------------------------------------------------------------------

def test_writer_requires_previous_report_and_critique_together():
    from agents.writer import write_report

    bundle = _bundle(_source(1))
    approved = CritiqueResult(
        status="approved", iteration=0, overall_assessment="OK."
    )
    with pytest.raises(ValueError, match="together"):
        write_report(_plan(), bundle, critique=approved)
    with pytest.raises(ValueError, match="together"):
        write_report(_plan(), bundle, previous_report=_report())


def test_evaluate_report_prompt_stays_within_tier_budget(fake_env, monkeypatch):
    """Large bundles and reports must produce a single request-sized prompt."""
    calls = _patch_llm(monkeypatch, [_approved_payload()])
    big_bundle = _bundle(
        *[_source(i, content="Long evidence passage. " * 120) for i in range(1, 18)]
    )
    big_report = _report("# Report\n" + ("Detailed claim [1]. " * 500))
    evaluate_report(_plan(), big_bundle, big_report, 0)
    from agents.groq_structured import TARGET_PROMPT_CHARS

    assert len(calls["user"]) <= TARGET_PROMPT_CHARS + 200
