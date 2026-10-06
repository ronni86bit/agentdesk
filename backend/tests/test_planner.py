"""Tests for the Planner Agent interface. No real API calls are made."""

import json

import pytest
from pydantic import ValidationError

from agents import planner as planner_module
from agents.planner import (
    MissingGroqKeyError,
    PlannerError,
    create_research_plan,
    research_plan_json_schema,
)
from agents.schemas import ResearchPlan
from config import DEFAULT_GROQ_MODEL, get_config, groq_litellm_model
from main import PlanRequest


def _patch_llm(monkeypatch, payloads):
    """Patch the direct structured-output helper with scripted outcomes."""
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

    monkeypatch.setattr(
        planner_module, "_request_structured_plan", fake_request
    )
    return calls


def _plan_payload(query="AI impact on healthcare in India"):
    return {
        "original_query": query,
        "sub_questions": [
            "Q1 about deployment?",
            "Q2 about outcomes?",
            "Q3 about regulation?",
        ],
    }


def test_default_model_is_gpt_oss_120b():
    assert DEFAULT_GROQ_MODEL == "openai/gpt-oss-120b"
    get_config.cache_clear()
    try:
        assert get_config().groq_model == "openai/gpt-oss-120b"
    finally:
        get_config.cache_clear()


def test_groq_routing_helper_prefixes_provider():
    assert groq_litellm_model("openai/gpt-oss-120b") == "groq/openai/gpt-oss-120b"
    assert groq_litellm_model("some-model") == "groq/some-model"
    with pytest.raises(ValueError):
        groq_litellm_model("")


def test_groq_routing_resolves_to_groq_provider_not_openai():
    from litellm import get_llm_provider

    model, provider, _, _ = get_llm_provider(
        groq_litellm_model(DEFAULT_GROQ_MODEL)
    )
    assert provider == "groq"
    assert model == "openai/gpt-oss-120b"


def test_empty_query_rejected_by_request_model():
    with pytest.raises(ValidationError):
        PlanRequest(query="")
    with pytest.raises(ValidationError):
        PlanRequest(query="   ")


def test_blank_query_rejected_by_planner_service():
    with pytest.raises(ValueError):
        create_research_plan("  ")


def test_research_plan_accepts_valid_output():
    plan = ResearchPlan(
        original_query="Research the impact of AI on healthcare in India",
        sub_questions=[
            "What AI applications are currently deployed in Indian healthcare?",
            "What measurable outcomes have been reported from AI in Indian healthcare?",
            "What regulatory and ethical challenges does AI face in Indian healthcare?",
            "Which organizations are active in AI-driven healthcare in India?",
        ],
    )
    assert len(plan.sub_questions) == 4


def test_research_plan_rejects_too_few_sub_questions():
    with pytest.raises(ValidationError):
        ResearchPlan(
            original_query="query",
            sub_questions=["only one question"],
        )


def test_research_plan_rejects_too_many_sub_questions():
    with pytest.raises(ValidationError):
        ResearchPlan(
            original_query="query",
            sub_questions=[f"question {i}" for i in range(6)],
        )


def test_research_plan_rejects_blank_entries():
    with pytest.raises(ValidationError):
        ResearchPlan(
            original_query="query",
            sub_questions=["valid question", "   ", "another valid question"],
        )


def test_missing_groq_key_raises_clean_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    get_config.cache_clear()
    try:
        with pytest.raises(MissingGroqKeyError):
            create_research_plan("valid research goal")
    finally:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        get_config.cache_clear()


def test_planner_service_with_mocked_llm(monkeypatch):
    expected = ResearchPlan(**_plan_payload())
    calls = _patch_llm(monkeypatch, [_plan_payload()])
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-tests")
    get_config.cache_clear()
    try:
        plan = create_research_plan("AI impact on healthcare in India")
        assert plan == expected
        assert "AI impact on healthcare in India" in calls["user"]
        assert "Research Planning Specialist" in calls["system"]
    finally:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        get_config.cache_clear()


def test_planner_wraps_execution_failures(monkeypatch):
    _patch_llm(
        monkeypatch, [PlannerError("LLM request returned HTTP 503")] * 2
    )
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-tests")
    get_config.cache_clear()
    try:
        with pytest.raises(PlannerError):
            create_research_plan("some research goal")
    finally:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        get_config.cache_clear()


def test_planner_retries_once_on_transient_failure(monkeypatch):
    calls = _patch_llm(
        monkeypatch,
        [PlannerError("LLM request failed: ConnectError"), _plan_payload()],
    )
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-tests")
    get_config.cache_clear()
    try:
        plan = create_research_plan("some research goal")
        assert plan.original_query == "AI impact on healthcare in India"
        assert calls["count"] == 2
    finally:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        get_config.cache_clear()


def test_planner_no_unbounded_retry(monkeypatch):
    calls = _patch_llm(
        monkeypatch, [PlannerError("LLM request returned HTTP 503")] * 10
    )
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-tests")
    get_config.cache_clear()
    try:
        with pytest.raises(PlannerError):
            create_research_plan("some research goal")
        assert calls["count"] == planner_module.MAX_LLM_ATTEMPTS == 2
    finally:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        get_config.cache_clear()


def test_planner_malformed_json_rejected(monkeypatch):
    _patch_llm(monkeypatch, ["{definitely not json", "{definitely not json"])
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-tests")
    get_config.cache_clear()
    try:
        with pytest.raises(PlannerError, match="not valid JSON"):
            create_research_plan("some research goal")
    finally:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        get_config.cache_clear()


def test_planner_missing_fields_rejected(monkeypatch):
    payload = _plan_payload()
    del payload["sub_questions"]
    _patch_llm(monkeypatch, [payload, payload])
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-tests")
    get_config.cache_clear()
    try:
        with pytest.raises(PlannerError, match="validation"):
            create_research_plan("some research goal")
    finally:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        get_config.cache_clear()


# ---------------------------------------------------------------------------
# Strict JSON-schema generation (native Structured Outputs compatibility)
# ---------------------------------------------------------------------------

def test_research_plan_schema_is_strict_compatible():
    schema = research_plan_json_schema()
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"original_query", "sub_questions"}
    props = schema["properties"]
    assert props["original_query"]["type"] == "string"
    assert props["sub_questions"]["type"] == "array"
    assert props["sub_questions"]["items"]["type"] == "string"
    # Pydantic Field constraints carry over as strict-compatible bounds.
    assert props["sub_questions"].get("minItems") == 3
    assert props["sub_questions"].get("maxItems") == 5
