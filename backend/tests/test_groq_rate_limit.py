"""Rate-limit behavior: TPM vs TPD distinction, reset-aware backoff, caps.

All offline: httpx and the LLM boundary are monkeypatched; no network access
and no real API key are involved. Companion tests for the benchmark-level
TPM/TPD semantics live in test_hallucination_benchmark.py.
"""

import types

import httpx
import pytest
from pydantic import BaseModel

from agents import groq_structured as gs
from agents import planner as planner_module
from agents import writer as writer_module
from agents.schemas import (
    EvidenceBundle,
    EvidenceSource,
    ResearchPlan,
    RetrievalSummary,
)
from config import get_config

# Realistic Groq 429 bodies (the exceeded bucket is named in the message).
TPM_BODY = (
    '{"error":{"message":"Rate limit reached for model `openai/gpt-oss-120b` '
    'in organization `org_test` service tier `on_demand` on tokens per minute '
    '(TPM). Limit: 8000, Used: 8000, Requested: 900."}}'
)
TPD_BODY = (
    '{"error":{"message":"Rate limit reached for model `openai/gpt-oss-120b` '
    'in organization `org_test` service tier `on_demand` on tokens per day '
    '(TPD). Limit: 100000, Used: 100000."}}'
)
OTHER_4XX_BODY = '{"error":{"message":"Invalid schema provided."}}'


class _Out(BaseModel):
    answer: str


def _cfg() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        groq_model="openai/gpt-oss-120b", groq_api_key="test-key",
        has_groq_key=True,
    )


def _patch_post(monkeypatch, status_code: int, body: str, headers: dict):
    def fake_post(url, **kwargs):
        return httpx.Response(
            status_code=status_code,
            content=body.encode("utf-8"),
            headers=headers,
            request=httpx.Request("POST", gs.GROQ_CHAT_COMPLETIONS_URL),
        )

    monkeypatch.setattr(gs.httpx, "post", fake_post)


def _request(monkeypatch, status_code: int, body: str,
             headers: dict | None = None):
    _patch_post(monkeypatch, status_code, body, headers or {})
    with pytest.raises(gs.StructuredOutputError) as excinfo:
        gs.request_structured_completion(
            "system", "user", _cfg(), "out", gs.strict_schema(_Out)
        )
    return excinfo.value


# ---------------------------------------------------------------------------
# Client: 429 scope classification + reset-aware backoff
# ---------------------------------------------------------------------------

def test_classify_rate_limit_unit():
    assert gs.classify_rate_limit(TPM_BODY) == "tpm"
    assert gs.classify_rate_limit(TPD_BODY) == "tpd"
    assert gs.classify_rate_limit(OTHER_4XX_BODY) is None
    assert gs.classify_rate_limit("") is None
    assert gs.classify_rate_limit(None) is None


def test_tpm_429_classified_and_uses_token_reset(monkeypatch):
    err = _request(monkeypatch, 429, TPM_BODY,
                   {"x-ratelimit-reset-tokens": "2.077s"})
    assert err.rate_limit_scope == "tpm"
    assert err.retry_after == pytest.approx(2.077)
    assert gs.backoff_delay(err) == pytest.approx(2.077)


def test_tpd_429_classified_and_backoff_capped_at_60(monkeypatch):
    err = _request(monkeypatch, 429, TPD_BODY, {"Retry-After": "3600"})
    assert err.rate_limit_scope == "tpd"
    assert gs.MAX_BACKOFF_SECONDS == 60.0
    assert gs.backoff_delay(err) == 60.0
    assert gs.backoff_delay(err) <= 60.0


def test_tpm_429_without_reset_headers_falls_back_to_short_default(monkeypatch):
    err = _request(monkeypatch, 429, TPM_BODY, {})
    assert err.rate_limit_scope == "tpm"
    assert err.retry_after is None
    assert gs.backoff_delay(err) == gs.DEFAULT_BACKOFF_SECONDS == 5.0


def test_non_429_error_carries_no_rate_limit_scope(monkeypatch):
    err = _request(monkeypatch, 400, OTHER_4XX_BODY)
    assert err.rate_limit_scope is None
    assert err.retry_after is None


def test_parse_reset_seconds_supported_formats():
    assert gs.parse_reset_seconds("2.077s") == pytest.approx(2.077)
    assert gs.parse_reset_seconds("1m30s") == pytest.approx(90.0)
    assert gs.parse_reset_seconds("45s") == pytest.approx(45.0)
    # Hour-scale values are not part of the documented header format.
    assert gs.parse_reset_seconds("1h") is None
    assert gs.parse_reset_seconds(None) is None
    assert gs.parse_reset_seconds("") is None


# ---------------------------------------------------------------------------
# Agent retry loops: TPD fails fast, TPM sleeps bounded then fails
# ---------------------------------------------------------------------------

def _fake_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-tests")
    get_config.cache_clear()


def _clear_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    get_config.cache_clear()


def test_planner_tpd_error_fails_fast_without_sleep_or_retry(monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    def fake_request(system_prompt, user_prompt, cfg):
        calls["n"] += 1
        raise planner_module.PlannerError(
            "LLM request returned HTTP 429: ... on tokens per day (TPD)",
            retry_after=3600.0,
            rate_limit_scope="tpd",
        )

    monkeypatch.setattr(planner_module, "_request_structured_plan", fake_request)
    _fake_key(monkeypatch)
    try:
        with pytest.raises(planner_module.PlannerError):
            planner_module.create_research_plan("some research goal")
    finally:
        _clear_key(monkeypatch)
    assert calls["n"] == 1  # daily cap: no second attempt
    assert sleeps == []     # ... and no backoff sleep


def test_writer_tpm_error_sleeps_bounded_then_fails(monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    calls = {"n": 0}

    def fake_request(system_prompt, user_prompt, cfg):
        calls["n"] += 1
        raise writer_module.WriterError(
            "LLM request returned HTTP 429: ... on tokens per minute (TPM)",
            retry_after=20.0,
            rate_limit_scope="tpm",
        )

    monkeypatch.setattr(writer_module, "_request_structured_draft", fake_request)
    plan = ResearchPlan(original_query="q?", sub_questions=["a?", "b?", "c?"])
    evidence = EvidenceBundle(
        original_query="q?",
        sources=[EvidenceSource(source_type="web", title="T",
                                url="https://example.com/", content="c")],
        retrieval_summary=RetrievalSummary(
            total_sources=1, web_sources=1, kb_sources=0,
            deduplicated_sources=1, kb_status="skipped", web_status="ok"),
    )
    with pytest.raises(writer_module.WriterError):
        writer_module._generate_report_draft(plan, evidence, None, None, _cfg())
    assert calls["n"] == writer_module.MAX_LLM_ATTEMPTS == 2
    assert len(sleeps) == 1 and sleeps[0] == 20.0


def test_writer_unscoped_429_still_retries_with_capped_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    def fake_request(system_prompt, user_prompt, cfg):
        raise writer_module.WriterError(
            "LLM request returned HTTP 429: unknown scope",
            retry_after=75.0,  # would exceed the cap without it
        )

    monkeypatch.setattr(writer_module, "_request_structured_draft", fake_request)
    plan = ResearchPlan(original_query="q?", sub_questions=["a?", "b?", "c?"])
    evidence = EvidenceBundle(
        original_query="q?",
        sources=[EvidenceSource(source_type="web", title="T",
                                url="https://example.com/", content="c")],
        retrieval_summary=RetrievalSummary(
            total_sources=1, web_sources=1, kb_sources=0,
            deduplicated_sources=1, kb_status="skipped", web_status="ok"),
    )
    with pytest.raises(writer_module.WriterError):
        writer_module._generate_report_draft(plan, evidence, None, None, _cfg())
    assert sleeps == [min(75.0, writer_module.MAX_BACKOFF_SECONDS)]
    assert sleeps[0] <= 60.0
