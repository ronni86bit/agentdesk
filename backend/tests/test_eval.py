"""Tests for the deterministic evaluation module (eval/scorer.py)."""

import pytest

from agents.schemas import (
    CritiqueResult,
    EvidenceBundle,
    EvidenceSource,
    EvaluationResult,
    Report,
    ReportReference,
    RetrievalSummary,
    ResearchPlan,
    ResearchRunResult,
)
from datetime import datetime, timezone
from eval.scorer import (
    evaluate_run,
    grounding_score,
    plan_coverage,
    valid_citation_rate,
)


def _source(n, source_type="web", **overrides):
    base = dict(
        source_type=source_type,
        title=f"Source {n}",
        url=f"https://example.com/{n}" if source_type == "web" else None,
        document_name=f"doc{n}.txt" if source_type == "kb" else None,
        content=f"Evidence {n}",
        score=0.9,
        source_id=f"src{n}",
    )
    return EvidenceSource(**{**base, **overrides})


def _bundle(sources):
    web = sum(1 for s in sources if s.source_type == "web")
    kb = sum(1 for s in sources if s.source_type == "kb")
    total = len(sources)
    return EvidenceBundle(
        original_query="goal",
        sources=sources,
        retrieval_summary=RetrievalSummary(
            total_sources=total, web_sources=web, kb_sources=kb,
            deduplicated_sources=0,
            kb_status="ok" if kb else "empty",
            web_status="ok" if web else "ok",
        ),
    )


def _plan():
    return ResearchPlan(
        original_query="AI healthcare India",
        sub_questions=[
            "current adoption of AI in Indian hospitals",
            "measurable patient outcomes",
            "regulatory challenges",
        ],
    )


def _result(markdown, sources=None, critique=None, final_status="approved",
            iteration_count=0, query="q"):
    return ResearchRunResult(
        query=query,
        plan=_plan(),
        evidence=_bundle(sources if sources is not None else [_source(1)]),
        report=Report(
            title="Report",
            markdown=markdown,
            references=[
                ReportReference(citation_id=n, source_type="web", title="t")
                for n in sorted({int(tok) for tok in __import__("re").findall(r"\[(\d+)\]", markdown)})
            ],
        ),
        critique=critique,
        iteration_count=iteration_count,
        final_status=final_status,
    )


STAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_full_coverage_every_claim_cited():
    md = (
        "# Report\n"
        "AI adoption in hospitals increased substantially last year [1].\n"
        "Patient outcomes improved measurably across studies [1].\n"
    )
    result = _result(md)
    evaluation = evaluate_run(result, now=STAMP)
    assert evaluation.citation_coverage == 1.0


def test_partial_coverage():
    md = (
        "# Report\n"
        "Cited claim with enough words to qualify [1].\n"
        "Uncited claim with enough words to qualify as a claim.\n"
    )
    result = _result(md)
    evaluation = evaluate_run(result, now=STAMP)
    assert 0.0 < evaluation.citation_coverage < 1.0
    assert evaluation.citation_coverage == pytest.approx(0.5)


def test_zero_citations_zero_coverage():
    md = (
        "# Report\n"
        "A claim sentence with more than six words total.\n"
        "Another claim sentence that also exceeds six words.\n"
    )
    result = _result(md)
    evaluation = evaluate_run(result, now=STAMP)
    assert evaluation.citation_coverage == 0.0


def test_no_claim_sentences_defaults_to_full_coverage():
    md = "# Report\n## Section\n- short\n"
    result = _result(md)
    evaluation = evaluate_run(result, now=STAMP)
    assert evaluation.citation_coverage == 1.0


def test_invalid_citations_lower_valid_rate():
    md = "Claimed fact with enough words here [1].\nOther claim [7].\n"
    rate = valid_citation_rate(md, {1})
    assert rate == pytest.approx(0.5)


def test_valid_rate_is_one_for_clean_report():
    md = "Claimed fact with enough words here [1]."
    assert valid_citation_rate(md, {1}) == 1.0
    assert valid_citation_rate("no citations at all", {1}) == 1.0


def test_retrieval_balance_percentages():
    sources = [_source(1), _source(2, "web"), _source(3, "kb")]
    result = _result("Claim [1].", sources=sources)
    evaluation = evaluate_run(result, now=STAMP)
    assert evaluation.total_sources == 3
    assert evaluation.web_percentage == pytest.approx(66.67)
    assert evaluation.kb_percentage == pytest.approx(33.33)


def test_web_only_and_kb_only_balances():
    web_only = _result("Claim [1].", sources=[_source(1)])
    evaluation = evaluate_run(web_only, now=STAMP)
    assert evaluation.web_percentage == 100.0
    assert evaluation.kb_percentage == 0.0

    kb_only = _result("Claim [1].", sources=[_source(1, "kb")])
    evaluation = evaluate_run(kb_only, now=STAMP)
    assert evaluation.kb_percentage == 100.0


def test_empty_evidence_zero_percentages():
    result = _result("Claim [1].", sources=[])
    evaluation = evaluate_run(result, now=STAMP)
    assert evaluation.total_sources == 0
    assert evaluation.web_percentage == 0.0
    assert evaluation.kb_percentage == 0.0


def test_critic_component_mapping():
    assert grounding_score(1, 1, 1, "approved") == pytest.approx(1.0)
    # max_revisions_reached: 0.40 + 0.25 + 0.20 + 0.15*0.5
    assert grounding_score(1, 1, 1, "max_revisions_reached") == pytest.approx(0.925)
    assert grounding_score(1, 1, 1, "revision_failed") == pytest.approx(0.85)
    assert grounding_score(0, 0, 0, "revision_failed") == 0.0


def test_grounding_score_formula_documented_weights():
    # 0.40*coverage + 0.25*valid + 0.20*plan + 0.15*critic
    score = grounding_score(0.5, 1.0, 1.0, "approved")
    assert score == pytest.approx(0.4 * 0.5 + 0.25 + 0.20 + 0.15)


def test_plan_coverage_keyword_heuristic():
    questions = [
        "current adoption of AI in Indian hospitals",
        "quantum blockchain synergies",
    ]
    body = (
        "Indian hospitals are adopting AI at a rapid pace. "
        "Adoption of AI systems in hospitals is measurable."
    )
    score = plan_coverage(questions, body)
    assert 0.0 < score < 1.0  # first covered, second absent


def test_plan_coverage_full_when_all_addressed():
    questions = ["patient outcomes", "regulation"]
    body = "Patient outcomes improved. Regulation is evolving quickly."
    assert plan_coverage(questions, body) == 1.0


def test_plan_coverage_empty_plan():
    assert plan_coverage([], "anything") == 1.0


def test_evaluate_run_fields_and_serialization():
    md = (
        "# Report\n"
        "AI in Indian hospitals adoption is rising quickly [1].\n"
        "Regulatory challenges remain significant this year [1].\n"
    )
    critique = CritiqueResult(status="approved", iteration=0)
    result = _result(md, sources=[_source(1), _source(2, "kb")],
                     critique=critique, iteration_count=0)
    evaluation = evaluate_run(result, now=STAMP)

    assert isinstance(evaluation, EvaluationResult)
    assert evaluation.timestamp == STAMP.isoformat()
    assert evaluation.query == result.query
    assert evaluation.critic_status == "approved"
    assert evaluation.iteration_count == 0
    assert 0.0 <= evaluation.grounding_score <= 1.0
    payload = evaluation.model_dump_json()
    assert "grounding_score" in payload
    assert "citation_coverage" in payload


def test_evaluation_none_by_default_on_result():
    result = _result("Claim [1].")
    assert result.evaluation is None
    assert result.timing is None
