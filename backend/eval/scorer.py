"""Deterministic evaluation of finished research runs.

`evaluate_run` computes an `EvaluationResult` from the final
`ResearchRunResult` with NO additional LLM calls. Every metric is an
explicitly-labeled heuristic — it measures mechanical grounding signals
(citations, coverage, retrieval balance), not factual truth.

Metric definitions
------------------
citation_coverage
    Fraction of "claim sentences" (heuristic: 6+ words, not headings,
    references entries, or transition openers) that carry at least one
    inline [N] citation. 1.0 when no claim sentences are found.

valid_citation_rate
    Valid inline citation references / total inline citation references.
    A reference is valid when its [N] maps to a real evidence source.
    Expected to be 1.0 for any accepted final report (invalid citations
    are rejected before this point); values below 1.0 indicate a
    non-writer failure path.

plan_coverage
    Fraction of planned sub-questions "addressed" by the report, where a
    sub-question counts as addressed when at least half of its salient
    keywords (lowercased, stopwords removed) appear in the report body.
    Limitation: this is lexical keyword overlap, NOT semantic coverage —
    a sub-question answered with pure synonyms scores as uncovered.

web_percentage / kb_percentage
    Retrieved-source balance: share of web vs. KB sources in the final
    evidence bundle (0/0 when no sources exist).

critic_component (internal, feeds grounding_score)
    final_status mapping: approved -> 1.0, max_revisions_reached -> 0.5,
    revision_failed -> 0.0.

grounding_score (composite heuristic, documented formula)
    0.40 * citation_coverage
    + 0.25 * valid_citation_rate
    + 0.20 * plan_coverage
    + 0.15 * critic_component
    A transparency score, not "accuracy": it rewards mechanically
    well-grounded reports, not verified truth.
"""

import re
from datetime import datetime, timezone

from agents import citations
from agents.schemas import EvaluationResult, ResearchRunResult

# Grounding score weights (documented, intentionally simple).
W_COVERAGE = 0.40
W_VALID_RATE = 0.25
W_PLAN = 0.20
W_CRITIC = 0.15

_CRITIC_COMPONENT = {
    "approved": 1.0,
    "max_revisions_reached": 0.5,
    "revision_failed": 0.0,
}

# Small stopword list for the lexical plan-coverage heuristic.
_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "how", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "there", "these", "this", "to", "was", "were", "what", "when", "where", "which", "who", "why", "will", "with"]
)

_MIN_KEYWORD_KEEP = 3  # keywords shorter than this carry little signal


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _report_body(markdown: str) -> str:
    """Report text with the References section removed from coverage math."""
    marker = markdown.lower().rfind("## references")
    if marker == -1:
        return markdown
    return markdown[:marker]


def valid_citation_rate(markdown: str, valid_ids: set[int]) -> float:
    """Valid inline citation references / total inline citation references."""
    found = citations.extract_citations(markdown)
    if not found:
        return 1.0
    return len(found & valid_ids) / len(found)


def _keywords(text: str) -> list[str]:
    return [
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) >= _MIN_KEYWORD_KEEP and w not in _STOPWORDS
    ]


def plan_coverage(plan_sub_questions: list[str], report_markdown: str) -> float:
    """Fraction of sub-questions whose keywords mostly appear in the report.

    Heuristic (lexical): a sub-question is "addressed" when at least half of
    its salient keywords occur anywhere in the report body. Documented
    limitation: synonym-only answers count as uncovered.
    """
    if not plan_sub_questions:
        return 1.0
    body = _report_body(report_markdown).lower()
    addressed = 0
    for question in plan_sub_questions:
        keywords = _keywords(question)
        if not keywords:
            # Nothing measurable in this sub-question; treat as addressed.
            addressed += 1
            continue
        hits = sum(1 for kw in keywords if kw in body)
        if hits / len(keywords) >= 0.5:
            addressed += 1
    return addressed / len(plan_sub_questions)


def critic_component(final_status: str) -> float:
    """Transparent quality-gate component of the grounding score."""
    return _CRITIC_COMPONENT.get(final_status, 0.0)


def grounding_score(
    coverage: float,
    valid_rate: float,
    plan_cov: float,
    final_status: str,
) -> float:
    """Composite documented heuristic (see module docstring)."""
    return _clamp01(
        W_COVERAGE * _clamp01(coverage)
        + W_VALID_RATE * _clamp01(valid_rate)
        + W_PLAN * _clamp01(plan_cov)
        + W_CRITIC * critic_component(final_status)
    )


def evaluate_run(
    result: ResearchRunResult,
    now: datetime | None = None,
) -> EvaluationResult:
    """Compute the deterministic EvaluationResult for a finished run."""
    citation_map = citations.assign_citation_ids(result.evidence)
    valid_ids = set(citation_map)

    coverage = citations.citation_coverage(result.report.markdown)
    valid_rate = valid_citation_rate(result.report.markdown, valid_ids)
    plan_cov = plan_coverage(
        list(result.plan.sub_questions), result.report.markdown
    )
    summary = result.evidence.retrieval_summary
    total = summary.total_sources
    web_pct = (summary.web_sources / total * 100.0) if total else 0.0
    kb_pct = (summary.kb_sources / total * 100.0) if total else 0.0

    return EvaluationResult(
        citation_coverage=round(_clamp01(coverage), 4),
        valid_citation_rate=round(_clamp01(valid_rate), 4),
        plan_coverage=round(_clamp01(plan_cov), 4),
        critic_status=(
            result.critique.status
            if result.critique is not None
            else result.final_status
        ),
        iteration_count=result.iteration_count,
        total_sources=total,
        web_sources=summary.web_sources,
        kb_sources=summary.kb_sources,
        web_percentage=round(web_pct, 2),
        kb_percentage=round(kb_pct, 2),
        grounding_score=round(
            grounding_score(coverage, valid_rate, plan_cov, result.final_status),
            4,
        ),
        timestamp=(now or datetime.now(timezone.utc)).isoformat(),
        query=result.query,
    )
