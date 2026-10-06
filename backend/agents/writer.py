"""Writer Agent: synthesizes a grounded, cited report from retrieved evidence.

The Writer has NO search tools. It only sees the numbered sources provided by
the backend; all reference metadata is constructed deterministically from the
EvidenceBundle after generation.

Structured output uses Groq's native Structured Outputs (`response_format`
`json_schema` with `strict: true`) via the shared `groq_structured` client,
with the JSON Schema generated from the Pydantic `ReportDraft` model — so a
malformed draft can never reach deterministic citation validation silently
and no Instructor retry loops are involved.
"""

import time

from crewai import Agent, LLM

from agents import citations
from agents.groq_structured import (
    MAX_BACKOFF_SECONDS,
    DEFAULT_BACKOFF_SECONDS,
    agent_system_prompt,
    parse_structured_output,
    request_structured_completion,
    source_budget_chars,
    strict_schema,
)
from agents.schemas import (
    CritiqueResult,
    EvidenceBundle,
    Report,
    ReportDraft,
    ResearchPlan,
)
from config import get_config, groq_litellm_model

DRAFT_SCHEMA_NAME = "report_draft"
MAX_LLM_ATTEMPTS = 2  # bounded; never an unbounded retry loop


class WriterError(Exception):
    """Base error for writer failures."""

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        rate_limit_scope: str | None = None,
    ):
        super().__init__(message)
        self.retry_after = retry_after
        # "tpm"/"tpd" when the failure was a provider rate limit that named
        # its bucket; None otherwise.
        self.rate_limit_scope = rate_limit_scope


class InsufficientEvidenceError(WriterError):
    """No usable evidence was supplied; refusing to generate a report."""


WRITER_ROLE = "Grounded Research Report Writer"
WRITER_GOAL = (
    "Write clear, well-structured research reports that synthesize ONLY the "
    "supplied evidence, with accurate inline numeric citations and no "
    "invented facts or sources."
)
WRITER_BACKSTORY = (
    "You are a rigorous research writer. You never invent sources, URLs, "
    "statistics, organizations, or study results. Every factual claim you "
    "make is grounded in the numbered evidence you were given, and you cite "
    "it inline like [1] or [2][5]. When evidence is missing, conflicting, or "
    "too weak, you say so explicitly instead of filling gaps from your own "
    "knowledge."
)

TASK_DESCRIPTION = """Write a professional research report answering the
research goal, using ONLY the numbered sources provided below.

Research goal: {goal}

Research plan (sub-questions to structure the report around):
{sub_questions}

Numbered sources (you may ONLY cite these IDs, inline as [N]):
{sources}
{citation_constraints}
Strict rules:
1. Cite ONLY source IDs from the list above. NEVER invent a citation number,
   URL, document name, statistic, organization, or study result.
2. Synthesize across sources in your own words; do not copy long passages.
3. Every factual claim drawn from evidence MUST carry at least one inline
   citation, e.g. "... expanded across diagnostics [1][3]."
4. If the evidence does not support a claim strongly enough: omit it, qualify
   it, or state explicitly that evidence is insufficient.
5. Distinguish evidence from interpretation.
6. Structure: title, executive summary, sections that follow the research
   plan, key findings/conclusion, then a "## References" section listing each
   cited ID as:
   [N] Title — URL (web) or Document (kb)
7. Keep the report focused; aim for roughly 600-800 words.
8. Use plain ASCII characters only: standard [N] brackets for citations,
   straight quotes, and regular dashes/hyphens. Do NOT use Unicode bracket
   variants (e.g. 【1】) or special typographic characters."""

TASK_EXPECTED_OUTPUT = (
    "A complete markdown research report with inline [N] citations restricted "
    "to the provided source IDs."
)

REVISION_BLOCK_TEMPLATE = """

You are REVISING your previous report after a quality audit (iteration {iteration}).

Your previous report:
---
{previous_markdown}
---

Auditor verdict: {status}

Citation issues:
{citation_issues}

Unsupported claims:
{unsupported_claims}

Logical issues:
{logical_issues}

Missing evidence / coverage gaps:
{missing_evidence}

Revision instructions (follow these exactly):
{revision_instructions}

Revision rules:
1. Revise the EXISTING report above. Preserve valid content wherever
   possible; change ONLY what the audit requires.
2. Use ONLY the numbered sources listed above. Do NOT add new sources,
   citations, statistics, organizations, or dates from anywhere else.
3. Fix every issue raised: remove or qualify unsupported claims, repair or
   remove invalid/irrelevant citations, close coverage gaps using the
   existing evidence, and state explicitly when evidence is insufficient.
4. NEVER silently drop all citations to avoid errors — every factual claim
   drawn from evidence must keep a valid inline [N] citation.
5. Output the COMPLETE revised report in the same structure and rules as
   above, not a diff.
"""


def _citation_constraints(citation_map: dict[int, object]) -> str:
    """Make the valid citation range extremely explicit for the model."""
    if not citation_map:
        return ""
    max_id = max(citation_map)
    range_text = (
        f"[1]" if max_id == 1 else f"[1] through [{max_id}]"
    )
    return f"""

AVAILABLE CITATIONS: {range_text} (exactly {len(citation_map)} source(s) are provided).
You MUST NOT emit any citation number outside this range. A citation number
above [{max_id}] or below 1 does NOT exist and will cause the report to be
rejected. Every inline citation you write must be an integer between 1 and
{max_id}.
"""


def _format_revision_block(previous_report: Report, critique: CritiqueResult) -> str:

    def _bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "- None"

    return REVISION_BLOCK_TEMPLATE.format(
        iteration=critique.iteration,
        previous_markdown=previous_report.markdown,
        status=critique.status,
        citation_issues=_bullets(critique.citation_issues),
        unsupported_claims=_bullets(critique.unsupported_claims),
        logical_issues=_bullets(critique.logical_issues),
        missing_evidence=_bullets(critique.missing_evidence),
        revision_instructions=_bullets(critique.revision_instructions),
    )


def _build_llm(cfg) -> LLM:
    if not cfg.has_groq_key:
        raise WriterError(
            "LLM provider is not configured. Set GROQ_API_KEY in the environment."
        )
    return LLM(
        model=groq_litellm_model(cfg.groq_model),
        api_key=cfg.groq_api_key,
        temperature=0.3,
    )


# ---------------------------------------------------------------------------
# Native Groq structured output (strict JSON Schema)
# ---------------------------------------------------------------------------

def report_draft_json_schema() -> dict:
    """Build a strict-mode-compatible JSON Schema from ReportDraft."""
    return strict_schema(ReportDraft)


def _request_structured_draft(
    system_prompt: str,
    user_prompt: str,
    cfg,
) -> str:
    """Call Groq chat completions with native strict JSON-schema output."""
    try:
        return request_structured_completion(
            system_prompt,
            user_prompt,
            cfg,
            schema_name=DRAFT_SCHEMA_NAME,
            schema=report_draft_json_schema(),
            temperature=0.3,
            # Bounds report length so downstream Critic prompts (which embed
            # the full report) stay inside the model's per-request budget.
            max_tokens=4096,
        )
    except Exception as exc:
        raise WriterError(
            str(exc),
            retry_after=getattr(exc, "retry_after", None),
            rate_limit_scope=getattr(exc, "rate_limit_scope", None),
        ) from exc


def _parse_draft(content: str) -> ReportDraft:
    """Parse and validate raw JSON text into a ReportDraft."""
    try:
        result = parse_structured_output(content, ReportDraft)
    except Exception as exc:
        raise WriterError(f"Writer {exc}") from exc
    return result  # type: ignore[no-any-return]


def _generate_report_draft(
    plan: ResearchPlan,
    evidence: EvidenceBundle,
    previous_report: Report | None,
    critique: CritiqueResult | None,
    cfg,
) -> ReportDraft:
    """Compose prompts and obtain a validated ReportDraft via Groq.

    Bounded to MAX_LLM_ATTEMPTS attempts total; retries cover transient
    transport/API failures only. Deterministic content-level failures
    (e.g. hallucinated citations) are NOT retried here.
    """
    # The CrewAI Writer Agent stays the semantic owner of role/goal/backstory.
    writer_agent = Agent(
        role=WRITER_ROLE,
        goal=WRITER_GOAL,
        backstory=WRITER_BACKSTORY,
        llm=_build_llm(cfg),
        verbose=False,
    )
    system_prompt = agent_system_prompt(
        writer_agent.role, writer_agent.goal, writer_agent.backstory
    )
    citation_map = citations.assign_citation_ids(evidence)
    constraints = _citation_constraints(citation_map)
    revision_block = (
        _format_revision_block(previous_report, critique)
        if critique is not None
        else ""
    )
    # Size the numbered-source block so the WHOLE prompt (including the
    # previous report during revisions) stays inside the model's per-request
    # token budget; free Groq tiers reject anything larger outright.
    fixed_chars = len(
        TASK_DESCRIPTION.format(
            goal=plan.original_query,
            sub_questions="\n".join(f"- {q}" for q in plan.sub_questions),
            sources="",
            citation_constraints=constraints,
        )
    ) + len(revision_block) + len(TASK_EXPECTED_OUTPUT) + 10
    budget = source_budget_chars(fixed_chars)
    description = TASK_DESCRIPTION.format(
        goal=plan.original_query,
        sub_questions="\n".join(f"- {q}" for q in plan.sub_questions),
        sources=citations.format_sources_for_prompt(
            citation_map,
            max_chars_per_source=min(1000, budget),
            max_total_chars=budget,
        ),
        citation_constraints=constraints,
    )
    if revision_block:
        description += revision_block
    user_prompt = description + "\n\n" + TASK_EXPECTED_OUTPUT

    last_error: WriterError | None = None
    for attempt in range(MAX_LLM_ATTEMPTS):
        try:
            content = _request_structured_draft(system_prompt, user_prompt, cfg)
            return _parse_draft(content)
        except WriterError as exc:
            last_error = exc
            if exc.rate_limit_scope == "tpd":
                # A daily-token cap cannot clear inside the backoff window;
                # surface it immediately instead of retrying.
                raise
            if attempt < MAX_LLM_ATTEMPTS - 1:
                # Explicit bounded backoff; honors Retry-After when present.
                delay = exc.retry_after if exc.retry_after is not None else DEFAULT_BACKOFF_SECONDS
                time.sleep(min(delay, MAX_BACKOFF_SECONDS))
    assert last_error is not None
    raise last_error


def write_report(
    plan: ResearchPlan,
    evidence: EvidenceBundle,
    previous_report: Report | None = None,
    critique: CritiqueResult | None = None,
) -> Report:
    """Synthesize a grounded report from a plan and its evidence bundle.

    For revisions, pass BOTH `previous_report` and `critique`; the Writer
    then fixes only the issues raised by the audit, using the SAME evidence
    bundle (no new retrieval, no new sources).

    Raises:
        ValueError: if the plan has no sub-questions or revision args mismatch.
        InsufficientEvidenceError: if there is no evidence to ground the report.
        WriterError: on LLM failure, malformed output, or invalid citations.
    """
    if not plan.sub_questions:
        raise ValueError("plan must contain at least one sub-question")
    if not evidence.sources:
        raise InsufficientEvidenceError(
            "Insufficient evidence to generate a grounded report."
        )
    if (previous_report is None) != (critique is None):
        raise ValueError(
            "previous_report and critique must be provided together for a revision"
        )

    cfg = get_config()
    citation_map = citations.assign_citation_ids(evidence)

    draft = _generate_report_draft(
        plan, evidence, previous_report, critique, cfg
    )
    if not draft.markdown.strip():
        raise WriterError("Writer returned malformed structured output")

    invalid = citations.validate_report_citations(
        draft.markdown, set(citation_map)
    )
    if invalid:
        raise WriterError(
            f"Writer produced invalid citation IDs not present in evidence: {invalid}"
        )

    return Report(
        title=draft.title.strip() or plan.original_query,
        markdown=draft.markdown,
        references=citations.build_references(draft.markdown, citation_map),
    )
