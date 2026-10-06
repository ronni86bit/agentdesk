"""Critic Agent: audits the Writer's report against plan and evidence.

The Critic is a grounding quality gate. It has NO search tools and NO access
to outside knowledge for corrections: it may only flag whether the report is
supported by the provided EvidenceBundle. When evidence does not support a
claim it says the claim is unsupported by the provided evidence; it never
substitutes a "correct" fact of its own.

Structured output uses Groq's native Structured Outputs (`response_format`
`json_schema` with `strict: true`) via the shared `groq_structured` client,
with the JSON Schema generated directly from the Pydantic `CritiqueResult`
model. The CrewAI Critic Agent definition is preserved for role/semantic
metadata; malformed output can never reach the orchestrator as an "approved"
verdict because every response is re-validated by Pydantic.
"""

import time

from crewai import Agent, LLM

from agents import citations
from agents.groq_structured import (
    MAX_BACKOFF_SECONDS,
    DEFAULT_BACKOFF_SECONDS,
    agent_system_prompt,
    parse_reset_seconds,
    parse_structured_output,
    request_structured_completion,
    source_budget_chars,
    strict_schema,
)
from agents.schemas import CritiqueResult, EvidenceBundle, Report, ResearchPlan
from config import get_config, groq_litellm_model

CRITIQUE_SCHEMA_NAME = "critique_result"
MAX_LLM_ATTEMPTS = 2  # bounded; never an unbounded retry loop
# Per-source cap for the numbered evidence block; its total size is computed
# dynamically (source_budget_chars) so the whole prompt fits the tier budget.
SOURCE_PROMPT_MAX_CHARS_PER_SOURCE = 500


class CriticError(Exception):
    """Base error for critic failures."""

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        rate_limit_scope: str | None = None,
    ):
        super().__init__(message)
        # Seconds the provider asked us to wait (HTTP 429); None otherwise.
        self.retry_after = retry_after
        # "tpm"/"tpd" when the failure was a provider rate limit that named
        # its bucket; None otherwise.
        self.rate_limit_scope = rate_limit_scope


class MissingGroqKeyError(CriticError):
    """GROQ_API_KEY is not configured."""


CRITIC_ROLE = "Research Quality and Citation Auditor"
CRITIC_GOAL = (
    "Decide whether a research report is factually grounded in the supplied "
    "evidence, and if not, issue precise, actionable revision instructions."
)
CRITIC_BACKSTORY = (
    "You are a meticulous research auditor. You verify that every factual "
    "claim in a report is supported by the numbered evidence you were given, "
    "that citations are relevant and complete, that planned sub-questions are "
    "covered, and that reasoning is logically sound. You are conservative: "
    "you do not flag ordinary transitions or style issues as failures. You "
    "never use your own world knowledge to supply missing facts; if the "
    "provided evidence does not support something, you say exactly that."
)

TASK_HEADER = """Audit the following research report for factual grounding.

Report iteration under review: {iteration}

Audit criteria (evaluate each):
A. Citation validity — every cited [N] must map to an actual source AND that
   source must genuinely support the claim it is attached to. Flag citations
   that are irrelevant to their claims or claims broader than the source.
B. Unsupported factual claims — identify statistics, named organizations,
   dates, causal claims, or comparisons that cannot be supported by the
   numbered evidence. Be conservative: ordinary reasoning, transitions, and
   structural statements are NOT hallucinations.
C. Evidence fidelity — check the report accurately represents source content;
   flag exaggeration, changed meaning, or unrelated sources combined into a
   misleading conclusion.
D. Research-plan coverage — every major planned sub-question must be addressed
   (semantic coverage is enough; identical wording is not required). Flag only
   important sub-questions that were completely ignored.
E. Logical quality — contradictions, unsupported causal leaps, excessive
   certainty, misleading conclusions, obvious logical gaps.
F. Citation completeness — evidence-dependent factual claims must carry at
   least one citation. Headings, transitions, and generic organizational
   statements do NOT need citations.

Grounding constraint (critical):
- Use ONLY the numbered evidence below when judging support.
- NEVER propose replacement facts, statistics, dates, or organizations from
  your own knowledge. If something is unsupported by the provided evidence,
  describe it as unsupported by the provided evidence.

Pass/fail policy:
- APPROVED only if there are no critical unsupported claims, no invalid or
  materially irrelevant citations, no major evidence distortion, major
  planned questions are covered, and citation coverage is reasonably complete.
- REVISION_NEEDED if important claims are unsupported, important citations
  are missing/irrelevant, evidence is materially misrepresented, major
  planned questions are missing, or serious logical problems exist.
- Do NOT reject a report for minor wording/style issues.

If status is revision_needed, revision_instructions MUST be concrete and
actionable (name the claim to remove/qualify and why, or name the statement
and which existing source [N] supports it). If approved, all issue lists and
revision_instructions must be empty.
"""

TASK_INPUT_TEMPLATE = """
Research goal: {goal}

Research plan sub-questions:
{sub_questions}

Numbered evidence sources (the ONLY permitted support):
{sources}

Report under audit:
---
{report_markdown}
---
"""


def _build_llm(cfg) -> LLM:
    if not cfg.has_groq_key:
        raise MissingGroqKeyError(
            "LLM provider is not configured. Set GROQ_API_KEY in the environment."
        )
    return LLM(
        model=groq_litellm_model(cfg.groq_model),
        api_key=cfg.groq_api_key,
        temperature=0.0,
    )


# ---------------------------------------------------------------------------
# Native Groq structured output (strict JSON Schema)
# ---------------------------------------------------------------------------

def critique_json_schema() -> dict:
    """Build a strict-mode-compatible JSON Schema from CritiqueResult."""
    return strict_schema(CritiqueResult)


# Kept as a module-level alias for backward-compatible test patching.
_parse_reset_seconds = parse_reset_seconds


def _request_structured_critique(
    system_prompt: str,
    user_prompt: str,
    cfg,
) -> str:
    """Call Groq chat completions with native strict JSON-schema output.

    Returns the raw message content string. Raises CriticError on any HTTP or
    API-level failure (preserving provider backoff info). Never logs or
    embeds the API key in error messages.
    """
    try:
        return request_structured_completion(
            system_prompt,
            user_prompt,
            cfg,
            schema_name=CRITIQUE_SCHEMA_NAME,
            schema=critique_json_schema(),
            temperature=0.0,
            max_tokens=4096,
        )
    except Exception as exc:
        raise CriticError(
            str(exc),
            retry_after=getattr(exc, "retry_after", None),
            rate_limit_scope=getattr(exc, "rate_limit_scope", None),
        ) from exc


def _parse_critique(content: str) -> CritiqueResult:
    """Parse and validate raw JSON text into a CritiqueResult.

    Raises CriticError for malformed JSON or semantically invalid data;
    invalid critiques never propagate beyond this function.
    """
    try:
        result = parse_structured_output(content, CritiqueResult)
    except Exception as exc:
        raise CriticError(f"Critic {exc}") from exc
    return result  # type: ignore[no-any-return]


def _generate_critique_structured(
    plan: ResearchPlan,
    evidence: EvidenceBundle,
    report: Report,
    iteration: int,
    cfg,
) -> CritiqueResult:
    """Compose prompts, call Groq with strict structured output, validate.

    Bounded to MAX_LLM_ATTEMPTS attempts total (covers transient transport
    errors and semantically-invalid responses). Raises CriticError when all
    attempts fail; never fabricates a fallback critique and never silently
    approves the report.
    """
    # The CrewAI Critic Agent stays the semantic owner of role/goal/backstory.
    critic_agent = Agent(
        role=CRITIC_ROLE,
        goal=CRITIC_GOAL,
        backstory=CRITIC_BACKSTORY,
        llm=_build_llm(cfg),
        verbose=False,
    )
    system_prompt = agent_system_prompt(
        critic_agent.role, critic_agent.goal, critic_agent.backstory
    )
    header = TASK_HEADER.format(iteration=iteration)
    template_args = {
        "goal": plan.original_query,
        "sub_questions": "\n".join(f"- {q}" for q in plan.sub_questions),
        "sources": "",
        "report_markdown": report.markdown,
    }
    # Size the numbered-source block so the WHOLE prompt stays inside the
    # model's per-request token budget (free Groq tiers reject larger ones).
    fixed_chars = len(header) + len(TASK_INPUT_TEMPLATE.format(**template_args))
    budget = source_budget_chars(fixed_chars)
    template_args["sources"] = citations.format_sources_for_prompt(
        citations.assign_citation_ids(evidence),
        max_chars_per_source=min(SOURCE_PROMPT_MAX_CHARS_PER_SOURCE, budget),
        max_total_chars=budget,
    )
    user_prompt = header + TASK_INPUT_TEMPLATE.format(**template_args)

    last_error: CriticError | None = None
    for attempt in range(MAX_LLM_ATTEMPTS):
        try:
            content = _request_structured_critique(system_prompt, user_prompt, cfg)
            return _parse_critique(content)
        except CriticError as exc:
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


def evaluate_report(
    plan: ResearchPlan,
    evidence: EvidenceBundle,
    report: Report,
    iteration: int = 0,
) -> CritiqueResult:
    """Run the Critic Agent and return a validated CritiqueResult.

    Raises:
        ValueError: if iteration is negative.
        CriticError: on missing key, request failure, or invalid output.
    """
    if iteration < 0:
        raise ValueError("iteration must be >= 0")

    cfg = get_config()
    if not cfg.has_groq_key:
        raise MissingGroqKeyError(
            "LLM provider is not configured. Set GROQ_API_KEY in the environment."
        )
    return _generate_critique_structured(plan, evidence, report, iteration, cfg)
