"""Planner Agent: decomposes a research goal into searchable sub-questions.

Structured output uses Groq's native Structured Outputs (`response_format`
`json_schema` with `strict: true`) via the shared `groq_structured` client,
with the JSON Schema generated from the Pydantic `ResearchPlan` model — no
Instructor retry loops, and a malformed plan can never be returned silently.
"""

import time

from crewai import Agent, LLM

from agents.groq_structured import (
    MAX_BACKOFF_SECONDS,
    DEFAULT_BACKOFF_SECONDS,
    agent_system_prompt,
    parse_structured_output,
    request_structured_completion,
    strict_schema,
)
from agents.schemas import ResearchPlan
from config import get_config, groq_litellm_model

PLAN_SCHEMA_NAME = "research_plan"
MAX_LLM_ATTEMPTS = 2  # bounded; never an unbounded retry loop


class PlannerError(Exception):
    """Base error for planner failures."""

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


class MissingGroqKeyError(PlannerError):
    """GROQ_API_KEY is not configured."""


PLANNER_ROLE = "Research Planning Specialist"
PLANNER_GOAL = (
    "Decompose any research goal into 3-5 precise, independently searchable "
    "sub-questions with maximal coverage of the topic and zero redundancy."
)
PLANNER_BACKSTORY = (
    "You are a senior research strategist who has designed thousands of "
    "research projects. You excel at breaking broad questions into specific, "
    "retrievable sub-questions that downstream retrieval agents can execute "
    "against web search and vector databases. You never attempt to answer the "
    "question yourself; you only plan."
)

TASK_DESCRIPTION = """Decompose the following research goal into 3-5 specific,
independently searchable sub-questions.

Rules:
1. Each sub-question must be directly relevant to the goal.
2. Each sub-question must be independently searchable (suitable for web search
   or vector retrieval).
3. Sub-questions must not overlap; each covers a distinct dimension
   (e.g., current state, measurable outcomes, challenges, key actors).
4. Be specific: include scope details such as geography, domain, or timeframe
   where the goal implies them.
5. Do NOT answer the questions. Only produce the plan.
6. Preserve the original research goal EXACTLY as given in original_query.

Research goal: {query}"""

TASK_EXPECTED_OUTPUT = (
    "A structured research plan containing the original query and between 3 "
    "and 5 distinct, searchable sub-questions."
)


def _build_llm(cfg) -> LLM:
    if not cfg.has_groq_key:
        raise MissingGroqKeyError(
            "LLM provider is not configured. Set GROQ_API_KEY in the environment."
        )
    return LLM(
        model=groq_litellm_model(cfg.groq_model),
        api_key=cfg.groq_api_key,
        temperature=0.2,
    )


# ---------------------------------------------------------------------------
# Native Groq structured output (strict JSON Schema)
# ---------------------------------------------------------------------------

def research_plan_json_schema() -> dict:
    """Build a strict-mode-compatible JSON Schema from ResearchPlan."""
    return strict_schema(ResearchPlan)


def _request_structured_plan(
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
            schema_name=PLAN_SCHEMA_NAME,
            schema=research_plan_json_schema(),
            temperature=0.2,
            max_tokens=2048,
        )
    except Exception as exc:
        raise PlannerError(
            str(exc),
            retry_after=getattr(exc, "retry_after", None),
            rate_limit_scope=getattr(exc, "rate_limit_scope", None),
        ) from exc


def _parse_plan(content: str) -> ResearchPlan:
    """Parse and validate raw JSON text into a ResearchPlan."""
    try:
        result = parse_structured_output(content, ResearchPlan)
    except Exception as exc:
        raise PlannerError(f"Planner {exc}") from exc
    return result  # type: ignore[no-any-return]


def _generate_plan_structured(user_query: str, cfg) -> ResearchPlan:
    """Compose prompts and obtain a validated ResearchPlan via Groq.

    Bounded to MAX_LLM_ATTEMPTS attempts total; retries cover transient
    transport/API failures only.
    """
    # The CrewAI Planner Agent stays the semantic owner of role/goal/backstory.
    planner_agent = Agent(
        role=PLANNER_ROLE,
        goal=PLANNER_GOAL,
        backstory=PLANNER_BACKSTORY,
        llm=_build_llm(cfg),
        verbose=False,
    )
    system_prompt = agent_system_prompt(
        planner_agent.role, planner_agent.goal, planner_agent.backstory
    )
    user_prompt = TASK_DESCRIPTION.format(query=user_query.strip()) + (
        "\n\n" + TASK_EXPECTED_OUTPUT
    )

    last_error: PlannerError | None = None
    for attempt in range(MAX_LLM_ATTEMPTS):
        try:
            content = _request_structured_plan(system_prompt, user_prompt, cfg)
            return _parse_plan(content)
        except PlannerError as exc:
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


def create_research_plan(user_query: str) -> ResearchPlan:
    """Run the Planner Agent and return a validated ResearchPlan.

    Raises:
        ValueError: if user_query is empty or planner output is invalid.
        MissingGroqKeyError: if GROQ_API_KEY is missing.
        PlannerError: on LLM request or validation failure.
    """
    if not user_query or not user_query.strip():
        raise ValueError("user_query must be a non-empty string")

    cfg = get_config()
    if not cfg.has_groq_key:
        raise MissingGroqKeyError(
            "LLM provider is not configured. Set GROQ_API_KEY in the environment."
        )
    return _generate_plan_structured(user_query, cfg)
