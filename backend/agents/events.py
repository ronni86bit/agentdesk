"""Structured SSE events for research pipeline progress.

The orchestrator emits `ResearchEvent` instances through an optional
callback; the streaming endpoint frames them as Server-Sent Events:

    event: <name>\n
    data: <json>\n
    \n

Event payloads are safe for direct rendering in a browser UI: they never
contain API keys, authorization headers, raw provider responses, prompts,
stack traces, or internal exception objects.
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class EventNames:
    """Stable vocabulary of streamable research events."""

    RESEARCH_STARTED = "research_started"
    PLANNER_STARTED = "planner_started"
    PLANNER_COMPLETED = "planner_completed"
    RETRIEVAL_STARTED = "retrieval_started"
    RETRIEVAL_COMPLETED = "retrieval_completed"
    WRITER_STARTED = "writer_started"
    WRITER_COMPLETED = "writer_completed"
    CRITIC_STARTED = "critic_started"
    CRITIC_COMPLETED = "critic_completed"
    REVISION_STARTED = "revision_started"
    REVISION_COMPLETED = "revision_completed"
    RESEARCH_COMPLETED = "research_completed"
    RESEARCH_FAILED = "research_failed"


class ResearchEvent(BaseModel):
    """One JSON-serializable progress event on the research stream."""

    event: str
    stage: str | None = None
    status: str | None = None
    message: str | None = None
    data: dict[str, Any] | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    iteration: int | None = None


def format_sse(event: ResearchEvent) -> str:
    """Frame one event using valid Server-Sent Events syntax."""
    return f"event: {event.event}\ndata: {event.model_dump_json()}\n\n"


# User-safe failure mapping: no exception details ever reach the client.
def safe_failure(exc: Exception) -> tuple[str, str]:
    """Map a pipeline exception to (safe_error_code, user_safe_message)."""
    from agents.critic import CriticError, MissingGroqKeyError as CriticMissingKey
    from agents.planner import (
        MissingGroqKeyError as PlannerMissingKey,
        PlannerError,
    )
    from agents.writer import InsufficientEvidenceError, WriterError
    from tools.retrieval import RetrievalError

    if isinstance(exc, (PlannerMissingKey, CriticMissingKey)):
        return "llm_not_configured", "LLM provider is not configured."
    if isinstance(exc, ValueError):
        return "invalid_request", str(exc)
    # Provider rate limits are detected via the structured errors' Retry-After
    # metadata or the normalized HTTP status embedded in the message — never
    # via raw exception text.
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None or (
        isinstance(exc, (PlannerError, WriterError, CriticError))
        and "HTTP 429" in str(exc)
    ):
        return (
            "rate_limited",
            "AI provider rate limit reached. Please try again later.",
        )
    if isinstance(exc, PlannerError):
        return "planner_failed", "Planning research questions failed."
    if isinstance(exc, RetrievalError):
        return "retrieval_failed", str(exc)
    if isinstance(exc, InsufficientEvidenceError):
        return "insufficient_evidence", str(exc)
    if isinstance(exc, WriterError):
        return "writer_failed", "Writing the grounded report failed."
    if isinstance(exc, CriticError):
        return "critic_failed", "Auditing the report failed."
    return "internal_error", "Research failed unexpectedly."
