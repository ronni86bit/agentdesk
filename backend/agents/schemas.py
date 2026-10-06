"""Structured output models shared across AgentDesk agents."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


SourceType = Literal["web", "kb"]

# User-selectable retrieval mode: which evidence sources may be queried.
RetrievalMode = Literal["both", "web", "kb"]
RETRIEVAL_MODES: tuple[str, ...] = ("both", "web", "kb")


class EvidenceSource(BaseModel):
    """One normalized retrieved source (web page or knowledge-base chunk)."""

    source_type: SourceType
    title: str = ""
    url: str | None = None
    document_name: str | None = None
    content: str = ""
    score: float | None = None
    source_id: str | None = None


class RetrievalSummary(BaseModel):
    """Diagnostics for one evidence-bundle retrieval run.

    A side that was not queried because of the retrieval mode reports
    status "skipped".
    """

    total_sources: int
    web_sources: int
    kb_sources: int
    deduplicated_sources: int
    kb_status: Literal["ok", "empty", "unavailable", "skipped"]
    web_status: Literal["ok", "empty", "unavailable", "skipped"]


class EvidenceBundle(BaseModel):
    """Normalized, deduplicated evidence passed downstream to the Writer."""

    original_query: str
    sources: list[EvidenceSource]
    retrieval_summary: RetrievalSummary


class ReportReference(BaseModel):
    """Authoritative reference metadata, built by the backend from evidence.

    The LLM never supplies URLs or document metadata; citation numbers in the
    report text map to these backend-constructed entries.
    """

    citation_id: int
    source_type: SourceType
    title: str = ""
    url: str | None = None
    document_name: str | None = None
    source_id: str | None = None


class Report(BaseModel):
    """Final grounded research report for one stage of the pipeline."""

    title: str
    markdown: str
    references: list[ReportReference]


class ReportDraft(BaseModel):
    """LLM-controlled portion of the report (no authoritative metadata)."""

    title: str
    markdown: str


class ResearchPlan(BaseModel):
    """Decomposition of a user's research goal into searchable sub-questions."""

    original_query: str
    sub_questions: list[str] = Field(min_length=3, max_length=5)

    @field_validator("original_query", "sub_questions", mode="before")
    @classmethod
    def reject_blank(cls, value):
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("must not be blank")
            return value
        if isinstance(value, list):
            cleaned = [item.strip() for item in value if isinstance(item, str) and item.strip()]
            if len(cleaned) != len(value):
                raise ValueError("sub_questions must not contain blank entries")
            return cleaned
        return value


class CritiqueResult(BaseModel):
    """Structured quality-gate verdict produced by the Critic Agent.

    The Critic evaluates grounding only (citations, evidence support, plan
    coverage, logic). It must never propose replacement facts from its own
    knowledge; issue descriptions reference the provided evidence.
    """

    status: Literal["approved", "revision_needed"]
    iteration: int = Field(ge=0)
    citation_issues: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    logical_issues: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    revision_instructions: list[str] = Field(default_factory=list)
    overall_assessment: str = ""

    @model_validator(mode="after")
    def check_status_consistency(self):
        if self.status == "approved":
            if self.revision_instructions:
                raise ValueError(
                    "approved critique must not contain revision_instructions"
                )
        else:
            if not self.revision_instructions:
                raise ValueError(
                    "revision_needed critique requires concrete revision_instructions"
                )
        return self


FinalStatus = Literal["approved", "max_revisions_reached", "revision_failed"]


class RunTiming(BaseModel):
    """Wall-clock stage durations (seconds) for one pipeline run.

    Telemetry only: used by the UI run summary and the benchmark harness.
    """

    planner_seconds: float = 0.0
    retrieval_seconds: float = 0.0
    writer_seconds: float = 0.0
    critic_seconds: float = 0.0
    revision_seconds: float = 0.0
    total_seconds: float = 0.0


class EvaluationResult(BaseModel):
    """Deterministic, heuristic evaluation of one finished research run.

    Computed by `eval.scorer` from the final `ResearchRunResult` with no
    additional LLM calls. All rates are heuristics, not measures of factual
    truth; see `eval/scorer.py` for the exact formulas.
    """

    citation_coverage: float = Field(ge=0.0, le=1.0)
    valid_citation_rate: float = Field(ge=0.0, le=1.0)
    plan_coverage: float = Field(ge=0.0, le=1.0)
    critic_status: str
    iteration_count: int = Field(ge=0)
    total_sources: int = Field(ge=0)
    web_sources: int = Field(ge=0)
    kb_sources: int = Field(ge=0)
    web_percentage: float = Field(ge=0.0, le=100.0)
    kb_percentage: float = Field(ge=0.0, le=100.0)
    grounding_score: float = Field(ge=0.0, le=1.0)
    timestamp: str
    query: str


class ResearchRunResult(BaseModel):
    """Serializable result of one full research pipeline run.

    `evidence` is the single EvidenceBundle used for the initial report AND
    every revision; it never changes during the loop. `iteration_count` is the
    number of successful revisions incorporated (0-2). `evaluation` and
    `timing` are attached at the end of the run by the orchestrator.
    """

    query: str
    plan: ResearchPlan
    evidence: EvidenceBundle
    report: Report
    critique: CritiqueResult | None = None
    iteration_count: int = Field(default=0, ge=0)
    final_status: FinalStatus
    retrieval_mode: RetrievalMode = "both"
    evaluation: EvaluationResult | None = None
    timing: RunTiming | None = None
