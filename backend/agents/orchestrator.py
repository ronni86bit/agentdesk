"""Research pipeline orchestration: Planner -> Retriever -> Writer -> Critic.

This module is the single service boundary between FastAPI and the agents.
FastAPI never touches CrewAI internals; it calls `run_research_pipeline` and
receives a serializable `ResearchRunResult`.

Self-correction loop:
    iteration 0 = initial Writer output
    revision 1..MAX_REVISIONS = revised outputs, each audited again
The loop always operates on the SAME EvidenceBundle (no new retrieval during
revision), so evaluation is reproducible and evidence cannot shift.

Retrieval modes: `retrieval_mode` ("both" | "web" | "kb") selects which
evidence sources the Retriever may query; "both" (default) preserves the
original behavior exactly.

Streaming support: an optional `event_callback(ResearchEvent)` receives
progress events at meaningful stage boundaries. The callback is passive —
the pipeline logic is identical whether or not one is supplied.

Telemetry + evaluation: stage wall-clock durations are recorded in
`RunTiming`, and a deterministic `EvaluationResult` (no extra LLM calls) is
computed by `eval.scorer` and attached to the final result.
"""

import time
from dataclasses import dataclass

from agents.critic import evaluate_report
from agents.events import EventNames, ResearchEvent, safe_failure
from agents.planner import create_research_plan
from agents.retriever import retrieve_for_plan
from agents.schemas import ResearchRunResult, RunTiming
from agents.writer import write_report
from eval.scorer import evaluate_run

MAX_REVISIONS = 2


def _emit(callback, event: str, **fields) -> None:
    """Send one event to the callback if streaming is enabled."""
    if callback is None:
        return
    callback(ResearchEvent(event=event, **fields))


def _issue_count(critique) -> int:
    return (
        len(critique.citation_issues)
        + len(critique.unsupported_claims)
        + len(critique.logical_issues)
        + len(critique.missing_evidence)
    )


@dataclass
class _StageClock:
    """Accumulates wall-clock seconds per pipeline stage."""

    planner: float = 0.0
    retrieval: float = 0.0
    writer: float = 0.0
    critic: float = 0.0
    revision: float = 0.0

    class Span:
        def __init__(self, clock: "_StageClock", attr: str):
            self.clock = clock
            self.attr = attr
            self.start = 0.0

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *exc_info):
            setattr(
                self.clock,
                self.attr,
                getattr(self.clock, self.attr) + (time.perf_counter() - self.start),
            )
            return False

    def span(self, attr: str) -> "Span":
        return _StageClock.Span(self, attr)

    def as_model(self, total: float) -> RunTiming:
        return RunTiming(
            planner_seconds=round(self.planner, 3),
            retrieval_seconds=round(self.retrieval, 3),
            writer_seconds=round(self.writer, 3),
            critic_seconds=round(self.critic, 3),
            revision_seconds=round(self.revision, 3),
            total_seconds=round(total, 3),
        )


def run_research_pipeline(
    query: str,
    event_callback=None,
    retrieval_mode: str = "both",
) -> ResearchRunResult:
    """Run the full pipeline with up to MAX_REVISIONS critique-driven revisions.

    Args:
        query: the user's research goal.
        event_callback: optional callable receiving `ResearchEvent` progress
            events; pass None (default) for silent operation.
        retrieval_mode: "both" (default), "web", or "kb" — which evidence
            sources the Retriever may query.

    Raises:
        ValueError: if query is empty or retrieval_mode is invalid.
        PlannerError / RetrievalError / WriterError / CriticError: from the
            underlying stage; initial-stage failures propagate to the caller
            after emitting a safe `research_failed` event (when streaming).
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if retrieval_mode not in ("both", "web", "kb"):
        raise ValueError(f"invalid retrieval mode: {retrieval_mode!r}")
    clean_query = query.strip()

    clock = _StageClock()
    run_started = time.perf_counter()

    def fail(exc: Exception) -> None:
        code, message = safe_failure(exc)
        _emit(
            event_callback,
            EventNames.RESEARCH_FAILED,
            status="failed",
            data={"error_code": code, "message": message},
        )

    def finish(result: ResearchRunResult) -> ResearchRunResult:
        """Attach timing + deterministic evaluation, emit completion event."""
        result.timing = clock.as_model(time.perf_counter() - run_started)
        result.evaluation = evaluate_run(result)
        _emit(
            event_callback,
            EventNames.RESEARCH_COMPLETED,
            status=result.final_status,
            iteration=result.iteration_count,
            data={
                "final_status": result.final_status,
                "iteration_count": result.iteration_count,
                "result": result.model_dump(),
            },
        )
        return result

    try:
        _emit(
            event_callback,
            EventNames.RESEARCH_STARTED,
            data={"query": clean_query, "retrieval_mode": retrieval_mode},
        )

        _emit(
            event_callback,
            EventNames.PLANNER_STARTED,
            stage="planner",
            status="started",
            message="Planning research questions",
        )
        with clock.span("planner"):
            plan = create_research_plan(clean_query)
        _emit(
            event_callback,
            EventNames.PLANNER_COMPLETED,
            stage="planner",
            status="completed",
            data={"sub_questions": len(plan.sub_questions)},
        )

        _emit(
            event_callback,
            EventNames.RETRIEVAL_STARTED,
            stage="retrieval",
            status="started",
            message="Retrieving evidence from knowledge base and web",
        )
        with clock.span("retrieval"):
            evidence = retrieve_for_plan(plan, mode=retrieval_mode)
        summary = evidence.retrieval_summary
        _emit(
            event_callback,
            EventNames.RETRIEVAL_COMPLETED,
            stage="retrieval",
            status="completed",
            data={
                "retrieval_mode": retrieval_mode,
                "total_sources": summary.total_sources,
                "web_sources": summary.web_sources,
                "kb_sources": summary.kb_sources,
                "deduplicated_sources": summary.deduplicated_sources,
                "kb_status": summary.kb_status,
                "web_status": summary.web_status,
            },
        )

        _emit(
            event_callback,
            EventNames.WRITER_STARTED,
            stage="writer",
            status="started",
            message="Writing grounded report",
        )
        with clock.span("writer"):
            report = write_report(plan, evidence)
        _emit(
            event_callback,
            EventNames.WRITER_COMPLETED,
            stage="writer",
            status="completed",
            data={
                "citation_count": len(report.references),
                "report_title": report.title,
            },
        )

        _emit(
            event_callback,
            EventNames.CRITIC_STARTED,
            stage="critic",
            status="started",
            iteration=0,
            message="Checking report against retrieved evidence",
        )
        with clock.span("critic"):
            critique = evaluate_report(plan, evidence, report, iteration=0)
        _emit(
            event_callback,
            EventNames.CRITIC_COMPLETED,
            stage="critic",
            status="completed",
            iteration=0,
            data={
                "status": critique.status,
                "iteration": critique.iteration,
                "issue_count": _issue_count(critique),
            },
        )
        iteration_count = 0

        while critique.status == "revision_needed" and iteration_count < MAX_REVISIONS:
            next_iteration = iteration_count + 1
            _emit(
                event_callback,
                EventNames.REVISION_STARTED,
                stage="revision",
                status="started",
                iteration=next_iteration,
                message="Revising report from audit feedback",
            )
            try:
                with clock.span("revision"):
                    revised = write_report(
                        plan,
                        evidence,
                        previous_report=report,
                        critique=critique,
                    )
            except Exception:
                # A failed revision (e.g. hallucinated citations rejected by the
                # deterministic validator) is a Writer failure. Never let an
                # invalid report through: keep the last valid one and stop safely.
                result = ResearchRunResult(
                    query=clean_query,
                    plan=plan,
                    evidence=evidence,
                    report=report,
                    critique=critique,
                    iteration_count=iteration_count,
                    final_status="revision_failed",
                    retrieval_mode=retrieval_mode,
                )
                return finish(result)
            report = revised
            iteration_count = next_iteration
            _emit(
                event_callback,
                EventNames.REVISION_COMPLETED,
                stage="revision",
                status="completed",
                iteration=next_iteration,
            )
            _emit(
                event_callback,
                EventNames.CRITIC_STARTED,
                stage="critic",
                status="started",
                iteration=iteration_count,
                message="Checking report against retrieved evidence",
            )
            with clock.span("critic"):
                critique = evaluate_report(plan, evidence, report, iteration=iteration_count)
            _emit(
                event_callback,
                EventNames.CRITIC_COMPLETED,
                stage="critic",
                status="completed",
                iteration=iteration_count,
                data={
                    "status": critique.status,
                    "iteration": critique.iteration,
                    "issue_count": _issue_count(critique),
                },
            )

        if critique.status == "approved":
            final_status = "approved"
        else:
            # Quality gate still not passed after the maximum allowed revisions.
            final_status = "max_revisions_reached"

        result = ResearchRunResult(
            query=clean_query,
            plan=plan,
            evidence=evidence,
            report=report,
            critique=critique,
            iteration_count=iteration_count,
            final_status=final_status,
            retrieval_mode=retrieval_mode,
        )
        return finish(result)
    except Exception as exc:
        fail(exc)
        raise
