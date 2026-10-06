"""Benchmark harness: run a fixed set of research queries and collect metrics.

Produces a defensible quality snapshot per query: grounding score, citation
coverage, critic outcome, iterations, source balance, status, and latency.
Each query runs through the SAME `run_research_pipeline` used by the API —
no separate evaluation path.

Usage (manual, consumes live provider quota):
    cd backend
    python -m eval.benchmark                 # default query set, live run
    python -m eval.benchmark --query "..."   # single custom query
    python -m eval.benchmark --mode kb       # pin retrieval mode

Automated tests call `run_benchmark` with a patched pipeline instead —
no live calls happen in CI-style tests.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from agents.orchestrator import run_research_pipeline
from eval.scorer import evaluate_run

DEFAULT_QUERIES = [
    "Research the impact of AI on healthcare in India",
    "Compare solar and wind energy for grid-scale electricity generation",
    "What are the main challenges of adopting electric vehicles in Europe?",
]

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def benchmark_record(result) -> dict:
    """Compact, JSON-serializable record for one finished run."""
    evaluation = evaluate_run(result)
    timing = result.timing
    return {
        "query": result.query,
        "final_status": result.final_status,
        "iteration_count": result.iteration_count,
        "critic_status": evaluation.critic_status,
        "citation_coverage": evaluation.citation_coverage,
        "valid_citation_rate": evaluation.valid_citation_rate,
        "plan_coverage": evaluation.plan_coverage,
        "grounding_score": evaluation.grounding_score,
        "total_sources": evaluation.total_sources,
        "web_sources": evaluation.web_sources,
        "kb_sources": evaluation.kb_sources,
        "total_seconds": timing.total_seconds if timing else None,
        "report_title": result.report.title,
        "error": None,
    }


def _error_record(query: str, message: str) -> dict:
    return {
        "query": query,
        "final_status": "failed",
        "iteration_count": None,
        "critic_status": None,
        "citation_coverage": None,
        "valid_citation_rate": None,
        "plan_coverage": None,
        "grounding_score": None,
        "total_sources": None,
        "web_sources": None,
        "kb_sources": None,
        "total_seconds": None,
        "report_title": None,
        "error": message,
    }


def run_benchmark(
    queries: list[str] | None = None,
    retrieval_mode: str = "both",
    event_callback=None,
) -> list[dict]:
    """Run every query through the pipeline and collect metric records.

    A failing query (e.g. provider rate limit) is recorded as a failed row;
    the benchmark never aborts on a single failure.
    """
    records: list[dict] = []
    for query in queries or DEFAULT_QUERIES:
        started = time.perf_counter()
        try:
            result = run_research_pipeline(
                query, event_callback=event_callback,
                retrieval_mode=retrieval_mode,
            )
            record = benchmark_record(result)
        except Exception as exc:  # noqa: BLE001 - one failure must not stop the run
            safe_message = str(exc) or type(exc).__name__
            record = _error_record(
                query.strip(),
                f"{type(exc).__name__}: {safe_message[:200]}",
            )
        record["retrieval_mode"] = retrieval_mode
        elapsed = time.perf_counter() - started
        if record["total_seconds"] is None:
            record["total_seconds"] = round(elapsed, 3)
        records.append(record)
    return records


def summarize(records: list[dict]) -> dict:
    """Aggregate summary over benchmark records (failed rows excluded)."""
    completed = [r for r in records if r["error"] is None]
    scores = [r["grounding_score"] for r in completed]
    return {
        "queries": len(records),
        "completed": len(completed),
        "failed": len(records) - len(completed),
        "avg_grounding_score": round(sum(scores) / len(scores), 4) if scores else None,
        "approved": sum(1 for r in completed if r["final_status"] == "approved"),
        "avg_iterations": (
            round(sum(r["iteration_count"] for r in completed) / len(completed), 2)
            if completed else None
        ),
    }


def save_results(records: list[dict], summary: dict, out_dir: Path = RESULTS_DIR) -> Path:
    """Persist one timestamped benchmark run as JSON; returns the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"benchmark-{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "summary": summary,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the AgentDesk research benchmark (live provider quota)."
    )
    parser.add_argument("--query", action="append", dest="queries",
                        help="Custom query (repeatable); overrides defaults.")
    parser.add_argument("--mode", default="both", choices=["both", "web", "kb"],
                        help="Retrieval mode (default: both).")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not write a results JSON file.")
    # --- Hallucination ablation benchmark (additive; see eval.hallucination) ---
    parser.add_argument("--variant", action="append", dest="variants",
                        choices=["baseline", "grounded", "agentdesk"],
                        help="Ablation variant to run (repeatable).")
    parser.add_argument("--all", action="store_true", dest="run_all",
                        help="Ablation: run baseline + grounded + agentdesk.")
    parser.add_argument("--dataset", default=None,
                        help="Ablation: benchmark dataset JSON path.")
    parser.add_argument("--runs", type=int, default=None,
                        help="Ablation: runs per question (default 1).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Ablation: only run the first N questions.")
    parser.add_argument("--output", default=None,
                        help="Ablation: output JSON path.")
    parser.add_argument("--mock", action="store_true",
                        help="Ablation: offline deterministic smoke run "
                             "(NOT a measurement).")
    parser.add_argument("--resume", default=None,
                        help="Ablation: partial-results JSONL — skip completed "
                             "cells, persist new rows incrementally, stop "
                             "cleanly on provider daily-token exhaustion.")
    args = parser.parse_args()

    # Any ablation flag routes to the hallucination benchmark harness.
    if args.variants or args.run_all or args.mock or any(
        v is not None for v in (args.dataset, args.runs, args.limit,
                                args.output, args.resume)
    ):
        from eval.hallucination import run_cli

        return run_cli(args)

    records = run_benchmark(queries=args.queries, retrieval_mode=args.mode)
    summary = summarize(records)
    print(json.dumps({"summary": summary, "records": records}, indent=2))
    if not args.no_save:
        path = save_results(records, summary)
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
