"""Hallucination/grounding ablation benchmark: baseline vs grounded vs full.

An ADDITIVE benchmark on top of the existing research pipeline. It never
changes pipeline behavior; it only invokes it at three escalation levels:

    baseline   raw LLM call: question only. No retrieval, no planner, no
               citation constraints, no critic, no revision.
    grounded   existing Planner -> Retriever -> Writer (citation-constrained),
               Critic/revision BYPASSED.
    agentdesk  the complete production pipeline via `run_research_pipeline`
               (Planner -> Retriever -> Writer -> Critic -> revision loop).

METRIC (see eval/claims.py):
    unsupported_claim_rate =
        (unsupported + contradicted claims) / total evaluated claims
    judged per claim against the question's AUTHORITATIVE REFERENCE CORPUS
    from the benchmark dataset — never against the answer's own citations.
    Citation coverage is reported SEPARATELY as a mechanical heuristic.

EVALUATION TIERS (never conflated):
    automated_heuristic  claim extraction heuristic + LLM-judge protocol
                         (same configured LLM, strict structured output,
                         temperature 0). An approximation, NOT ground truth.
    manual_verification  a human audits the preserved per-claim records.
                         Only manually verified numbers may be cited as
                         verified measurements.

REPRODUCIBILITY: every record stores timestamp, benchmark version, model,
temperatures, dataset sha256, variant, run index, raw answer/report,
retrieved source IDs, per-claim verdicts with justifications, counts,
citation metrics, critic status, iterations, latencies, and errors.
API keys are never recorded.

If execution is blocked (missing keys, quota, unfetchable reference corpus,
failed generation or judging) the affected metric is reported as undefined
(error row / None) — never fabricated.

CLI (see eval/benchmark.py, which delegates to this module):
    python -m eval.benchmark --all                  # live ablation, all variants
    python -m eval.benchmark --variant baseline     # single variant
    python -m eval.benchmark --all --mock           # offline smoke run (NOT a measurement)
    python -m eval.benchmark --all --dataset path.json --runs 2 --output out.json
    python -m eval.benchmark --all --resume eval/results/hallucination-partial.json
        # resumable: completed cells are skipped, new rows appended to the
        # JSONL incrementally, and a provider daily-token (TPD) exhaustion
        # stops the run cleanly; re-run the same command later to continue.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from pydantic import BaseModel

from agents import citations
from agents.groq_structured import (
    parse_structured_output,
    request_structured_completion,
    strict_schema,
)
from eval.claims import (
    CLASSIFICATION_LABELS,
    ClaimEvaluation,
    ClaimJudgeOutput,
    ClaimVerdict,
    absolute_rate_reduction,
    classify_claims,
    extract_factual_claims,
    relative_rate_reduction,
    unsupported_claim_rate,
)
from eval.dataset import (
    BENCHMARK_VERSION,
    DATASET_PATH,
    BenchmarkDataset,
    BenchmarkQuestion,
    dataset_sha256,
    load_dataset,
)
from eval.scorer import valid_citation_rate

VARIANTS: tuple[str, ...] = ("baseline", "grounded", "agentdesk")

# Provider rate-limit markers (Groq HTTP 429 body text). A per-MINUTE limit
# (TPM) only fails the affected cell — the run continues and the cell stays
# unfinished for --resume to retry. A per-DAY exhaustion (TPD) makes the
# benchmark STOP CLEANLY instead of churning failed rows; execution resumes
# later with --resume (completed cells are never re-run).
TPM_MARKER = "tokens per minute"
TPD_MARKER = "tokens per day"
INTERRUPTED_TPD = "tokens_per_day_exhausted"

BASELINE_TEMPERATURE = 0.3   # matches the Writer temperature for fairness
JUDGE_TEMPERATURE = 0.0      # deterministic judging protocol

CORPUS_FETCH_TIMEOUT = 15.0
MAX_CORPUS_CHARS_PER_SOURCE = 6000
MAX_CORPUS_CHARS_TOTAL = 12000
JUDGE_MAX_TOKENS = 6000

RESULTS_DIR = Path(__file__).resolve().parent / "results"

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


class BaselineAnswer(BaseModel):
    """Strict transport schema for the ungrounded baseline completion.

    The schema wraps ONLY the transport (native structured outputs); it does
    not constrain or steer the answer content beyond "single string".
    """

    answer: str


# ---------------------------------------------------------------------------
# Authoritative reference corpus
# ---------------------------------------------------------------------------

def _strip_html(raw: str) -> str:
    text = _TAG_RE.sub(" ", raw)
    return _WS_RE.sub(" ", text).strip()


def build_reference_corpus(
    question: BenchmarkQuestion,
    fetch_url=None,
) -> tuple[str, str, list[str]]:
    """Assemble the authoritative corpus for one question.

    Prefers manually curated inline `content`; otherwise fetches the question's
    reference URLs live (plain GET, no search). Returns
    (corpus_text, status, urls_used) where status is one of:
        "inlined"     curated text present, no fetching needed
        "fetched"     all reference URLs fetched successfully
        "partial"     some URLs fetched (unfetchable ones are skipped)
        "unavailable" no corpus could be established -> metric stays unpopulated
    """
    inlined: list[str] = []
    urls: list[str] = []
    failures = 0

    fetch = fetch_url or _fetch_url_text
    for source in question.reference_sources:
        if (source.content or "").strip():
            title = source.title or source.url or "reference"
            inlined.append(f"[{title}] {source.content.strip()}")
        elif source.url:
            text = fetch(source.url)
            if text:
                title = source.title or source.url
                urls.append(source.url)
                inlined.append(f"[{title} — {source.url}] {text}")
            else:
                failures += 1

    if not inlined:
        return "", "unavailable", urls
    corpus = "\n\n".join(inlined)[:MAX_CORPUS_CHARS_TOTAL]
    if failures:
        status = "partial"
    elif question.has_inlined_content:
        status = "inlined"
    else:
        status = "fetched"
    return corpus, status, urls


def _fetch_url_text(url: str) -> str | None:
    """Fetch one reference URL and return truncated plain text (None on failure).

    Uses plain httpx GET — no search API, no API key involved.
    """
    try:
        response = httpx.get(
            url, timeout=CORPUS_FETCH_TIMEOUT, follow_redirects=True,
            headers={"User-Agent": "AgentDesk-benchmark/1.0"},
        )
        if response.status_code != 200:
            return None
        return _strip_html(response.text)[:MAX_CORPUS_CHARS_PER_SOURCE]
    except (httpx.HTTPError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Variant generation (existing pipeline components only — no behavior changes)
# ---------------------------------------------------------------------------

def baseline_answer(question_text: str) -> tuple[str, float]:
    """Direct LLM call with the question only. Returns (answer, latency)."""
    from config import get_config

    cfg = get_config()
    system = (
        "You are a helpful assistant with broad knowledge. Answer the user's "
        "research question directly, factually, and concisely."
    )
    user = question_text
    started = time.perf_counter()
    content = request_structured_completion(
        system, user, cfg,
        schema_name="baseline_answer",
        schema=strict_schema(BaselineAnswer),
        temperature=BASELINE_TEMPERATURE,
        max_tokens=2048,
    )
    parsed = parse_structured_output(content, BaselineAnswer)
    return parsed.answer, time.perf_counter() - started


def grounded_writer_report(question_text: str, retrieval_mode: str = "both"):
    """Planner -> Retriever -> Writer with the Critic/revision stage bypassed.

    Composes the SAME production components used by the pipeline, without
    modifying any of them. Returns (result_dict, latency_seconds) where
    result_dict carries plan/evidence/report fields for the record.
    """
    from agents.planner import create_research_plan
    from agents.retriever import retrieve_for_plan
    from agents.writer import write_report

    started = time.perf_counter()
    plan = create_research_plan(question_text)
    evidence = retrieve_for_plan(plan, mode=retrieval_mode)
    report = write_report(plan, evidence)
    latency = time.perf_counter() - started
    return {
        "report_title": report.title,
        "raw_text": report.markdown,
        "retrieved_source_ids": [
            s.source_id or s.url or s.title or f"source-{i}"
            for i, s in enumerate(evidence.sources, start=1)
        ],
        "total_sources": evidence.retrieval_summary.total_sources,
        "web_sources": evidence.retrieval_summary.web_sources,
        "kb_sources": evidence.retrieval_summary.kb_sources,
    }, latency


def agentdesk_pipeline_result(question_text: str, retrieval_mode: str = "both"):
    """Full production pipeline (Planner -> Retriever -> Writer -> Critic)."""
    from agents.orchestrator import run_research_pipeline

    started = time.perf_counter()
    result = run_research_pipeline(question_text, retrieval_mode=retrieval_mode)
    latency = time.perf_counter() - started
    return {
        "report_title": result.report.title,
        "raw_text": result.report.markdown,
        "retrieved_source_ids": [
            s.source_id or s.url or s.title or f"source-{i}"
            for i, s in enumerate(result.evidence.sources, start=1)
        ],
        "total_sources": result.evidence.retrieval_summary.total_sources,
        "web_sources": result.evidence.retrieval_summary.web_sources,
        "kb_sources": result.evidence.retrieval_summary.kb_sources,
        "critic_status": (
            result.critique.status if result.critique else result.final_status
        ),
        "iteration_count": result.iteration_count,
        "final_status": result.final_status,
    }, latency


# ---------------------------------------------------------------------------
# LLM-judge claim classification (automated tier — explicitly NOT ground truth)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are a strict, skeptical fact-checking judge.
You classify factual claims against a fixed evidence corpus.

Rules:
1. ONLY the provided evidence corpus counts. Your own knowledge is irrelevant.
2. A claim is "supported" ONLY if the corpus directly entails it. Topic
   overlap is NOT support. Approximate numbers that the corpus does not
   state are NOT support.
3. A claim is "contradicted" ONLY if the corpus directly contradicts it.
4. A claim is "unsupported" if it is on-topic but the corpus does not
   substantiate it.
5. A claim is "unverifiable" if the corpus cannot settle it (too vague,
   outside the corpus, purely evaluative).
6. NEVER treat citation markers like [1] as evidence. They carry no weight.
7. For every verdict give a short justification and, when possible, quote
   the exact corpus passage that decides the classification."""

JUDGE_USER_TEMPLATE = """QUESTION:
{question}

EVIDENCE CORPUS (the ONLY admissible evidence):
---
{corpus}
---

CLAIMS (classify every index):
{claims_json}

Return one verdict per claim index using the strict schema. Classification
must be exactly one of: supported, unsupported, contradicted, unverifiable.
In evidence_quote, quote the exact corpus passage that decides the
classification, or use an empty string when none applies."""


def _judge_schema() -> dict:
    """Strict-mode JSON schema for the judge, built flat and by hand.

    `strict_schema()` in agents.groq_structured only normalizes the ROOT of a
    Pydantic-generated schema. The judge is the only schema in the codebase
    with a NESTED model, and Pydantic renders nested models through $defs,
    where the helper does not strip titles/defaults, add
    additionalProperties=false, or complete the required list — Groq rejects
    such a payload with HTTP 400. This schema is flat (inline items, no
    $defs), every field required, no defaults, no optional/null unions —
    exactly what strict structured outputs demand. Parsing still validates
    against ClaimJudgeOutput via parse_structured_output.
    """
    verdict = {
        "type": "object",
        "properties": {
            "claim_index": {"type": "integer", "minimum": 0},
            "classification": {
                "type": "string",
                "enum": list(CLASSIFICATION_LABELS),
            },
            "justification": {"type": "string"},
            "evidence_quote": {"type": "string"},
        },
        "required": [
            "claim_index", "classification", "justification", "evidence_quote",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"verdicts": {"type": "array", "items": verdict}},
        "required": ["verdicts"],
        "additionalProperties": False,
    }


def judge_claims(
    question: BenchmarkQuestion,
    corpus: str,
    claims: list[str],
) -> tuple[list[ClaimVerdict], float]:
    """LLM-judge a batch of claims against the corpus. Returns (verdicts, latency)."""
    from config import get_config

    cfg = get_config()
    claims_json = json.dumps(
        [{"index": i, "claim": text} for i, text in enumerate(claims)],
        ensure_ascii=False,
    )
    system = JUDGE_SYSTEM_PROMPT
    user = JUDGE_USER_TEMPLATE.format(
        question=question.question, corpus=corpus, claims_json=claims_json,
    )
    started = time.perf_counter()
    content = request_structured_completion(
        system, user, cfg,
        schema_name="claim_judgement",
        schema=_judge_schema(),
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS,
    )
    parsed = parse_structured_output(content, ClaimJudgeOutput)
    return parsed.verdicts, time.perf_counter() - started


def _merge_verdicts(
    claims: list[str], verdicts: list[ClaimVerdict]
) -> tuple[list[ClaimEvaluation], int]:
    """Pair verdicts with claims by index; missing verdicts are preserved and
    flagged (counted as unverifiable with an explicit justification)."""
    by_index = {v.claim_index: v for v in verdicts}
    evaluations: list[ClaimEvaluation] = []
    missing = 0
    for index, text in enumerate(claims):
        verdict = by_index.get(index)
        if verdict is None:
            missing += 1
            evaluations.append(
                ClaimEvaluation(
                    claim=text, claim_index=index,
                    classification="unverifiable",
                    justification="judge did not return a verdict for this claim",
                )
            )
        else:
            evaluations.append(
                ClaimEvaluation(
                    claim=text, claim_index=index,
                    classification=verdict.classification,
                    justification=verdict.justification,
                    # Empty quote means "none provided" in audit records.
                    evidence_quote=verdict.evidence_quote or None,
                )
            )
    return evaluations, missing


# ---------------------------------------------------------------------------
# Per-question/variant records
# ---------------------------------------------------------------------------

def _base_record(
    question: BenchmarkQuestion,
    variant: str,
    run_index: int,
    execution_mode: str,
    model: str | None,
) -> dict:
    return {
        "question_id": question.id,
        "question": question.question,
        "domain": question.domain,
        "variant": variant,
        "run_index": run_index,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "execution_mode": execution_mode,
        "model": model,
        "temperature": BASELINE_TEMPERATURE,
        "judge_temperature": JUDGE_TEMPERATURE,
        "source_curation": question.source_curation,
        "corpus_status": None,
        "corpus_urls_used": [],
        "generation": {
            "kind": "answer" if variant == "baseline" else "report",
            "title": None,
            "raw_text": None,
            "retrieved_source_ids": [],
            "latency_seconds": None,
            "error": None,
        },
        "claims": [],
        "claim_counts": None,
        "unsupported_claim_rate": None,
        "judge_missing_verdicts": None,
        "citation_coverage": None,
        "valid_citation_rate": None,
        "critic_status": None,
        "iteration_count": None,
        "final_status": None,
        "judging_latency_seconds": None,
        "error": None,
    }


def _citation_metrics(raw_text: str, retrieved_source_ids: list[str]) -> tuple[float | None, float | None]:
    """Mechanical citation heuristics for grounded reports (separate from the
    unsupported-claim metric by design). Baseline answers have no citation
    contract, so metrics are None there."""
    if not raw_text or "[" not in raw_text:
        return None, None
    valid_ids = set(range(1, len(retrieved_source_ids) + 1))
    coverage = citations.citation_coverage(raw_text)
    rate = valid_citation_rate(raw_text, valid_ids)
    return round(coverage, 4), round(rate, 4)


def run_question_variant(
    dataset: BenchmarkDataset,
    question: BenchmarkQuestion,
    variant: str,
    run_index: int,
    retrieval_mode: str = "both",
    execution_mode: str = "live",
    fetch_url=None,
) -> dict:
    """Run one question through one benchmark variant and judge its claims.

    Failures at any stage are recorded on the row (error field, rate None);
    the benchmark never aborts and never invents a metric.
    """
    record = _base_record(question, variant, run_index, execution_mode, model=None)
    try:
        from config import get_config

        record["model"] = get_config().groq_model
    except Exception:  # noqa: BLE001 - config always loads in practice
        record["model"] = None

    # 1. Authoritative corpus (needed for judging in every variant).
    if execution_mode == "mock":
        corpus, corpus_status, urls = (
            (
                f"MOCK CORPUS for {question.id}: deterministic offline text used "
                "only to exercise the harness plumbing. NOT real evidence."
            ),
            "mock", [],
        )
    else:
        corpus, corpus_status, urls = build_reference_corpus(
            question, fetch_url=fetch_url
        )
    record["corpus_status"] = corpus_status
    record["corpus_urls_used"] = urls

    if corpus_status == "unavailable":
        record["error"] = (
            "evidence corpus unavailable: reference sources could not be "
            "established (missing curation and unfetchable URLs); metric "
            "left unpopulated"
        )
        return record

    # 2. Generate the answer/report for this variant.
    try:
        if execution_mode == "mock":
            generation, latency = _mock_generation(question, variant)
        elif variant == "baseline":
            answer, latency = baseline_answer(question.question)
            generation = {
                "title": None, "raw_text": answer,
                "retrieved_source_ids": [],
            }
        elif variant == "grounded":
            generation, latency = grounded_writer_report(
                question.question, retrieval_mode=retrieval_mode
            )
        else:  # agentdesk
            generation, latency = agentdesk_pipeline_result(
                question.question, retrieval_mode=retrieval_mode
            )
        record["generation"].update(generation)
        record["generation"]["latency_seconds"] = round(latency, 3)
    except Exception as exc:  # noqa: BLE001 - one failure must not stop the run
        record["generation"]["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        record["error"] = "generation failed; metric left unpopulated"
        return record

    raw_text = generation.get("raw_text") or ""
    source_ids = generation.get("retrieved_source_ids") or []

    # 3. Extract claims (documented heuristic).
    claims = extract_factual_claims(raw_text)
    if not claims:
        record["error"] = "no factual claims extracted; metric left unpopulated"
        record["claim_counts"] = classify_claims([])
        return record

    # 4. Judge claims against the corpus (automated tier).
    try:
        if execution_mode == "mock":
            verdicts, judging_latency = _mock_judge(corpus, claims)
        else:
            verdicts, judging_latency = judge_claims(question, corpus, claims)
        record["judging_latency_seconds"] = round(judging_latency, 3)
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"judging failed: {type(exc).__name__}: {str(exc)[:180]}"
        return record

    evaluations, missing = _merge_verdicts(claims, verdicts)
    record["claims"] = [e.model_dump() for e in evaluations]
    record["judge_missing_verdicts"] = missing
    counts = classify_claims(verdicts)
    record["claim_counts"] = counts
    record["unsupported_claim_rate"] = unsupported_claim_rate(counts)

    # 5. Separate mechanical citation metrics (grounded variants only).
    if variant in ("grounded", "agentdesk"):
        coverage, rate = _citation_metrics(raw_text, source_ids)
        record["citation_coverage"] = coverage
        record["valid_citation_rate"] = rate

    record["critic_status"] = generation.get("critic_status")
    record["iteration_count"] = generation.get("iteration_count")
    record["final_status"] = generation.get("final_status")
    return record


# ---------------------------------------------------------------------------
# Mock mode: deterministic offline plumbing check (NEVER a measurement)
# ---------------------------------------------------------------------------

MOCK_NOTE = (
    "Mock execution: deterministic stubs exercised the harness offline. "
    "These numbers are NOT measurements of real model or pipeline behavior "
    "and must never be reported as benchmark results."
)


def _mock_generation(question: BenchmarkQuestion, variant: str) -> tuple[dict, float]:
    if variant == "baseline":
        text = (
            f"{question.question} This is a deterministic mock baseline answer. "
            "Global malaria deaths fell by roughly forty percent between 2000 "
            "and 2015 according to general knowledge, and internet users "
            "reached about five billion people worldwide in recent years."
        )
        return {"title": None, "raw_text": text, "retrieved_source_ids": []}, 0.001
    report_text = (
        f"# Mock report for {question.id}\n"
        "The mock evidence states a deterministic fact used for plumbing "
        "verification [1].\n\n"
        "A second fully cited sentence keeps the coverage heuristic happy [1].\n\n"
        "## References\n[1] Mock source for "
        f"{question.id} — https://example.invalid/mock"
    )
    extra: dict = {}
    if variant == "agentdesk":
        extra = {
            "critic_status": "approved",
            "iteration_count": 0,
            "final_status": "approved",
        }
    return {
        "title": f"Mock report for {question.id}",
        "raw_text": report_text,
        "retrieved_source_ids": [f"mock:{question.id}"],
        "total_sources": 1, "web_sources": 1, "kb_sources": 0,
        **extra,
    }, 0.002


def _mock_judge(corpus: str, claims: list[str]) -> tuple[list[ClaimVerdict], float]:
    """Deterministic rule: >=3 corpus keyword hits -> supported, else unverifiable.

    Purely mechanical; exists so the full harness path runs offline in tests
    and --mock smoke runs. Explicitly not a real judge.
    """
    corpus_lower = corpus.lower()
    verdicts = []
    for index, claim in enumerate(claims):
        words = [
            w for w in re.findall(r"[a-z]{4,}", claim.lower())
        ][:8]
        hits = sum(1 for w in set(words) if w in corpus_lower)
        classification = "supported" if hits >= 3 else "unverifiable"
        verdicts.append(
            ClaimVerdict(
                claim_index=index, classification=classification,
                justification="deterministic mock rule", evidence_quote="",
            )
        )
    return verdicts, 0.0005


# ---------------------------------------------------------------------------
# Aggregation + persistence
# ---------------------------------------------------------------------------

def summarize_variant(records: list[dict]) -> dict:
    """Aggregate one variant's rows. Failed rows are excluded from metrics."""
    completed = [r for r in records if r["error"] is None]
    totals = {label: 0 for label in ("supported", "unsupported", "contradicted", "unverifiable")}
    for record in completed:
        for label in totals:
            totals[label] += (record.get("claim_counts") or {}).get(label, 0)
    total_claims = sum(totals.values())
    coverage_values = [
        r["citation_coverage"] for r in completed if r["citation_coverage"] is not None
    ]
    valid_values = [
        r["valid_citation_rate"] for r in completed if r["valid_citation_rate"] is not None
    ]
    gen_latencies = [
        r["generation"]["latency_seconds"]
        for r in completed if r["generation"]["latency_seconds"] is not None
    ]
    summary = {
        "questions_total": len(records),
        "questions_completed": len(completed),
        "questions_failed": len(records) - len(completed),
        "total_claims": total_claims,
        "supported_claims": totals["supported"],
        "unsupported_claims": totals["unsupported"],
        "contradicted_claims": totals["contradicted"],
        "unverifiable_claims": totals["unverifiable"],
        "unsupported_claim_rate": (
            round(
                (totals["unsupported"] + totals["contradicted"]) / total_claims, 4
            ) if total_claims else None
        ),
        "avg_generation_latency_seconds": (
            round(sum(gen_latencies) / len(gen_latencies), 3)
            if gen_latencies else None
        ),
    }
    if coverage_values:
        summary["avg_citation_coverage"] = round(
            sum(coverage_values) / len(coverage_values), 4
        )
        summary["avg_valid_citation_rate"] = round(
            sum(valid_values) / len(valid_values), 4
        ) if valid_values else None
    if records and records[0]["variant"] == "agentdesk":
        approved = sum(
            1 for r in completed if r.get("critic_status") == "approved"
        )
        iterations = [
            r["iteration_count"] for r in completed if r["iteration_count"] is not None
        ]
        summary["critic_approval_rate"] = (
            round(approved / len(completed), 4) if completed else None
        )
        summary["avg_iterations"] = (
            round(sum(iterations) / len(iterations), 2) if iterations else None
        )
    return summary


def compare_variants(variant_summaries: dict[str, dict]) -> dict:
    """Baseline-relative reductions. Undefined values stay None — never invented."""
    baseline_rate = (variant_summaries.get("baseline") or {}).get(
        "unsupported_claim_rate"
    )
    comparison: dict = {"baseline_unsupported_claim_rate": baseline_rate}
    for name in ("grounded", "agentdesk"):
        rate = (variant_summaries.get(name) or {}).get("unsupported_claim_rate")
        comparison[f"{name}_unsupported_claim_rate"] = rate
        comparison[f"{name}_absolute_reduction"] = absolute_rate_reduction(
            baseline_rate, rate
        )
        comparison[f"{name}_relative_reduction_percent"] = relative_rate_reduction(
            baseline_rate, rate
        )
    grounded_rate = comparison.get("grounded_unsupported_claim_rate")
    agentdesk_rate = comparison.get("agentdesk_unsupported_claim_rate")
    comparison["agentdesk_vs_grounded_absolute_reduction"] = (
        absolute_rate_reduction(grounded_rate, agentdesk_rate)
    )
    return comparison


def environment_metadata(
    dataset: BenchmarkDataset,
    dataset_path: Path,
    retrieval_mode: str,
    execution_mode: str,
) -> dict:
    """Reproducibility metadata. NEVER includes API keys or other secrets."""
    from config import get_config

    cfg = get_config()
    return {
        "benchmark": "agentdesk-hallucination",
        "benchmark_version": BENCHMARK_VERSION,
        "execution_mode": execution_mode,
        "evaluation_tier": (
            "mock_not_a_measurement"
            if execution_mode == "mock"
            else "automated_heuristic_llm_judge_not_ground_truth"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": cfg.groq_model,
        "temperature_baseline": BASELINE_TEMPERATURE,
        "temperature_judge": JUDGE_TEMPERATURE,
        "retrieval_mode": retrieval_mode,
        "dataset": {
            "path": str(dataset_path),
            "version": dataset.version,
            "benchmark": dataset.benchmark,
            "sha256": dataset_sha256(dataset_path),
            "questions": len(dataset.questions),
        },
        "python": __import__("platform").python_version(),
    }


def run_hallucination_benchmark(
    dataset_path: Path | str | None = None,
    variants: list[str] | None = None,
    runs: int = 1,
    retrieval_mode: str = "both",
    execution_mode: str = "live",
    limit: int | None = None,
    fetch_url=None,
    resume_path: Path | str | None = None,
) -> dict:
    """Run the ablation benchmark. Returns the full machine-readable result.

    Every (question, variant, run) cell is independent: a failure records an
    error row and the benchmark continues. No fabricated numbers anywhere.

    Resumable execution (--resume PATH):
        PATH is a JSONL file of per-cell rows. Successfully completed
        (question_id, variant, run_index) cells are loaded from it, SKIPPED,
        and every newly finished row — success or final failure — is appended
        to the file incrementally as soon as it completes, so an interrupted
        run never loses work. A per-minute rate limit (TPM) only fails that
        cell; the run continues. If the provider reports daily-token
        exhaustion (TPD), the run stops CLEANLY (no retries, no sleeps): the
        partial result is returned with the flag
        `interrupted="tokens_per_day_exhausted"` and can be continued later
        with the same --resume argument. Failed rows already present in the
        file are re-run; successful rows never are.
    """
    chosen_variants = [v for v in (variants or list(VARIANTS)) if v in VARIANTS]
    resolved_path = Path(dataset_path) if dataset_path else DATASET_PATH
    dataset = load_dataset(resolved_path)
    questions = dataset.questions[:limit] if limit else dataset.questions

    rows: list[dict] = []
    done: set[tuple[str, str, int]] = set()
    resume_file = Path(resume_path) if resume_path else None
    if resume_file is not None and resume_file.exists():
        loaded_rows, done = _load_resume_rows(resume_file)
        rows.extend(loaded_rows)

    interrupted: str | None = None
    for run_index in range(1, runs + 1):
        for question in questions:
            for variant in chosen_variants:
                key = (question.id, variant, run_index)
                if key in done:
                    continue
                row = run_question_variant(
                    dataset, question, variant, run_index,
                    retrieval_mode=retrieval_mode,
                    execution_mode=execution_mode,
                    fetch_url=fetch_url,
                )
                rows.append(row)
                if resume_file is not None:
                    _append_resume_row(resume_file, row)
                if _is_tpd_error(row):
                    interrupted = INTERRUPTED_TPD
                    break
                done.add(key)
            if interrupted:
                break
        if interrupted:
            break

    variant_summaries = {
        variant: summarize_variant(
            [r for r in rows if r["variant"] == variant]
        )
        for variant in chosen_variants
    }
    result = {
        "environment": environment_metadata(
            dataset, resolved_path, retrieval_mode, execution_mode
        ),
        "interrupted": interrupted,
        "note": MOCK_NOTE if execution_mode == "mock" else (
            "Automated LLM-judge protocol: unsupported-claim rates are "
            "automated-heuristic estimates, NOT ground truth and NOT manually "
            "verified. Manual verification of per-claim records is required "
            "before citing any number as a verified measurement."
        ),
        "summary": {
            "variants": variant_summaries,
            "comparison": compare_variants(variant_summaries),
        },
        "per_question": rows,
    }
    return result


def _is_tpd_error(row: dict) -> bool:
    """True when the row failed because the provider's daily token cap (TPD)
    is exhausted. Such a failure is not cell-specific: the whole run stops."""
    if not row.get("error"):
        return False
    blob = (row.get("error") or "") + "|" + (
        (row.get("generation") or {}).get("error") or ""
    )
    return TPD_MARKER in blob


def _is_tpm_error(row: dict) -> bool:
    """True when the row failed on a per-minute rate limit (TPM). Unlike TPD
    this is cell-scoped: the run continues and --resume retries the cell."""
    if not row.get("error"):
        return False
    blob = (row.get("error") or "") + "|" + (
        (row.get("generation") or {}).get("error") or ""
    )
    return TPM_MARKER in blob


def _load_resume_rows(
    path: Path,
) -> tuple[list[dict], set[tuple[str, str, int]]]:
    """Load successfully completed cells from a partial-results JSONL.

    Failed rows are ignored (they will be re-run); malformed lines are
    skipped. Duplicate successful cells keep their first occurrence.
    """
    rows: list[dict] = []
    done: set[tuple[str, str, int]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or row.get("error"):
            continue
        key = (row.get("question_id"), row.get("variant"), row.get("run_index"))
        if key in done:
            continue
        done.add(key)
        rows.append(row)
    return rows, done


def _append_resume_row(path: Path, row: dict) -> None:
    """Persist one finished row incrementally (crash-safe append)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def save_results(result: dict, out_path: Path | str | None = None) -> Path:
    """Persist one benchmark result as machine-readable JSON; returns the path."""
    if out_path:
        path = Path(out_path)
    else:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        mode = result.get("environment", {}).get("execution_mode", "live")
        path = RESULTS_DIR / f"hallucination-{mode}-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI (invoked through python -m eval.benchmark --variant/--all/...)
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hallucination/grounding ablation benchmark: "
            "baseline vs grounded writer vs full AgentDesk."
        )
    )
    parser.add_argument("--dataset", default=None, help="Dataset JSON path.")
    parser.add_argument("--runs", type=int, default=1, help="Runs per question.")
    parser.add_argument("--variant", action="append", dest="variants",
                        choices=list(VARIANTS),
                        help="Variant to run (repeatable).")
    parser.add_argument("--all", action="store_true", dest="run_all",
                        help="Run all three variants.")
    parser.add_argument("--output", default=None, help="Output JSON path.")
    parser.add_argument("--mock", action="store_true",
                        help="Offline deterministic smoke run (NOT a measurement).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N questions.")
    parser.add_argument("--resume", default=None,
                        help="Partial-results JSONL: skip completed cells, "
                             "append new rows incrementally, stop cleanly on "
                             "provider daily-token exhaustion.")
    return parser


def run_cli(args: argparse.Namespace) -> int:
    """Execute the hallucination benchmark from parsed benchmark.py CLI args."""
    variants = list(VARIANTS) if args.run_all else (args.variants or list(VARIANTS))
    execution_mode = "mock" if args.mock else "live"
    resume_path = getattr(args, "resume", None)
    result = run_hallucination_benchmark(
        dataset_path=args.dataset,
        variants=variants,
        runs=max(1, args.runs or 1),
        retrieval_mode="both",
        execution_mode=execution_mode,
        limit=args.limit,
        resume_path=resume_path,
    )
    print(json.dumps(result["summary"], indent=2))
    if not getattr(args, "no_save", False):
        path = save_results(result, args.output)
        print(f"Saved: {path}")
    tpm_failed = sum(1 for r in result["per_question"] if _is_tpm_error(r))
    if tpm_failed:
        print(
            f"NOTE: {tpm_failed} cell(s) failed on a per-minute rate limit "
            "(TPM). They are recorded as failed rows with no metric, and "
            "--resume will retry them."
        )
    if result.get("interrupted") == INTERRUPTED_TPD:
        completed = sum(1 for r in result["per_question"] if not r["error"])
        print(
            f"INTERRUPTED: provider daily token limit (TPD) reached. "
            f"{completed} completed rows preserved in "
            f"{resume_path or 'the results file'}. Resume later with:\n"
            f"  python -m eval.benchmark --all --resume {resume_path}"
        )
        return 2
    if execution_mode == "mock":
        print(MOCK_NOTE)
    return 0
