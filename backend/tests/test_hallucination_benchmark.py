"""Tests for the hallucination/grounding ablation benchmark.

Everything runs OFFLINE: generation and judging are either mocked or run in
deterministic mock mode. No test in this file contacts external APIs.
"""

import argparse
import json

import pytest

import eval.benchmark as benchmark_module
import eval.hallucination as hb
from eval.claims import (
    ClaimVerdict,
    absolute_rate_reduction,
    classify_claims,
    extract_factual_claims,
    relative_rate_reduction,
    unsupported_claim_rate,
)
from eval.dataset import BenchmarkQuestion, ReferenceSource, load_dataset


def _question(qid="q01", curation="curated", content="Authoritative text. "
              "The global malaria death toll declined steadily over the decade."):
    return BenchmarkQuestion(
        id=qid,
        question=f"Question {qid}?",
        domain="health",
        source_curation=curation,
        reference_sources=[
            ReferenceSource(title="WHO", url="https://www.who.int/", notes="n",
                            content=content)
        ],
    )


# ---------------------------------------------------------------------------
# Claim extraction heuristic
# ---------------------------------------------------------------------------

def test_extract_factual_claims_filters_noise():
    text = (
        "# Title\n"
        "Global malaria deaths declined steadily over the decade [1].\n"
        "This report summarizes the findings.\n"
        "Short. line.\n"
        "Is this a question?\n"
        "The World Bank estimated poverty at eight percent in 2022 [2].\n"
    )
    claims = extract_factual_claims(text)
    assert len(claims) == 2
    assert all("[1]" not in c and "[2]" not in c for c in claims)
    assert not any(c.lower().startswith("this report") for c in claims)


def test_extract_factual_claims_deduplicates_and_caps():
    text = ("Solar costs fell sharply over the past decade. " * 40)
    claims = extract_factual_claims(text)
    assert len(claims) == 1  # duplicates removed
    many = " ".join(f"Claim number {i} states a distinct factual matter today." for i in range(40))
    assert len(extract_factual_claims(many)) == 30  # documented cap


# ---------------------------------------------------------------------------
# Metric math + zero-division handling
# ---------------------------------------------------------------------------

def test_unsupported_claim_rate_counts_unsupported_and_contradicted():
    counts = {"supported": 6, "unsupported": 2, "contradicted": 1, "unverifiable": 1}
    assert unsupported_claim_rate(counts) == pytest.approx(0.3)


def test_unsupported_claim_rate_zero_division_returns_none():
    assert unsupported_claim_rate({"supported": 0, "unsupported": 0,
                                   "contradicted": 0, "unverifiable": 0}) is None
    assert unsupported_claim_rate({}) is None


def test_relative_reduction_zero_baseline_is_undefined():
    assert relative_rate_reduction(0.0, 0.0) is None
    assert relative_rate_reduction(None, 0.1) is None
    assert relative_rate_reduction(0.4, 0.1) == pytest.approx(75.0)


def test_absolute_reduction_none_safe():
    assert absolute_rate_reduction(None, 0.1) is None
    assert absolute_rate_reduction(0.5, None) is None
    assert absolute_rate_reduction(0.5, 0.2) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Classification aggregation + verdict merging
# ---------------------------------------------------------------------------

def test_classify_claims_aggregates_counts():
    verdicts = [
        ClaimVerdict(claim_index=0, classification="supported"),
        ClaimVerdict(claim_index=1, classification="contradicted"),
        ClaimVerdict(claim_index=2, classification="supported"),
        ClaimVerdict(claim_index=3, classification="unverifiable"),
    ]
    counts = classify_claims(verdicts)
    assert counts == {"supported": 2, "unsupported": 0, "contradicted": 1,
                      "unverifiable": 1}
    assert unsupported_claim_rate(counts) == pytest.approx(0.25)


def test_merge_verdicts_flags_missing_indices():
    claims = ["claim one is here today", "claim two is also present"]
    verdicts = [ClaimVerdict(claim_index=1, classification="supported",
                             justification="quote", evidence_quote="...")]
    evaluations, missing = hb._merge_verdicts(claims, verdicts)
    assert missing == 1
    assert evaluations[0].classification == "unverifiable"
    assert "did not return a verdict" in evaluations[0].justification
    assert evaluations[1].classification == "supported"


# ---------------------------------------------------------------------------
# Dataset loading + hashing
# ---------------------------------------------------------------------------

def test_load_dataset_validates_shipped_file():
    dataset = load_dataset()
    assert dataset.version == "1.1.0"
    assert dataset.benchmark == "agentdesk-hallucination"
    assert len(dataset.questions) == 20
    ids = [q.id for q in dataset.questions]
    assert len(ids) == len(set(ids))
    # v1.1.0: all questions curated with verified inlined evidence.
    assert all(not q.needs_manual_curation for q in dataset.questions)
    assert all(q.has_inlined_content for q in dataset.questions)


def test_load_dataset_rejects_duplicate_ids(tmp_path):
    from pydantic import ValidationError

    payload = {"questions": [{"id": "q1", "question": "a?"}, {"id": "q1", "question": "b?"}]}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_dataset(path)


def test_dataset_sha256_is_content_sensitive(tmp_path):
    from eval.dataset import dataset_sha256
    path = tmp_path / "d.json"
    path.write_text('{"questions": [{"id": "q1", "question": "a?"}]}', encoding="utf-8")
    h1 = dataset_sha256(path)
    path.write_text('{"questions": [{"id": "q2", "question": "a?"}]}', encoding="utf-8")
    h2 = dataset_sha256(path)
    assert h1 != h2


# ---------------------------------------------------------------------------
# Reference corpus
# ---------------------------------------------------------------------------

def test_corpus_prefers_inlined_content_without_fetching():
    question = _question(content="Curated authoritative content.")
    corpus, status, urls = hb.build_reference_corpus(question, fetch_url=lambda u: None)
    assert status == "inlined"
    assert "Curated authoritative content." in corpus
    assert urls == []


def test_corpus_fetches_when_not_curated():
    question = _question(curation="manual_review_required", content=None)
    corpus, status, urls = hb.build_reference_corpus(
        question, fetch_url=lambda u: "<html>Fetched WHO page text about malaria.</html>"
    )
    assert status == "fetched"
    assert "Fetched WHO page text about malaria." in corpus
    assert urls == ["https://www.who.int/"]


def test_corpus_unavailable_leaves_metric_unpopulated():
    question = _question(content=None)
    dataset = __import__("eval.dataset", fromlist=["BenchmarkDataset"]).BenchmarkDataset(
        questions=[question])
    record = hb.run_question_variant(
        dataset, question, "baseline", 1, fetch_url=lambda u: None,
    )
    assert record["corpus_status"] == "unavailable"
    assert record["error"] and "corpus unavailable" in record["error"]
    assert record["unsupported_claim_rate"] is None


def test_corpus_partial_status_when_some_urls_fail():
    question = BenchmarkQuestion(
        id="q05", question="Q?", domain="d",
        reference_sources=[
            ReferenceSource(title="A", url="https://a.example/", content=None),
            ReferenceSource(title="B", url="https://b.example/", content=None),
        ],
    )

    def fetch(url):
        return "fetched text with substance" if "a.example" in url else None

    corpus, status, urls = hb.build_reference_corpus(question, fetch_url=fetch)
    assert status == "partial"
    assert urls == ["https://a.example/"]
    assert "fetched text with substance" in corpus


# ---------------------------------------------------------------------------
# Full harness in mock mode (offline; plumbing verification only)
# ---------------------------------------------------------------------------

def _args(**overrides):
    base = {
        "queries": None, "mode": "both", "no_save": False,
        "variants": None, "run_all": True, "dataset": None, "runs": None,
        "limit": None, "output": None, "mock": True,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_mock_run_produces_flagged_result(tmp_path):
    result = hb.run_hallucination_benchmark(
        variants=["baseline", "grounded", "agentdesk"],
        execution_mode="mock",
        fetch_url=lambda u: None,
    )
    env = result["environment"]
    assert env["execution_mode"] == "mock"
    assert env["evaluation_tier"] == "mock_not_a_measurement"
    assert result["note"] == hb.MOCK_NOTE
    assert "sha256" in env["dataset"]
    assert env["model"]  # recorded
    # One row per variant for every question; rows carry reproducibility fields.
    assert len(result["per_question"]) == 20 * 3
    row = result["per_question"][0]
    for field in ("timestamp", "benchmark_version" , "model", "variant",
                  "run_index", "claims", "claim_counts",
                  "unsupported_claim_rate", "error"):
        assert field in row or field in env
    # Rates are computed from the deterministic mock judge.
    for variant in ("baseline", "grounded", "agentdesk"):
        summary = result["summary"]["variants"][variant]
        assert summary["questions_completed"] == 20
        assert summary["total_claims"] > 0
        assert summary["unsupported_claim_rate"] is not None
    comparison = result["summary"]["comparison"]
    assert comparison["baseline_unsupported_claim_rate"] is not None
    # No secrets anywhere in the serialized result.
    blob = json.dumps(result)
    assert "GROQ_API_KEY" not in blob and "TAVILY_API_KEY" not in blob


def test_mock_run_serializes_and_saves(tmp_path):
    result = hb.run_hallucination_benchmark(
        variants=["baseline"], execution_mode="mock", limit=1,
        fetch_url=lambda u: None,
    )
    path = hb.save_results(result, out_path=tmp_path / "out.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["environment"]["benchmark_version"] == "1.0.0"
    assert len(data["per_question"]) == 1
    row = data["per_question"][0]
    assert row["variant"] == "baseline"
    assert row["generation"]["raw_text"]
    assert row["claims"]


def test_run_cli_mock_end_to_end(tmp_path, capsys):
    args = _args(dataset=None, limit=2, output=str(tmp_path / "mock.json"))
    exit_code = hb.run_cli(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "unsupported_claim_rate" in out
    assert (tmp_path / "mock.json").exists()
    saved = json.loads((tmp_path / "mock.json").read_text(encoding="utf-8"))
    assert saved["environment"]["execution_mode"] == "mock"
    assert len(saved["per_question"]) == 2 * 3


# ---------------------------------------------------------------------------
# Failure handling: generation/judging errors never abort or fabricate
# ---------------------------------------------------------------------------

def test_generation_failure_records_error_row(monkeypatch):
    question = _question()
    dataset = __import__("eval.dataset", fromlist=["BenchmarkDataset"]).BenchmarkDataset(
        questions=[question])

    def boom(question_text):
        raise RuntimeError("LLM request returned HTTP 429")

    monkeypatch.setattr(hb, "baseline_answer", boom)
    record = hb.run_question_variant(
        dataset, question, "baseline", 1, fetch_url=lambda u: None,
    )
    assert "429" in record["generation"]["error"]
    assert record["error"] == "generation failed; metric left unpopulated"
    assert record["unsupported_claim_rate"] is None


def test_judging_failure_records_error_row(monkeypatch):
    question = _question()
    dataset = __import__("eval.dataset", fromlist=["BenchmarkDataset"]).BenchmarkDataset(
        questions=[question])

    def boom(*a, **k):
        raise ValueError("LLM returned an unusable response body")

    monkeypatch.setattr(hb, "judge_claims", boom)
    record = hb.run_question_variant(
        dataset, question, "baseline", 1, fetch_url=lambda u: None,
    )
    assert record["error"].startswith("judging failed")
    assert record["unsupported_claim_rate"] is None
    assert record["generation"]["raw_text"]  # answer preserved for audit


def test_summarize_variant_excludes_failed_rows():
    rows = [
        {"error": None, "claim_counts": {"supported": 3, "unsupported": 1,
                                         "contradicted": 0, "unverifiable": 1},
         "citation_coverage": 0.9, "valid_citation_rate": 1.0,
         "generation": {"latency_seconds": 1.0}, "variant": "grounded",
         "critic_status": None, "iteration_count": None},
        {"error": "generation failed", "claim_counts": None,
         "citation_coverage": None, "valid_citation_rate": None,
         "generation": {"latency_seconds": None}, "variant": "grounded",
         "critic_status": None, "iteration_count": None},
    ]
    summary = hb.summarize_variant(rows)
    assert summary["questions_total"] == 2
    assert summary["questions_completed"] == 1
    assert summary["questions_failed"] == 1
    assert summary["total_claims"] == 5
    assert summary["unsupported_claim_rate"] == pytest.approx(0.2)
    assert summary["avg_citation_coverage"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Per-variant comparison + reproducibility metadata
# ---------------------------------------------------------------------------

def test_compare_variants_math_and_undefined_handling():
    summaries = {
        "baseline": {"unsupported_claim_rate": 0.5},
        "grounded": {"unsupported_claim_rate": 0.25},
        "agentdesk": {"unsupported_claim_rate": 0.10},
    }
    comparison = hb.compare_variants(summaries)
    assert comparison["grounded_absolute_reduction"] == pytest.approx(0.25)
    assert comparison["grounded_relative_reduction_percent"] == pytest.approx(50.0)
    assert comparison["agentdesk_relative_reduction_percent"] == pytest.approx(80.0)
    # Positive = agentdesk rate is lower than grounded (an improvement).
    assert comparison["agentdesk_vs_grounded_absolute_reduction"] == pytest.approx(0.15)


def test_compare_variants_zero_baseline_yields_undefined_not_zero():
    comparison = hb.compare_variants({
        "baseline": {"unsupported_claim_rate": 0.0},
        "grounded": {"unsupported_claim_rate": 0.0},
        "agentdesk": {"unsupported_claim_rate": 0.0},
    })
    assert comparison["grounded_relative_reduction_percent"] is None
    assert comparison["agentdesk_relative_reduction_percent"] is None
    assert comparison["grounded_absolute_reduction"] == pytest.approx(0.0)


def test_environment_metadata_contains_no_secrets():
    from eval.dataset import DATASET_PATH, dataset_sha256
    dataset = load_dataset(DATASET_PATH)
    env = hb.environment_metadata(dataset, DATASET_PATH, "both", "live")
    assert env["evaluation_tier"] == "automated_heuristic_llm_judge_not_ground_truth"
    assert env["model"]
    assert env["dataset"]["sha256"] == dataset_sha256(DATASET_PATH)
    blob = json.dumps(env)
    assert "GROQ_API_KEY" not in blob and "TAVILY_API_KEY" not in blob
    assert env["model"].count("/") <= 1  # model id, not a key
    # The environment block never carries an api_key-shaped field.
    assert "api_key" not in blob.lower()


# ---------------------------------------------------------------------------
# Resumable execution: incremental persistence, skip-completed, clean TPD stop
# ---------------------------------------------------------------------------

def _mk_row(qid: str, variant: str, error: str | None = None,
            rate: float | None = 0.0) -> dict:
    return {
        "question_id": qid, "question": f"Q {qid}?", "domain": "d",
        "variant": variant, "run_index": 1, "timestamp": "t",
        "execution_mode": "live", "model": "m", "error": error,
        "unsupported_claim_rate": rate, "claims": [],
        "claim_counts": {"supported": 0, "unsupported": 0,
                         "contradicted": 0, "unverifiable": 0},
        "citation_coverage": None, "valid_citation_rate": None,
        "critic_status": None, "iteration_count": None, "final_status": None,
        "judging_latency_seconds": None,
        "generation": {"kind": "answer", "title": None, "raw_text": "x",
                       "retrieved_source_ids": [], "latency_seconds": 0.1,
                       "error": None},
    }


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )


def test_resume_skips_completed_cells_and_runs_the_rest(tmp_path, monkeypatch):
    partial = tmp_path / "partial.jsonl"
    _write_jsonl(partial, [_mk_row("q01", "baseline"),
                           _mk_row("q01", "grounded"),
                           _mk_row("q01", "agentdesk")])
    executed = []

    def fake_cell(dataset, question, variant, run_index, **kwargs):
        executed.append((question.id, variant))
        row = _mk_row(question.id, variant, rate=0.5)
        row["generation"]["raw_text"] = (
            "A factual claim sentence with enough words to count as one."
        )
        row["claims"] = [{"claim": "c", "claim_index": 0,
                          "classification": "supported",
                          "justification": "j", "evidence_quote": "q"}]
        row["claim_counts"] = {"supported": 1, "unsupported": 0,
                               "contradicted": 0, "unverifiable": 0}
        return row

    monkeypatch.setattr(hb, "run_question_variant", fake_cell)
    result = hb.run_hallucination_benchmark(
        variants=["baseline", "grounded", "agentdesk"],
        execution_mode="live", limit=2, resume_path=partial,
    )
    # q01 was fully completed in the partial file -> only q02's 3 cells run.
    assert executed == [("q02", "baseline"), ("q02", "grounded"),
                        ("q02", "agentdesk")]
    assert result["interrupted"] is None
    assert len(result["per_question"]) == 6
    # Completed rows were appended incrementally to the same file.
    file_rows = [json.loads(l) for l in
                 partial.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(file_rows) == 6


def test_resume_reruns_failed_rows_but_never_successful_ones(tmp_path, monkeypatch):
    partial = tmp_path / "partial.jsonl"
    _write_jsonl(partial, [
        _mk_row("q01", "baseline"),                     # success -> skipped
        _mk_row("q01", "grounded", error="ConnectError: x"),  # failure -> rerun
    ])
    executed = []

    def fake_cell(dataset, question, variant, run_index, **kwargs):
        executed.append((question.id, variant))
        row = _mk_row(question.id, variant, rate=0.25)
        row["generation"]["raw_text"] = "Some factual statement of real length."
        row["claims"] = [{"claim": "c", "claim_index": 0,
                          "classification": "unsupported",
                          "justification": "j", "evidence_quote": ""}]
        row["claim_counts"] = {"supported": 0, "unsupported": 1,
                               "contradicted": 0, "unverifiable": 0}
        return row

    monkeypatch.setattr(hb, "run_question_variant", fake_cell)
    result = hb.run_hallucination_benchmark(
        variants=["baseline", "grounded"],
        execution_mode="live", limit=1, resume_path=partial,
    )
    assert executed == [("q01", "grounded")]
    assert len(result["per_question"]) == 2
    assert not any(r["error"] for r in result["per_question"])


def test_tpd_exhaustion_stops_run_cleanly_and_persists(tmp_path, monkeypatch):
    partial = tmp_path / "partial.jsonl"
    calls = []

    def fake_cell(dataset, question, variant, run_index, **kwargs):
        calls.append((question.id, variant))
        if (question.id, variant) == ("q01", "grounded"):
            row = _mk_row(question.id, variant,
                          error="WriterError: LLM request returned HTTP 429")
            row["error"] = "generation failed; metric left unpopulated"
            row["generation"]["error"] = (
                "WriterError: LLM request returned HTTP 429: "
                '{"error":{"message":"Rate limit reached ... on tokens per '
                'day (TPD)"}}'
            )
            return row
        row = _mk_row(question.id, variant, rate=0.1)
        row["generation"]["raw_text"] = "A sentence that is long enough here."
        row["claims"] = [{"claim": "c", "claim_index": 0,
                          "classification": "supported",
                          "justification": "j", "evidence_quote": "q"}]
        row["claim_counts"] = {"supported": 1, "unsupported": 0,
                               "contradicted": 0, "unverifiable": 0}
        return row

    monkeypatch.setattr(hb, "run_question_variant", fake_cell)
    result = hb.run_hallucination_benchmark(
        variants=["baseline", "grounded", "agentdesk"],
        execution_mode="live", limit=3, resume_path=partial,
    )
    # TPD hit on q01/grounded -> run stops immediately (no q01/agentdesk etc.).
    assert result["interrupted"] == "tokens_per_day_exhausted"
    assert calls == [("q01", "baseline"), ("q01", "grounded")]
    # Everything completed so far is persisted, including the TPD-failed row.
    file_rows = [json.loads(l) for l in
                 partial.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(file_rows) == 2
    # Resuming from that file afterwards loads only the successful row.
    rows, done = hb._load_resume_rows(partial)
    assert len(rows) == 1 and done == {("q01", "baseline", 1)}


def test_tpm_failure_continues_run_and_leaves_cell_for_resume(tmp_path, monkeypatch):
    partial = tmp_path / "partial.jsonl"
    calls = []

    def fake_cell(dataset, question, variant, run_index, **kwargs):
        calls.append((question.id, variant))
        if (question.id, variant) == ("q01", "grounded"):
            row = _mk_row(question.id, variant,
                          error="generation failed; metric left unpopulated")
            row["unsupported_claim_rate"] = None
            row["generation"]["error"] = (
                "StructuredOutputError: LLM request returned HTTP 429: "
                '{"error":{"message":"Rate limit reached ... on tokens per '
                'minute (TPM)"}}'
            )
            return row
        row = _mk_row(question.id, variant, rate=0.1)
        row["generation"]["raw_text"] = "A sentence that is long enough here."
        row["claims"] = [{"claim": "c", "claim_index": 0,
                          "classification": "supported",
                          "justification": "j", "evidence_quote": "q"}]
        row["claim_counts"] = {"supported": 1, "unsupported": 0,
                               "contradicted": 0, "unverifiable": 0}
        return row

    monkeypatch.setattr(hb, "run_question_variant", fake_cell)
    result = hb.run_hallucination_benchmark(
        variants=["baseline", "grounded", "agentdesk"],
        execution_mode="live", limit=2, resume_path=partial,
    )
    # TPM is cell-scoped: the run continues through every remaining cell.
    assert result["interrupted"] is None
    assert calls == [
        ("q01", "baseline"), ("q01", "grounded"), ("q01", "agentdesk"),
        ("q02", "baseline"), ("q02", "grounded"), ("q02", "agentdesk"),
    ]
    failed = [r for r in result["per_question"]
              if (r["question_id"], r["variant"]) == ("q01", "grounded")]
    assert len(failed) == 1
    assert failed[0]["unsupported_claim_rate"] is None
    assert hb._is_tpm_error(failed[0]) and not hb._is_tpd_error(failed[0])
    # The failed row is persisted; only the 5 successful cells count as done,
    # so a later --resume re-runs exactly the TPM-failed cell.
    file_rows = [json.loads(l) for l in
                 partial.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(file_rows) == 6
    _, done = hb._load_resume_rows(partial)
    assert done == {
        ("q01", "baseline", 1), ("q01", "agentdesk", 1),
        ("q02", "baseline", 1), ("q02", "grounded", 1), ("q02", "agentdesk", 1),
    }


def test_rate_limit_row_classification_distinguishes_tpm_from_tpd():
    tpd_row = {"error": "generation failed", "generation":
               {"error": "HTTP 429: ... on tokens per day (TPD)"}}
    tpm_row = {"error": "generation failed", "generation":
               {"error": "HTTP 429: ... on tokens per minute (TPM)"}}
    other = {"error": "ConnectError: boom", "generation": {"error": None}}
    ok = {"error": None, "generation": {"error": None}}
    assert hb._is_tpd_error(tpd_row) and not hb._is_tpm_error(tpd_row)
    assert hb._is_tpm_error(tpm_row) and not hb._is_tpd_error(tpm_row)
    assert not hb._is_tpd_error(other) and not hb._is_tpm_error(other)
    assert not hb._is_tpd_error(ok) and not hb._is_tpm_error(ok)


def test_run_cli_notes_tpm_failed_cells(tmp_path, monkeypatch, capsys):
    def fake_cell(dataset, question, variant, run_index, **kwargs):
        if (question.id, variant) == ("q01", "baseline"):
            row = _mk_row(question.id, variant,
                          error="generation failed; metric left unpopulated")
            row["generation"]["error"] = (
                "StructuredOutputError: LLM request returned HTTP 429: "
                '... on tokens per minute (TPM)"'
            )
            return row
        return _mk_row(question.id, variant, rate=0.0)

    monkeypatch.setattr(hb, "run_question_variant", fake_cell)
    args = _args(run_all=True, mock=False, no_save=True, limit=1)
    exit_code = hb.run_cli(args)
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "per-minute rate limit" in out
    assert "--resume" in out


def test_resume_with_missing_file_behaves_like_fresh_run(tmp_path, monkeypatch):
    executed = []

    def fake_cell(dataset, question, variant, run_index, **kwargs):
        executed.append((question.id, variant))
        row = _mk_row(question.id, variant, rate=0.0)
        row["generation"]["raw_text"] = "Another sentence long enough to keep."
        row["claims"] = []
        row["claim_counts"] = {"supported": 0, "unsupported": 0,
                               "contradicted": 0, "unverifiable": 0}
        return row

    monkeypatch.setattr(hb, "run_question_variant", fake_cell)
    result = hb.run_hallucination_benchmark(
        variants=["baseline"], execution_mode="live", limit=2,
        resume_path=tmp_path / "does-not-exist.jsonl",
    )
    assert len(executed) == 2
    assert result["interrupted"] is None


def test_resumed_rows_are_included_in_aggregates(tmp_path, monkeypatch):
    partial = tmp_path / "partial.jsonl"
    prior = _mk_row("q01", "baseline", rate=0.5)
    prior["claim_counts"] = {"supported": 1, "unsupported": 1,
                             "contradicted": 0, "unverifiable": 0}
    _write_jsonl(partial, [prior])

    def fake_cell(dataset, question, variant, run_index, **kwargs):
        row = _mk_row(question.id, variant, rate=0.0)
        row["generation"]["raw_text"] = "Long enough sentence for the filter."
        row["claims"] = []
        row["claim_counts"] = {"supported": 2, "unsupported": 0,
                               "contradicted": 0, "unverifiable": 0}
        return row

    monkeypatch.setattr(hb, "run_question_variant", fake_cell)
    result = hb.run_hallucination_benchmark(
        variants=["baseline"], execution_mode="live", limit=2,
        resume_path=partial,
    )
    summary = result["summary"]["variants"]["baseline"]
    assert summary["total_claims"] == 4      # resumed row: 2 claims + new row: 2
    assert summary["unsupported_claims"] == 1
    assert summary["unsupported_claim_rate"] == pytest.approx(0.25)
    assert summary["questions_completed"] == 2


# ---------------------------------------------------------------------------
# Backward compatibility of the legacy harness + CLI routing
# ---------------------------------------------------------------------------

def test_legacy_benchmark_still_importable_and_functional(monkeypatch):
    from agents.schemas import (
        CritiqueResult,
        EvidenceBundle,
        EvidenceSource,
        Report,
        ResearchPlan,
        ResearchRunResult,
        RetrievalSummary,
        RunTiming,
    )
    source = EvidenceSource(source_type="web", title="S1",
                            url="https://example.com/1", content="E. " * 5,
                            score=0.9, source_id="https://example.com/1")

    def fake_pipeline(query, event_callback=None, retrieval_mode="both"):
        return ResearchRunResult(
            query=query,
            plan=ResearchPlan(original_query=query, sub_questions=["a?", "b?", "c?"]),
            evidence=EvidenceBundle(
                original_query=query, sources=[source],
                retrieval_summary=RetrievalSummary(
                    total_sources=1, web_sources=1, kb_sources=0,
                    deduplicated_sources=0, kb_status="empty", web_status="ok"),
            ),
            report=Report(title="R", markdown="# R\nA cited claim appears here [1].\n\n## References\n[1] S1",
                          references=[]),
            critique=CritiqueResult(status="approved", iteration=0),
            iteration_count=0, final_status="approved",
            retrieval_mode=retrieval_mode, timing=RunTiming(total_seconds=0.5),
        )

    monkeypatch.setattr(benchmark_module, "run_research_pipeline", fake_pipeline)
    records = benchmark_module.run_benchmark(queries=["legacy query"])
    assert records[0]["final_status"] == "approved"
    assert benchmark_module.summarize(records)["completed"] == 1


def test_legacy_cli_flags_route_to_legacy_path(monkeypatch, capsys):
    # No ablation flags -> main() must run the LEGACY harness path.
    monkeypatch.setattr("sys.argv", ["prog", "--no-save"])
    monkeypatch.setattr(
        benchmark_module, "run_benchmark",
        lambda queries=None, retrieval_mode="both": [
            {"query": "x", "final_status": "approved", "iteration_count": 0,
             "critic_status": "approved", "citation_coverage": 1.0,
             "valid_citation_rate": 1.0, "plan_coverage": 1.0,
             "grounding_score": 1.0, "total_sources": 1, "web_sources": 1,
             "kb_sources": 0, "total_seconds": 1.0, "report_title": "R",
             "error": None, "retrieval_mode": "both"}],
    )
    called = {"hallucination": False}
    monkeypatch.setattr(hb, "run_cli",
                        lambda a: called.__setitem__("hallucination", True) or 0)
    exit_code = benchmark_module.main()
    assert exit_code == 0
    assert not called["hallucination"]
    out = capsys.readouterr().out
    assert '"summary"' in out and '"completed"' in out


def test_ablation_cli_flags_route_to_hallucination(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--variant", "baseline", "--mock",
                                     "--no-save"])
    called = {"hallucination": False}
    monkeypatch.setattr(hb, "run_cli",
                        lambda a: called.__setitem__("hallucination", True) or 0)
    exit_code = benchmark_module.main()
    assert exit_code == 0
    assert called["hallucination"]
