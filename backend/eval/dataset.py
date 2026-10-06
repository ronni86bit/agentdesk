"""Benchmark dataset loading and validation for the hallucination benchmark.

The dataset (`eval/benchmark_dataset.json`) defines the fixed question set and
the authoritative reference sources each question should be judged against.

IMPORTANT — reference source policy:
- Reference URLs in the dataset are institution-level pointers (WHO, World
  Bank, IEA, ...) supplied as CURATION STARTING POINTS. They are NOT claimed
  to be the exact authoritative document for any question.
- Every question in the shipped dataset carries
  `source_curation: "manual_review_required"` until a human curator pins the
  exact authoritative documents (and optionally inlines their text as
  `content`) — per-question curation status is preserved in every result.
- A question whose references have no inline `content` requires LIVE FETCHING
  of its reference URLs at benchmark time; fetch failures mark the question's
  corpus unavailable and the question is EXCLUDED from the affected metric
  (never silently scored against nothing).

Schema (v1):
{
  "benchmark": "agentdesk-hallucination",
  "version": "1.0.0",
  "description": "...",
  "questions": [
    {
      "id": "q01",
      "question": "...",
      "domain": "...",
      "source_curation": "manual_review_required" | "curated",
      "reference_sources": [
        {"title": "...", "url": "...", "notes": "...",
         "content": "... (optional inlined authoritative text)"}
      ]
    }
  ]
}
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

DATASET_PATH = Path(__file__).resolve().parent / "benchmark_dataset.json"

BENCHMARK_NAME = "agentdesk-hallucination"
BENCHMARK_VERSION = "1.0.0"


class ReferenceSource(BaseModel):
    """One authoritative reference pointer (curation starting point)."""

    title: str
    url: str | None = None
    notes: str = ""
    content: str | None = None  # inlined authoritative text (manual curation)


class BenchmarkQuestion(BaseModel):
    """One benchmark question with its authoritative reference set."""

    id: str
    question: str
    domain: str = ""
    source_curation: str = "manual_review_required"
    reference_sources: list[ReferenceSource] = Field(default_factory=list)

    @field_validator("id", "question")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @property
    def needs_manual_curation(self) -> bool:
        return self.source_curation != "curated"

    @property
    def has_inlined_content(self) -> bool:
        return any(
            (src.content or "").strip() for src in self.reference_sources
        )


class BenchmarkDataset(BaseModel):
    """Validated dataset container."""

    benchmark: str = BENCHMARK_NAME
    version: str = BENCHMARK_VERSION
    description: str = ""
    questions: list[BenchmarkQuestion] = Field(min_length=1)

    @field_validator("questions")
    @classmethod
    def unique_ids(cls, value: list[BenchmarkQuestion]) -> list[BenchmarkQuestion]:
        ids = [q.id for q in value]
        if len(ids) != len(set(ids)):
            raise ValueError("question ids must be unique")
        return value


def load_dataset(path: Path | str | None = None) -> BenchmarkDataset:
    """Load and validate the benchmark dataset JSON."""
    dataset_path = Path(path) if path is not None else DATASET_PATH
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    return BenchmarkDataset.model_validate(data)


def dataset_sha256(path: Path | str | None = None) -> str:
    """Content hash of the dataset file — recorded in every run for reproducibility."""
    dataset_path = Path(path) if path is not None else DATASET_PATH
    return hashlib.sha256(dataset_path.read_bytes()).hexdigest()


def run_timestamp() -> str:
    """ISO-8601 UTC timestamp for run records."""
    return datetime.now(timezone.utc).isoformat()
