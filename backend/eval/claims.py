"""Factual-claim extraction, classification, and unsupported-claim-rate math.

This module implements the measurement layer of the hallucination benchmark
(`eval.hallucination`). It is deliberately separate from `eval.scorer`:

- `eval.scorer` computes MECHANICAL grounding signals (citation coverage,
  valid-citation rate) from report structure alone. A high citation coverage
  does NOT mean claims are true — a report can cite a source for a claim the
  source does not make.
- This module supports the Unsupported Claim Rate metric: factual claims are
  extracted from a generated answer and each is CLASSIFIED against an
  authoritative evidence corpus by an LLM judge. A citation marker is never
  treated as evidence of support; only corpus entailment counts.

Claim extraction is an explicitly documented HEURISTIC (sentence splitting +
filters). Claim classification is an LLM-judge protocol, NOT ground truth;
every machine-readable result carries `evaluation_tier` metadata so automated
judge output can never be mistaken for manually verified measurements.

Classification labels (fixed vocabulary, enforced by strict schema):
    supported     - the evidence corpus directly entails the claim
    unsupported   - on-topic but the corpus does not support it
    contradicted  - the corpus directly contradicts the claim
    unverifiable  - the corpus cannot settle the claim either way

Primary metric:
    unsupported_or_contradicted = unsupported + contradicted
    unsupported_claim_rate = unsupported_or_contradicted / total_evaluated
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

ClaimLabel = Literal["supported", "unsupported", "contradicted", "unverifiable"]
CLASSIFICATION_LABELS: tuple[str, ...] = (
    "supported", "unsupported", "contradicted", "unverifiable",
)

# Primary-metric labels: treated as unsupported factual claims.
UNSUPPORTED_LABELS: frozenset[str] = frozenset({"unsupported", "contradicted"})

# ---------------------------------------------------------------------------
# Claim extraction heuristic (documented; deterministic)
# ---------------------------------------------------------------------------

# Same sentence splitting approach as agents.citations: sentence enders or
# newlines delimit candidates.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_MARKDOWN_NOISE = re.compile(r"!\[[^\]]*\]\([^)]*\)|\[[^\]]*\]\([^)]*\)|[#>*`_]+")
_CITATION_MARKERS = re.compile(r"\[(\d{1,3})\](?!\()")

# Sentences that talk about the answer itself instead of the world.
_SELF_REFERENTIAL = (
    "this report", "this answer", "this response", "the report", "the evidence",
    "the sources", "the study", "as an ai", "in summary", "in conclusion",
    "note that", "it is worth", "the following", "based on the provided",
    "based on the evidence", "per the", "according to the provided",
)

# Interrogative / hedged fragments are not checkable factual claims.
_MIN_CLAIM_WORDS = 6
_MAX_CLAIMS_PER_ANSWER = 30  # documented cap; keeps judge prompts bounded


def clean_claim_text(sentence: str) -> str:
    """Normalize one candidate sentence into claim text.

    Strips markdown decoration and bullet/list prefixes; citation markers
    like [1] are removed from the claim TEXT itself (a citation is metadata,
    not part of the factual assertion, and must never influence the judge).
    """
    text = _MARKDOWN_NOISE.sub(" ", sentence)
    text = _CITATION_MARKERS.sub("", text)
    text = re.sub(r"^[\s\-\*\|\d\.\)]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_factual_claim(candidate: str) -> bool:
    lowered = candidate.lower()
    if len(lowered.split()) < _MIN_CLAIM_WORDS:
        return False
    if lowered.startswith(_SELF_REFERENTIAL):
        return False
    return not lowered.endswith("?")


def extract_factual_claims(answer_text: str) -> list[str]:
    """Extract factual claims from a generated answer (documented heuristic).

    Deterministic: markdown is stripped, text is split into sentences, and
    self-referential/hedged/short fragments are dropped. At most
    `_MAX_CLAIMS_PER_ANSWER` claims are kept, in answer order.
    """
    claims: list[str] = []
    seen: set[str] = set()
    for raw in _SENTENCE_SPLIT.split(answer_text or ""):
        text = clean_claim_text(raw)
        if not text or not _is_factual_claim(text):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        claims.append(text)
        if len(claims) >= _MAX_CLAIMS_PER_ANSWER:
            break
    return claims


# ---------------------------------------------------------------------------
# Judge output model (parsed from strict structured output)
# ---------------------------------------------------------------------------

class ClaimVerdict(BaseModel):
    """One judge classification for one extracted claim.

    All fields are plain required-in-schema strings/ints with defaults only
    for Pydantic-side tolerance: strict structured-output schemas must not
    contain `default`, `anyOf`/null unions, or optional properties.
    """

    claim_index: int = Field(ge=0)
    classification: ClaimLabel
    justification: str = ""
    evidence_quote: str = ""


class ClaimJudgeOutput(BaseModel):
    """Strict structured output of one judge call over a claim batch."""

    verdicts: list[ClaimVerdict] = Field(default_factory=list)


class ClaimEvaluation(BaseModel):
    """Raw per-claim record preserved for audit (never aggregated away)."""

    claim: str
    claim_index: int
    classification: ClaimLabel
    justification: str = ""
    evidence_quote: str | None = None


# ---------------------------------------------------------------------------
# Aggregation + metric math (total functions; zero-division safe)
# ---------------------------------------------------------------------------

def classify_claims(verdicts: list[ClaimVerdict]) -> dict[str, int]:
    """Count verdicts per label. Unknown labels cannot occur (schema-bound)."""
    counts = {label: 0 for label in CLASSIFICATION_LABELS}
    for verdict in verdicts:
        counts[verdict.classification] += 1
    return counts


def unsupported_claim_rate(counts: dict[str, int]) -> float | None:
    """(unsupported + contradicted) / total evaluated claims.

    The denominator is ALL evaluated claims (supported, unsupported,
    contradicted, unverifiable) — every claim the judge evaluated counts.
    Returns None when no claims were evaluated (documented zero-division
    handling: the metric is reported as undefined, never as 0.0 or 1.0).
    """
    total = sum(counts.get(label, 0) for label in CLASSIFICATION_LABELS)
    if total == 0:
        return None
    bad = sum(counts.get(label, 0) for label in UNSUPPORTED_LABELS)
    return bad / total


def absolute_rate_reduction(baseline_rate: float | None, variant_rate: float | None) -> float | None:
    """Baseline rate minus variant rate (percentage points). None if either undefined."""
    if baseline_rate is None or variant_rate is None:
        return None
    return baseline_rate - variant_rate


def relative_rate_reduction(baseline_rate: float | None, variant_rate: float | None) -> float | None:
    """((baseline - variant) / baseline) * 100.

    Returns None (undefined, never invented) when either rate is undefined or
    when the baseline rate is exactly zero.
    """
    if baseline_rate is None or variant_rate is None or baseline_rate == 0:
        return None
    return ((baseline_rate - variant_rate) / baseline_rate) * 100.0
