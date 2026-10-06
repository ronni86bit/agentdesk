"""Deterministic citation handling for the Writer Agent.

Citation IDs are assigned by the backend (not the LLM), validated after
generation, and resolved to authoritative references. The LLM may only use
the numbers it was given.
"""

from __future__ import annotations

import re

from agents.schemas import EvidenceBundle, EvidenceSource, ReportReference

# Inline numeric citations: [1], [2][3] — but not markdown links like [1](url)
CITATION_PATTERN = re.compile(r"\[(\d{1,3})\](?!\()")

PROVISIONAL_MIN_CLAIM_WORDS = 6
_PROVISIONAL_TRANSITIONS = (
    "in summary", "overall", "therefore", "furthermore",
    "moreover", "however,", "additionally", "in addition",
    "this section", "the following", "as discussed",
)


def assign_citation_ids(evidence: EvidenceBundle) -> dict[int, EvidenceSource]:
    """Map deterministic 1-based citation IDs to bundle sources (bundle order)."""
    return {i: source for i, source in enumerate(evidence.sources, start=1)}


def format_sources_for_prompt(
    citation_map: dict[int, EvidenceSource],
    max_chars_per_source: int = 1000,
    max_total_chars: int = 12000,
) -> str:
    """Render numbered sources for the Writer prompt within a token budget.

    Content is truncated per source and overall so the prompt stays inside
    Groq free-tier limits (~8k TPM); IDs and metadata stay exact. The
    per-source allowance shrinks proportionally when many sources are present.
    """
    ids = sorted(citation_map)
    if not ids:
        return ""

    # Pass 1: build exact headers so their real size is reserved first.
    headers: dict[int, str] = {}
    for citation_id in ids:
        source = citation_map[citation_id]
        if source.source_type == "web":
            headers[citation_id] = (
                f"SOURCE [{citation_id}]\nType: web\n"
                f"Title: {source.title}\nURL: {source.url}"
            )
        else:
            headers[citation_id] = (
                f"SOURCE [{citation_id}]\nType: kb\n"
                f"Title: {source.title}\nDocument: {source.document_name}"
            )

    header_total = sum(len(headers[i]) + 2 for i in ids)  # +2 for separators
    content_budget_total = max(len(ids), max_total_chars - header_total)
    per_source = max(
        150, min(max_chars_per_source, content_budget_total // len(ids))
    )

    # Pass 2: fill content within the remaining budget, clamping as we go.
    blocks = []
    used = header_total
    for citation_id in ids:
        source = citation_map[citation_id]
        remaining_budget = content_budget_total - (used - header_total)
        limit = min(per_source, remaining_budget)
        if limit <= 0:
            content = ""
        else:
            full = source.content.strip()
            content = full[:limit]
            if len(full) > limit:
                content += "..."
        used += len(content)
        blocks.append(f"{headers[citation_id]}\nContent:\n{content}")
    return "\n\n".join(blocks)


def extract_citations(markdown: str) -> set[int]:
    """Extract all inline numeric citation IDs used in a report body."""
    return {int(match) for match in CITATION_PATTERN.findall(markdown)}


def validate_report_citations(
    report_markdown: str,
    available_source_ids: set[int],
) -> list[int]:
    """Return citation IDs present in the text but not in available_source_ids."""
    return sorted(extract_citations(report_markdown) - available_source_ids)


def build_references(
    markdown: str,
    citation_map: dict[int, EvidenceSource],
) -> list[ReportReference]:
    """Build authoritative references for exactly the citations used, in ID order.

    Citation numbers are preserved even if non-contiguous (e.g. [1], [3]).
    """
    used = sorted(extract_citations(markdown))
    references = []
    for citation_id in used:
        source = citation_map[citation_id]
        references.append(
            ReportReference(
                citation_id=citation_id,
                source_type=source.source_type,
                title=source.title,
                url=source.url,
                document_name=source.document_name,
                source_id=source.source_id,
            )
        )
    return references


# ---------------------------------------------------------------------------
# Provisional citation-coverage heuristic (to be upgraded by eval module)
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _is_claim_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    if len(lowered.split()) < PROVISIONAL_MIN_CLAIM_WORDS:
        return False
    if lowered.startswith(_PROVISIONAL_TRANSITIONS):
        return False
    # Headings and reference entries are not claim sentences; table rows and
    # list items ARE eligible (leading markers are stripped before matching).
    if sentence.startswith(("#", "[", ">")):
        return False
    return True


def _strip_leading_markers(sentence: str) -> str:
    return re.sub(r"^[\s\-\*\|\d\.]+", "", sentence)


def citation_coverage(report_markdown: str) -> float:
    """Provisional heuristic: fraction of claim sentences carrying >=1 valid
    inline citation. Deterministic; intended to be replaced/upgraded by the
    dedicated evaluation module later. Returns 1.0 when no claims are found.
    """
    citations_found = extract_citations(report_markdown)
    claims = [
        s for s in _split_sentences(report_markdown) if _is_claim_sentence(s)
    ]
    if not claims:
        return 1.0
    cited = sum(
        1
        for s in claims
        if extract_citations(_strip_leading_markers(s)) & citations_found
    )
    return cited / len(claims)
