"""Parallel retrieval orchestration: Qdrant KB + Tavily web -> EvidenceBundle."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from agents.schemas import (
    EvidenceBundle,
    EvidenceSource,
    RetrievalSummary,
)
from tools.qdrant_tool import QdrantUnavailableError, search_qdrant
from tools.tavily_tool import TavilyUnavailableError, search_tavily


class RetrievalError(Exception):
    """Both retrieval sources failed; no evidence could be gathered."""


def normalize_url(url: str | None) -> str | None:
    """Normalize a URL for deterministic duplicate detection."""
    if not url:
        return None
    parsed = urlparse(url.strip().lower())
    if not parsed.netloc:
        return None
    host = parsed.netloc
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/")
    suffix = f"?{parsed.query}" if parsed.query else ""
    return f"{host}{path}{suffix}"


def _score_key(source: EvidenceSource) -> float:
    return source.score if source.score is not None else 0.0


def kb_dedup_key(source: EvidenceSource) -> str:
    identity = source.source_id or source.document_name or ""
    content_hash = hashlib.sha1(source.content.encode("utf-8")).hexdigest()
    return f"{identity}:{content_hash}"


def dedup_key(source: EvidenceSource) -> str | None:
    if source.source_type == "web":
        normalized = normalize_url(source.url)
        return f"web:{normalized}" if normalized else None
    return f"kb:{kb_dedup_key(source)}"


def deduplicate(sources: list[EvidenceSource]) -> tuple[list[EvidenceSource], int]:
    """Deterministic dedup keeping the best-scoring version of each duplicate.

    Ordering is stable: web sources before KB sources, then by descending
    native score, then by dedup key. Native scores are preserved as-is; a
    missing score sorts last within its type.
    """
    best: dict[str, EvidenceSource] = {}
    fallback_counter = 0
    duplicates_removed = 0

    for source in sources:
        key = dedup_key(source)
        if key is None:
            key = f"web:none:{fallback_counter}"
            fallback_counter += 1
        current = best.get(key)
        if current is None:
            best[key] = source
        elif ((_score_key(source), len(source.content)) >
              (_score_key(current), len(current.content))):
            best[key] = source
            duplicates_removed += 1
        else:
            duplicates_removed += 1

    ordered = sorted(
        best.items(),
        key=lambda item: (
            0 if item[1].source_type == "web" else 1,
            -_score_key(item[1]),
            item[0],
        ),
    )
    return [source for _, source in ordered], duplicates_removed


def retrieve_evidence(
    queries: list[str],
    web_limit: int = 5,
    kb_limit: int = 5,
    original_query: str = "",
    mode: str = "both",
) -> EvidenceBundle:
    """Retrieve evidence for each query from the sources the mode permits.

    Modes:
        "both" — Qdrant KB + Tavily web in parallel (default).
        "web"  — Tavily only; the KB side reports status "skipped".
        "kb"   — Qdrant only; the web side reports status "skipped".

    Web and KB failures are independent. Raises RetrievalError when no
    evidence can be gathered at all:
        "both" — only when BOTH sources hard-failed;
        "web"/"kb" — as soon as the single selected source hard-fails
                     (a controlled, clearly-attributed failure).
    """
    cleaned_queries = [q.strip() for q in queries if q and q.strip()]
    if not cleaned_queries:
        raise ValueError("queries must contain at least one non-empty string")
    if mode not in ("both", "web", "kb"):
        raise ValueError(f"invalid retrieval mode: {mode!r}")

    kb_errors: list[Exception] = []
    web_errors: list[Exception] = []

    def run_kb(query: str) -> list[EvidenceSource]:
        try:
            return search_qdrant(query, limit=kb_limit)
        except QdrantUnavailableError as exc:
            kb_errors.append(exc)
            return []

    def run_web(query: str) -> list[EvidenceSource]:
        try:
            return search_tavily(query, limit=web_limit)
        except TavilyUnavailableError as exc:
            web_errors.append(exc)
            return []

    use_kb = mode in ("both", "kb")
    use_web = mode in ("both", "web")
    worker_count = len(cleaned_queries) * (2 if mode == "both" else 1)
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {}
        for query in cleaned_queries:
            if use_kb:
                futures[pool.submit(run_kb, query)] = "kb"
            if use_web:
                futures[pool.submit(run_web, query)] = "web"
        collected: list[EvidenceSource] = []
        for future in futures:
            collected.extend(future.result())

    sources, duplicates_removed = deduplicate(collected)

    kb_has_results = any(s.source_type == "kb" for s in sources)
    web_has_results = any(s.source_type == "web" for s in sources)

    # Distinguish: available-with-results / available-but-empty /
    # unavailable / not-queried-because-of-mode.
    kb_status = (
        "ok" if kb_has_results
        else "unavailable" if kb_errors
        else "empty"
    ) if use_kb else "skipped"
    web_status = (
        "ok" if web_has_results or not web_errors
        else "unavailable"
    ) if use_web else "skipped"

    # Explicit, mode-attributed failure when the selected source(s) all
    # hard-failed. Single-source modes fail loudly instead of silently
    # returning an empty bundle.
    if not sources:
        if mode == "web" and web_errors:
            raise RetrievalError(
                "Web search is unavailable; no evidence could be gathered "
                "(retrieval mode: web)"
            )
        if mode == "kb" and kb_errors:
            raise RetrievalError(
                "Knowledge base is unavailable; no evidence could be "
                "gathered (retrieval mode: kb)"
            )
        if kb_errors and web_errors:
            raise RetrievalError(
                "Retrieval failed: no evidence could be gathered "
                "(KB unavailable, web unavailable)"
            )

    summary = RetrievalSummary(
        total_sources=len(sources),
        web_sources=sum(1 for s in sources if s.source_type == "web"),
        kb_sources=sum(1 for s in sources if s.source_type == "kb"),
        deduplicated_sources=duplicates_removed,
        kb_status=kb_status,
        web_status=web_status,
    )
    return EvidenceBundle(
        original_query=original_query or cleaned_queries[0],
        sources=sources,
        retrieval_summary=summary,
    )
