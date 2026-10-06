"""Tavily live web search tool."""

from __future__ import annotations

import logging

from agents.schemas import EvidenceSource
from config import get_config

logger = logging.getLogger(__name__)


class TavilyUnavailableError(Exception):
    """Tavily is not configured or the search failed."""


def _get_client():
    from tavily import TavilyClient

    cfg = get_config()
    if not cfg.tavily_api_key:
        raise TavilyUnavailableError("Web search is not configured")
    return TavilyClient(api_key=cfg.tavily_api_key)


def search_tavily(query: str, limit: int = 5) -> list[EvidenceSource]:
    """Run a research-depth web search and normalize results.

    Raises TavilyUnavailableError when unconfigured or on API failure.
    """
    if not query.strip():
        return []
    try:
        client = _get_client()
        response = client.search(
            query=query.strip(),
            search_depth="advanced",
            max_results=limit,
            include_answer=False,
            include_raw_content="markdown",
        )
    except TavilyUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize HTTP/SDK errors
        raise TavilyUnavailableError(
            f"Web search failed: {type(exc).__name__}"
        ) from exc

    sources: list[EvidenceSource] = []
    for result in response.get("results", []):
        raw = result.get("raw_content") or ""
        content = raw or result.get("content") or ""
        score = result.get("score")
        sources.append(
            EvidenceSource(
                source_type="web",
                title=result.get("title", ""),
                url=result.get("url"),
                document_name=None,
                content=str(content),
                score=float(score) if score is not None else None,
                source_id=result.get("url"),
            )
        )
    logger.info("tavily search returned %d sources", len(sources))
    return sources
