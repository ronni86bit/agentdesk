"""Qdrant retrieval tool backed by a local sentence-transformers embedding model."""

from __future__ import annotations

import logging
from functools import lru_cache

from qdrant_client import QdrantClient, models

from agents.schemas import EvidenceSource
from config import get_config

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "agentdesk_knowledge"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class QdrantUnavailableError(Exception):
    """Qdrant could not be reached or queried."""


@lru_cache
def _get_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")


def _embed(text: str) -> list[float]:
    return _get_embedding_model().encode(text).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with the shared local embedding model."""
    if not texts:
        return []
    return _get_embedding_model().encode(texts).tolist()


def delete_points_by_filter(collection: str, field: str, value: str) -> bool:
    """Delete all points whose payload `field` equals `value`.

    Qdrant does not report affected-point counts for deletes; the return
    value reflects whether the delete operation was acknowledged. Raises
    QdrantUnavailableError on transport failure.
    """
    try:
        client = _get_client()
        result = client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=field,
                            match=models.MatchValue(value=value),
                        )
                    ]
                )
            ),
        )
    except Exception as exc:  # noqa: BLE001 - normalize connection/transport errors
        raise QdrantUnavailableError(
            f"Qdrant delete failed: {type(exc).__name__}"
        ) from exc
    return result is not None


def count_points(collection: str) -> int:
    """Count all points in a collection. Raises QdrantUnavailableError."""
    try:
        client = _get_client()
        return client.count(collection_name=collection, exact=True).count
    except Exception as exc:  # noqa: BLE001
        raise QdrantUnavailableError(
            f"Qdrant count failed: {type(exc).__name__}"
        ) from exc


def _get_client() -> QdrantClient:
    return QdrantClient(url=get_config().qdrant_url, timeout=10)


def ensure_collection(client: QdrantClient, collection: str) -> None:
    """Create the collection with correct vector config if it does not exist.

    Never deletes or modifies an existing collection.
    """
    if client.collection_exists(collection):
        return
    dim = len(_embed("dimension probe"))
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=dim,
            distance=models.Distance.COSINE,
        ),
    )


def search_qdrant(
    query: str,
    limit: int = 5,
    collection: str = DEFAULT_COLLECTION,
) -> list[EvidenceSource]:
    """Embed the query and search Qdrant. Raises QdrantUnavailableError on failure."""
    if not query.strip():
        return []
    try:
        client = _get_client()
        ensure_collection(client, collection)
        vector = _embed(query)
        response = client.query_points(
            collection_name=collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
    except Exception as exc:  # noqa: BLE001 - normalize connection/transport errors
        raise QdrantUnavailableError(
            f"Qdrant search failed: {type(exc).__name__}"
        ) from exc

    sources: list[EvidenceSource] = []
    for point in response.points:
        payload = point.payload or {}
        content = (
            payload.get("content")
            or payload.get("text")
            or ""
        )
        sources.append(
            EvidenceSource(
                source_type="kb",
                title=payload.get("title", ""),
                url=payload.get("url"),
                document_name=payload.get("document_name"),
                content=str(content),
                score=float(point.score) if point.score is not None else None,
                source_id=str(payload.get("source_id") or point.id),
            )
        )
    logger.info("qdrant search returned %d sources", len(sources))
    return sources
