"""Knowledge-base ingestion: PDF/text -> normalize -> chunk -> embed -> Qdrant.

Pipeline (all deterministic except the embedding model itself):

    file bytes
      -> extract_text        (pypdf for .pdf, UTF-8 decode for .txt/.md)
      -> normalize           (whitespace collapsing, blank-line preservation)
      -> chunk_text          (documented policy below)
      -> embed_texts         (shared all-MiniLM-L6-v2, same model as retrieval)
      -> Qdrant upsert       (deterministic UUID point IDs)

Chunking policy (deliberately simple, documented, deterministic):
    Target ~1200 characters per chunk with ~200 characters of overlap.
    Text is split into paragraphs; paragraphs are accumulated until the
    target is reached; an oversized paragraph is hard-split at sentence
    boundaries, falling back to a raw character cut. Each chunk carries a
    ~200-character tail of the previous chunk as context overlap. 1200
    characters is roughly the all-MiniLM-L6-v2 context window (256 tokens),
    keeping chunk embeddings inside a single tokenization window.

Duplicate policy:
    Identity is the SHA-256 content hash. Re-uploading identical content
    REPLACES the existing document's chunks (delete + re-ingest) and
    reports status "replaced". Same filename with different content is a
    different document. No unlimited duplicate chunk growth is possible.

Point payload schema (sufficient for retrieval + UI without extra queries):
    document_id, document_name, title, content, text (duplicate for the
    legacy reader), chunk_index, total_chunks, source_type="kb",
    source_id, content_hash, file_type, created_at

`source_id` is f"{document_id}:{chunk_index}" — stable across re-ingestion
so citation deduplication in tools/retrieval.py stays consistent.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

from qdrant_client import models

from agents.schemas import EvidenceSource
from kb.models import (
    DocumentMetadata,
    IngestResult,
    KBDeleteResult,
    KBDocumentList,
    KBStatus,
)
from tools.qdrant_tool import (
    DEFAULT_COLLECTION,
    QdrantUnavailableError,
    count_points,
    delete_points_by_filter,
    embed_texts,
    ensure_collection,
    _get_client,
)

ALLOWED_EXTENSIONS = {".pdf": "pdf", ".txt": "txt", ".md": "md"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
CHUNK_TARGET_CHARS = 1200
CHUNK_OVERLAP_CHARS = 200


class KBIngestError(Exception):
    """User-safe ingestion failure (bad file, empty content, oversized)."""


# ---------------------------------------------------------------------------
# Extraction + normalization
# ---------------------------------------------------------------------------

def file_type_for(filename: str) -> str:
    dot = filename.rfind(".")
    ext = filename[dot:].lower() if dot != -1 else ""
    if ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise KBIngestError(
            f"Unsupported file type. Allowed: {allowed}"
        )
    return ALLOWED_EXTENSIONS[ext]


def extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from PDF/TXT/MD bytes. Raises KBIngestError."""
    file_type = file_type_for(filename)
    if not data:
        raise KBIngestError("Uploaded file is empty.")
    if file_type == "pdf":
        return _extract_pdf(data)
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise KBIngestError(
            "Text file is not valid UTF-8. Please re-save it as UTF-8."
        )


def _extract_pdf(data: bytes) -> str:
    from io import BytesIO

    from pypdf import PdfReader

    try:
        reader = PdfReader(BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 - normalize parser errors
        raise KBIngestError(
            "PDF could not be read. It may be corrupted or password-protected."
        ) from exc
    text = "\n\n".join(p for p in pages if p.strip())
    if not text.strip():
        raise KBIngestError(
            "No extractable text found in the PDF (it may be a scanned image)."
        )
    return text


def normalize_text(text: str) -> str:
    """Normalize line endings and collapse trailing whitespace per line.

    Paragraph breaks (blank lines) are preserved for the chunker.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    target_chars: int = CHUNK_TARGET_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Deterministic paragraph-accumulating chunker (policy in module doc)."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    # Explode oversized paragraphs into sentence-sized pieces.
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= target_chars:
            pieces.append(paragraph)
            continue
        for sentence in _split_sentences(paragraph):
            if len(sentence) <= target_chars:
                pieces.append(sentence)
            else:
                # Pathological unsplittable text: hard character cut.
                for i in range(0, len(sentence), target_chars):
                    pieces.append(sentence[i : i + target_chars])

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for piece in pieces:
        extra = len(piece) + (1 if current else 0)
        if current and current_len + extra > target_chars:
            chunks.append("\n".join(current))
            # Carry a trailing-overlap tail into the next chunk for context.
            tail = "\n".join(current)[-overlap_chars:].strip()
            current = [tail] if tail else []
            current_len = len(tail)
        current.append(piece)
        current_len += len(piece) + (1 if len(current) > 1 else 0)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _split_sentences(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def document_id_for(hash_hex: str) -> str:
    """Short stable document identifier derived from the content hash."""
    return hash_hex[:16]


def point_id_for(document_id: str, chunk_index: int) -> str:
    """Deterministic UUID point ID (Qdrant requires UUID/int IDs)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"agentdesk:{document_id}:{chunk_index}"))


# ---------------------------------------------------------------------------
# Ingest / list / delete / status
# ---------------------------------------------------------------------------

def _payload_payload(chunk: str, meta: DocumentMetadata, index: int) -> dict:
    return {
        "document_id": meta.document_id,
        "document_name": meta.document_name,
        "title": meta.document_name,
        "content": chunk,
        # "text" duplicates content for the legacy qdrant reader fallback.
        "text": chunk,
        "chunk_index": index,
        "total_chunks": meta.chunk_count,
        "source_type": "kb",
        "source_id": f"{meta.document_id}:{index}",
        "content_hash": meta.content_hash,
        "file_type": meta.file_type,
        "created_at": meta.created_at,
    }


def document_exists(document_id: str, collection: str = DEFAULT_COLLECTION) -> bool:
    try:
        client = _get_client()
        ensure_collection(client, collection)
        points, _ = client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
        )
    except QdrantUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise QdrantUnavailableError(
            f"Qdrant lookup failed: {type(exc).__name__}"
        ) from exc
    return len(points) > 0


def ingest_document(
    filename: str,
    data: bytes,
    collection: str = DEFAULT_COLLECTION,
) -> IngestResult:
    """Ingest one uploaded document end-to-end. Raises KBIngestError for
    user-safe problems and QdrantUnavailableError for store failures."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise KBIngestError("File is too large (limit: 10 MB).")

    file_type = file_type_for(filename)
    text = normalize_text(extract_text(filename, data))
    if not text:
        raise KBIngestError("Document contains no extractable text.")
    chunks = chunk_text(text)
    if not chunks:
        raise KBIngestError("Document produced no indexable content.")

    hash_hex = content_hash(data)
    document_id = document_id_for(hash_hex)
    replacing = document_exists(document_id, collection)

    meta = DocumentMetadata(
        document_id=document_id,
        document_name=filename,
        file_type=file_type,  # type: ignore[arg-type]
        chunk_count=len(chunks),
        created_at=datetime.now(timezone.utc).isoformat(),
        content_hash=hash_hex,
        size_bytes=len(data),
    )

    vectors = embed_texts(chunks)
    if len(vectors) != len(chunks):
        raise KBIngestError("Embedding failed; please retry.")

    if replacing:
        delete_points_by_filter(collection, "document_id", document_id)

    try:
        client = _get_client()
        ensure_collection(client, collection)
        client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=point_id_for(document_id, i),
                    vector=vector,
                    payload=_payload_payload(chunk, meta, i),
                )
                for i, (chunk, vector) in enumerate(zip(chunks, vectors))
            ],
        )
    except QdrantUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise QdrantUnavailableError(
            f"Qdrant upsert failed: {type(exc).__name__}"
        ) from exc

    return IngestResult(
        status="replaced" if replacing else "created",
        document=meta,
    )


def _scroll_all(collection: str):
    client = _get_client()
    ensure_collection(client, collection)
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
        )
        yield from points
        if offset is None:
            return


def list_documents(collection: str = DEFAULT_COLLECTION) -> KBDocumentList:
    """Aggregate stored chunks into unique documents, newest first."""
    grouped: dict[str, dict] = {}
    try:
        for point in _scroll_all(collection):
            payload = point.payload or {}
            document_id = payload.get("document_id")
            if not document_id:
                continue
            entry = grouped.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "document_name": payload.get("document_name", ""),
                    "file_type": payload.get("file_type", "txt"),
                    "chunk_count": 0,
                    "created_at": payload.get("created_at", ""),
                    "content_hash": payload.get("content_hash", ""),
                    "size_bytes": 0,
                },
            )
            entry["chunk_count"] += 1
            created = payload.get("created_at", "")
            if created > entry["created_at"]:
                entry["created_at"] = created
    except QdrantUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise QdrantUnavailableError(
            f"Qdrant listing failed: {type(exc).__name__}"
        ) from exc

    documents = sorted(
        (DocumentMetadata(**entry) for entry in grouped.values()),
        key=lambda d: (d.created_at, d.document_id),
        reverse=True,
    )
    return KBDocumentList(documents=documents, total=len(documents))


def delete_document(document_id: str, collection: str = DEFAULT_COLLECTION) -> KBDeleteResult:
    """Delete every chunk of one document. Raises KeyError-equivalent
    (ValueError) when the document does not exist."""
    chunk_count = _chunk_count_for(document_id, collection)
    if chunk_count == 0:
        raise ValueError(f"Document {document_id} not found.")
    delete_points_by_filter(collection, "document_id", document_id)
    return KBDeleteResult(
        status="deleted",
        document_id=document_id,
        chunks_deleted=chunk_count,
    )


def _chunk_count_for(document_id: str, collection: str) -> int:
    try:
        client = _get_client()
        ensure_collection(client, collection)
        points, _ = client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
            limit=10000,
            with_payload=False,
        )
    except QdrantUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise QdrantUnavailableError(
            f"Qdrant lookup failed: {type(exc).__name__}"
        ) from exc
    return len(points)


def kb_status(collection: str = DEFAULT_COLLECTION) -> KBStatus:
    """Availability snapshot; never raises for an unreachable store."""
    try:
        client = _get_client()
        ensure_collection(client, collection)
        chunks = count_points(collection)
    except QdrantUnavailableError as exc:
        return KBStatus(
            available=False,
            collection=collection,
            error=str(exc),
        )
    seen: set[str] = set()
    try:
        for point in _scroll_all(collection):
            payload = point.payload or {}
            document_id = payload.get("document_id")
            if document_id and document_id not in seen:
                seen.add(document_id)
    except QdrantUnavailableError:
        pass  # listing is best-effort for the status endpoint
    return KBStatus(
        available=True,
        collection=collection,
        documents=len(seen),
        chunks=chunks,
    )


# ---------------------------------------------------------------------------
# Retrieval-side helper: convert stored payload back to EvidenceSource
# ---------------------------------------------------------------------------

def payload_to_source(payload: dict, score: float | None = None) -> EvidenceSource:
    """Build an EvidenceSource from a stored KB chunk payload."""
    return EvidenceSource(
        source_type="kb",
        title=payload.get("title", ""),
        url=None,
        document_name=payload.get("document_name"),
        content=payload.get("content", ""),
        score=score,
        source_id=payload.get("source_id"),
    )
