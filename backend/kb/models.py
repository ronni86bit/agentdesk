"""Document metadata models for knowledge-base ingestion."""

from typing import Literal

from pydantic import BaseModel, Field


KBFileType = Literal["pdf", "txt", "md"]


class DocumentMetadata(BaseModel):
    """Stable identity + provenance for one ingested KB document.

    `document_id` is derived deterministically from the file content hash,
    so re-uploading identical content targets the same document.
    """

    document_id: str
    document_name: str
    file_type: KBFileType
    chunk_count: int = Field(ge=0)
    created_at: str
    content_hash: str
    size_bytes: int = Field(ge=0)
    status: Literal["indexed"] = "indexed"


class IngestResult(BaseModel):
    """Outcome of one upload.

    Duplicate policy (deterministic): same content hash -> the existing
    document's chunks are replaced atomically by the new embedding run and
    status is "replaced". A different file with the same NAME is a separate
    document (identity is content-based, not name-based).
    """

    status: Literal["created", "replaced"]
    document: DocumentMetadata


class KBDocumentList(BaseModel):
    documents: list[DocumentMetadata]
    total: int = Field(ge=0)


class KBStatus(BaseModel):
    """Knowledge-base availability snapshot (no credentials)."""

    available: bool
    collection: str
    documents: int = 0
    chunks: int = 0
    error: str | None = None


class KBDeleteResult(BaseModel):
    status: Literal["deleted"]
    document_id: str
    chunks_deleted: int = Field(ge=0)
