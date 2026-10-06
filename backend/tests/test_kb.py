"""Tests for knowledge-base ingestion. Qdrant and the embedding model are
mocked; PDF extraction is exercised for real through pypdf+fpdf2."""

import types

import pytest
from fastapi.testclient import TestClient

import tools.qdrant_tool as qdrant_tool
from kb import ingest as kb_ingest
from main import app


# ---------------------------------------------------------------------------
# Fake Qdrant infrastructure
# ---------------------------------------------------------------------------

class FakeQdrantClient:
    def __init__(self):
        self.points = {}  # id -> payload
        self.upserts = []
        self.deleted_filters = []
        self._exists = False

    def collection_exists(self, name):
        return self._exists

    def create_collection(self, collection_name, vectors_config):
        self._exists = True

    def upsert(self, collection_name, points):
        self.upserts.append(list(points))
        for p in points:
            self.points[p.id] = p.payload

    def scroll(self, collection_name, scroll_filter=None, limit=256,
               offset=None, with_payload=True):
        must = scroll_filter.must if scroll_filter is not None else None
        results = []
        for pid, payload in self.points.items():
            if must:
                cond = must[0]
                if payload.get(cond.key) != cond.match.value:
                    continue
            results.append(types.SimpleNamespace(
                id=pid,
                payload=payload if with_payload else None,
            ))
        start = offset if isinstance(offset, int) else 0
        return results[start:start + limit], None

    def delete(self, collection_name, points_selector):
        cond = points_selector.filter.must[0]
        self.deleted_filters.append((cond.key, cond.match.value))
        self.points = {
            pid: pl for pid, pl in self.points.items()
            if pl.get(cond.key) != cond.match.value
        }
        return types.SimpleNamespace(operation_id=1)

    def count(self, collection_name, exact=True):
        return types.SimpleNamespace(count=len(self.points))


@pytest.fixture
def fake_kb(monkeypatch):
    client = FakeQdrantClient()
    monkeypatch.setattr(kb_ingest, "_get_client", lambda: client)
    monkeypatch.setattr(
        kb_ingest, "embed_texts",
        lambda texts: [[0.1] * 384 for _ in texts],
    )
    monkeypatch.setattr(
        kb_ingest, "delete_points_by_filter",
        lambda collection, field, value: client.delete(
            collection,
            points_selector=types.SimpleNamespace(
                filter=types.SimpleNamespace(
                    must=[types.SimpleNamespace(key=field, match=types.SimpleNamespace(value=value))]
                )
            ),
        ),
    )
    monkeypatch.setattr(
        kb_ingest, "count_points",
        lambda collection: client.count(collection).count,
    )
    # ensure_collection's dimension probe lives in qdrant_tool globals.
    monkeypatch.setattr(qdrant_tool, "_embed", lambda text: [0.2] * 384)
    return client


# ---------------------------------------------------------------------------
# Validation and text extraction
# ---------------------------------------------------------------------------

def test_allowed_extensions():
    assert kb_ingest.file_type_for("a.pdf") == "pdf"
    assert kb_ingest.file_type_for("b.TXT") == "txt"
    assert kb_ingest.file_type_for("c.Md") == "md"


def test_unsupported_extension_rejected():
    with pytest.raises(kb_ingest.KBIngestError, match="Unsupported"):
        kb_ingest.file_type_for("virus.exe")
    with pytest.raises(kb_ingest.KBIngestError):
        kb_ingest.file_type_for("noextension")


def test_empty_file_rejected():
    with pytest.raises(kb_ingest.KBIngestError, match="empty"):
        kb_ingest.extract_text("a.txt", b"")


def test_txt_extraction():
    assert kb_ingest.extract_text("a.txt", "hello world".encode()) == "hello world"


def test_md_extraction_with_bom():
    data = b"\xef\xbb\xbf# Title\n\nBody"
    assert kb_ingest.extract_text("a.md", data).startswith("# Title")


def test_invalid_utf8_rejected_safely():
    with pytest.raises(kb_ingest.KBIngestError, match="UTF-8"):
        kb_ingest.extract_text("a.txt", b"\xff\xfe\xfa")


def test_pdf_extraction_real_roundtrip():
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    pdf.multi_cell(0, 8, "AgentDesk KB ingestion test document.")
    data = bytes(pdf.output())
    text = kb_ingest.extract_text("sample.pdf", data)
    assert "AgentDesk KB ingestion test document" in text


def test_normalize_text_preserves_paragraphs():
    text = "Line one  \r\n\r\nLine two\t\r\n"
    normalized = kb_ingest.normalize_text(text)
    assert normalized == "Line one\n\nLine two"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def test_chunking_small_text_single_chunk():
    chunks = kb_ingest.chunk_text("A short paragraph.\n\nAnother one.")
    assert len(chunks) == 1
    assert "Another one." in chunks[0]


def test_chunking_long_text_multiple_chunks_deterministic():
    paragraph = (
        "Artificial intelligence transforms healthcare diagnostics. " * 30
    )
    text = "\n\n".join(paragraph for _ in range(10))
    chunks_a = kb_ingest.chunk_text(text)
    chunks_b = kb_ingest.chunk_text(text)
    assert chunks_a == chunks_b  # deterministic
    assert len(chunks_a) > 1
    assert all(len(c) <= kb_ingest.CHUNK_TARGET_CHARS + kb_ingest.CHUNK_OVERLAP_CHARS + 50
               for c in chunks_a)


def test_chunking_overlap_carries_context():
    sentence = "The quick brown fox jumps over the lazy dog. "
    text = "\n\n".join(sentence * 40 for _ in range(3))
    chunks = kb_ingest.chunk_text(text)
    if len(chunks) > 1:
        # Second chunk starts with a tail of the first.
        assert chunks[1].startswith(chunks[0][-kb_ingest.CHUNK_OVERLAP_CHARS:].strip()[:20])


def test_chunking_empty_text():
    assert kb_ingest.chunk_text("") == []
    assert kb_ingest.chunk_text("\n\n  \n\n") == []


# ---------------------------------------------------------------------------
# Identity / duplicate policy
# ---------------------------------------------------------------------------

def test_document_id_deterministic_from_content():
    h1 = kb_ingest.content_hash(b"same bytes")
    h2 = kb_ingest.content_hash(b"same bytes")
    assert kb_ingest.document_id_for(h1) == kb_ingest.document_id_for(h2)
    assert kb_ingest.document_id_for(h1) != kb_ingest.document_id_for(kb_ingest.content_hash(b"other"))


def test_point_ids_deterministic():
    a = kb_ingest.point_id_for("doc1", 0)
    b = kb_ingest.point_id_for("doc1", 0)
    assert a == b
    assert a != kb_ingest.point_id_for("doc1", 1)


# ---------------------------------------------------------------------------
# Ingest / list / delete / status (Qdrant mocked)
# ---------------------------------------------------------------------------

def test_ingest_document_creates_points(fake_kb):
    data = "Fact one.\n\nFact two.\n\nFact three.".encode()
    result = kb_ingest.ingest_document("notes.txt", data)
    assert result.status == "created"
    assert result.document.chunk_count >= 1
    assert result.document.document_name == "notes.txt"
    assert result.document.file_type == "txt"
    assert len(fake_kb.upserts) == 1
    point = fake_kb.upserts[0][0]
    payload = point.payload
    assert payload["source_type"] == "kb"
    assert payload["document_id"] == result.document.document_id
    assert payload["source_id"] == f"{result.document.document_id}:0"
    assert payload["chunk_index"] == 0
    assert payload["total_chunks"] == result.document.chunk_count
    assert payload["content"]


def test_duplicate_ingestion_replaces(fake_kb):
    data = "Deterministic duplicate policy test.".encode()
    first = kb_ingest.ingest_document("doc.txt", data)
    assert first.status == "created"
    second = kb_ingest.ingest_document("doc.txt", data)
    assert second.status == "replaced"
    assert second.document.document_id == first.document.document_id
    # Old chunks were deleted before re-upsert; no duplicate growth.
    assert len(fake_kb.deleted_filters) == 1
    listing = kb_ingest.list_documents()
    assert listing.total == 1
    assert listing.documents[0].chunk_count == second.document.chunk_count


def test_same_name_different_content_is_new_document(fake_kb):
    kb_ingest.ingest_document("doc.txt", b"content A")
    result = kb_ingest.ingest_document("doc.txt", b"content B")
    assert result.status == "created"
    assert kb_ingest.list_documents().total == 2


def test_ingest_rejects_oversized_files(fake_kb, monkeypatch):
    monkeypatch.setattr(kb_ingest, "MAX_UPLOAD_BYTES", 16)
    with pytest.raises(kb_ingest.KBIngestError, match="large"):
        kb_ingest.ingest_document("big.txt", b"x" * 32)


def test_ingest_rejects_no_text_pdf(fake_kb):
    with pytest.raises(kb_ingest.KBIngestError):
        kb_ingest.ingest_document("scan.pdf", b"%PDF-1.4 not really a pdf")


def test_list_documents_groups_and_sorts(fake_kb):
    import time

    kb_ingest.ingest_document("older.txt", b"first document content")
    time.sleep(0.03)  # guarantee distinct timestamps (Windows clock resolution)
    kb_ingest.ingest_document("newer.md", b"second document content")
    listing = kb_ingest.list_documents()
    assert listing.total == 2
    names = {d.document_name for d in listing.documents}
    assert names == {"older.txt", "newer.md"}
    # Newest first.
    assert listing.documents[0].document_name == "newer.md"


def test_delete_document(fake_kb):
    result = kb_ingest.ingest_document("todelete.txt", b"delete me content")
    doc_id = result.document.document_id
    deleted = kb_ingest.delete_document(doc_id)
    assert deleted.status == "deleted"
    assert deleted.document_id == doc_id
    assert deleted.chunks_deleted == result.document.chunk_count
    assert kb_ingest.list_documents().total == 0


def test_delete_missing_document_raises(fake_kb):
    with pytest.raises(ValueError, match="not found"):
        kb_ingest.delete_document("nonexistent")


def test_kb_status_available(fake_kb):
    kb_ingest.ingest_document("s.txt", b"status probe document")
    status = kb_ingest.kb_status()
    assert status.available is True
    assert status.documents == 1
    assert status.chunks >= 1
    assert status.error is None


def test_kb_status_unavailable_never_raises(monkeypatch):
    def boom():
        raise qdrant_tool.QdrantUnavailableError("Qdrant search failed: ConnectError")

    monkeypatch.setattr(kb_ingest, "_get_client", boom)
    status = kb_ingest.kb_status()
    assert status.available is False
    assert status.error is not None
    assert "GROQ" not in (status.error or "")


# ---------------------------------------------------------------------------
# API endpoints (TestClient; ingest layer mocked at module boundary)
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(app)


def test_upload_endpoint_txt(client, fake_kb):
    resp = client.post(
        "/kb/upload",
        files={"file": ("hello.txt", b"Hello ingestion world.", "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "created"
    assert body["document"]["document_name"] == "hello.txt"
    assert body["document"]["chunk_count"] >= 1


def test_upload_endpoint_rejects_bad_extension(client, fake_kb):
    resp = client.post(
        "/kb/upload",
        files={"file": ("bad.exe", b"MZ...", "application/octet-stream")},
    )
    assert resp.status_code == 422
    assert "Unsupported" in resp.json()["detail"]


def test_upload_endpoint_rejects_empty_file(client, fake_kb):
    resp = client.post(
        "/kb/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert resp.status_code == 422


def test_documents_endpoint(client, fake_kb):
    kb_ingest.ingest_document("listed.txt", b"listing endpoint content")
    resp = client.get("/kb/documents")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["documents"][0]["document_name"] == "listed.txt"


def test_delete_endpoint(client, fake_kb):
    result = kb_ingest.ingest_document("gone.txt", b"delete endpoint content")
    resp = client.delete(f"/kb/documents/{result.document.document_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"
    missing = client.delete(f"/kb/documents/{result.document.document_id}")
    assert missing.status_code == 404


def test_status_endpoint_unavailable(client, monkeypatch):
    def boom():
        raise qdrant_tool.QdrantUnavailableError("Qdrant count failed: ConnectError")

    monkeypatch.setattr(kb_ingest, "_get_client", boom)
    resp = client.get("/kb/status")
    assert resp.status_code == 200
    assert resp.json()["available"] is False
