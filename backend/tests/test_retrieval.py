"""Tests for the retrieval layer. All external services are mocked."""

import pytest
from pydantic import ValidationError

import tools.qdrant_tool as qdrant_tool
import tools.retrieval as retrieval
import tools.tavily_tool as tavily_tool
from agents.schemas import EvidenceBundle, EvidenceSource, RetrievalSummary
from config import get_config
from tools.retrieval import (
    RetrievalError,
    deduplicate,
    normalize_url,
    retrieve_evidence,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

def _web_source(**overrides):
    base = dict(
        source_type="web",
        title="AI in Indian healthcare",
        url="https://example.com/article",
        content="Some content",
        score=0.9,
    )
    return EvidenceSource(**{**base, **overrides})


def _kb_source(**overrides):
    base = dict(
        source_type="kb",
        title="Report",
        document_name="report.pdf",
        content="KB chunk",
        score=0.8,
        source_id="doc1",
    )
    return EvidenceSource(**{**base, **overrides})


def test_evidence_source_web_valid():
    source = _web_source()
    assert source.document_name is None
    assert source.source_type == "web"


def test_evidence_source_kb_valid():
    source = _kb_source()
    assert source.url is None
    assert source.source_type == "kb"


def test_evidence_source_invalid_type_rejected():
    with pytest.raises(ValidationError):
        EvidenceSource(source_type="twitter", title="x", content="y")


def test_evidence_bundle_valid():
    bundle = EvidenceBundle(
        original_query="q",
        sources=[_web_source(), _kb_source()],
        retrieval_summary=RetrievalSummary(
            total_sources=2,
            web_sources=1,
            kb_sources=1,
            deduplicated_sources=0,
            kb_status="ok",
            web_status="ok",
        ),
    )
    assert len(bundle.sources) == 2


def test_evidence_summary_invalid_status_rejected():
    with pytest.raises(ValidationError):
        RetrievalSummary(
            total_sources=0,
            web_sources=0,
            kb_sources=0,
            deduplicated_sources=0,
            kb_status="broken",
            web_status="ok",
        )


# ---------------------------------------------------------------------------
# Qdrant tool (mocked client + embeddings)
# ---------------------------------------------------------------------------

class _FakePoint:
    def __init__(self, pid, score, payload):
        self.id = pid
        self.score = score
        self.payload = payload


class _FakeQueryResponse:
    def __init__(self, points):
        self.points = points


class _FakeCollections:
    def __init__(self, exists):
        self.exists = exists


class _FakeQdrantClient:
    def __init__(self, points=None, collection_exists=False):
        self.points = points or []
        self._exists = collection_exists
        self.created_with = None
        self.searched = None

    def collection_exists(self, collection):
        return self._exists

    def create_collection(self, collection_name, vectors_config):
        self.created_with = (collection_name, vectors_config)
        self._exists = True

    def query_points(self, **kwargs):
        self.searched = kwargs
        return _FakeQueryResponse(self.points)


@pytest.fixture
def fake_client(monkeypatch):
    client = _FakeQdrantClient(collection_exists=True)
    monkeypatch.setattr(qdrant_tool, "_get_client", lambda: client)
    monkeypatch.setattr(qdrant_tool, "_embed", lambda text: [0.1] * 384)
    return client


def test_qdrant_normal_conversion(fake_client):
    fake_client.points = [
        _FakePoint(1, 0.92, {
            "content": "chunk text",
            "title": "Doc title",
            "document_name": "a.pdf",
            "source_id": "abc",
        })
    ]
    results = qdrant_tool.search_qdrant("query", limit=3)
    assert len(results) == 1
    source = results[0]
    assert source.source_type == "kb"
    assert source.content == "chunk text"
    assert source.document_name == "a.pdf"
    assert source.score == pytest.approx(0.92)
    assert source.source_id == "abc"
    assert fake_client.searched["collection_name"] == qdrant_tool.DEFAULT_COLLECTION


def test_qdrant_missing_payload_fields_uses_defaults(fake_client):
    fake_client.points = [_FakePoint(7, 0.5, {})]
    results = qdrant_tool.search_qdrant("query")
    source = results[0]
    assert source.title == ""
    assert source.content == ""
    assert source.url is None
    assert source.document_name is None
    assert source.source_id == "7"


def test_qdrant_payload_text_fallback(fake_client):
    fake_client.points = [_FakePoint(2, 0.4, {"text": "legacy field"})]
    assert qdrant_tool.search_qdrant("q")[0].content == "legacy field"


def test_qdrant_empty_collection_returns_no_sources(fake_client):
    assert qdrant_tool.search_qdrant("anything") == []


def test_qdrant_connection_error_normalized(monkeypatch):
    def boom():
        raise ConnectionError("refused")

    monkeypatch.setattr(qdrant_tool, "_get_client", boom)
    with pytest.raises(qdrant_tool.QdrantUnavailableError) as excinfo:
        qdrant_tool.search_qdrant("query")
    assert "GROQ" not in str(excinfo.value)
    assert "TAVILY" not in str(excinfo.value)


def test_qdrant_collection_created_only_when_missing(monkeypatch):
    client = _FakeQdrantClient(collection_exists=False)
    monkeypatch.setattr(qdrant_tool, "_get_client", lambda: client)
    monkeypatch.setattr(qdrant_tool, "_embed", lambda text: [0.0] * 384)

    qdrant_tool.ensure_collection(client, "agentdesk_knowledge")
    assert client.created_with is not None
    name, vectors = client.created_with
    assert name == "agentdesk_knowledge"
    assert vectors.size == 384  # all-MiniLM-L6-v2 dimensionality
    from qdrant_client import models as qmodels

    assert vectors.distance == qmodels.Distance.COSINE

    # Existing collections must never be recreated or deleted.
    client.created_with = None
    qdrant_tool.ensure_collection(client, "agentdesk_knowledge")
    assert client.created_with is None


# ---------------------------------------------------------------------------
# Tavily tool (mocked)
# ---------------------------------------------------------------------------

class _FakeTavilyClient:
    def __init__(self, response=None, error=None):
        self.response = response or {}
        self.error = error
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def _tavily_response():
    return {
        "results": [
            {
                "title": "Result A",
                "url": "https://example.com/a",
                "content": "snippet text",
                "raw_content": "# full markdown\nmore detail",
                "score": 0.77,
            },
            {"title": "No content", "url": "https://example.com/b"},
        ]
    }


def _patch_tavily(monkeypatch, fake):
    monkeypatch.setattr(tavily_tool, "_get_client", lambda: fake)


def test_tavily_normal_conversion(monkeypatch):
    fake = _FakeTavilyClient(response=_tavily_response())
    _patch_tavily(monkeypatch, fake)
    results = tavily_tool.search_tavily("AI healthcare India", limit=2)
    assert len(results) == 2
    first = results[0]
    assert first.source_type == "web"
    assert first.title == "Result A"
    assert first.url == "https://example.com/a"
    assert "full markdown" in first.content
    assert first.score == pytest.approx(0.77)
    assert fake.calls[0]["search_depth"] == "advanced"
    assert fake.calls[0]["max_results"] == 2


def test_tavily_missing_content_defaults_to_empty(monkeypatch):
    fake = _FakeTavilyClient(response=_tavily_response())
    _patch_tavily(monkeypatch, fake)
    second = tavily_tool.search_tavily("q")[1]
    assert second.content == ""


def test_tavily_api_failure_raises_unavailable(monkeypatch):
    fake = _FakeTavilyClient(error=RuntimeError("502 from upstream"))
    _patch_tavily(monkeypatch, fake)
    with pytest.raises(tavily_tool.TavilyUnavailableError):
        tavily_tool.search_tavily("q")


def test_tavily_missing_key_raises_unavailable(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(tavily_tool, "_get_client", tavily_tool._get_client)
    get_config.cache_clear()
    try:
        with pytest.raises(tavily_tool.TavilyUnavailableError):
            tavily_tool.search_tavily("q")
    finally:
        get_config.cache_clear()


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_normalize_url_strips_scheme_www_and_trailing_slash():
    assert (
        normalize_url("HTTPS://www.Example.com/Path/")
        == "example.com/path"
    )
    assert normalize_url(None) is None


def test_duplicate_web_urls_removed_best_score_kept():
    low = _web_source(url="https://example.com/a", score=0.3, content="weak")
    high = _web_source(url="https://example.com/a/", score=0.9, content="rich")
    result, removed = deduplicate([low, high])
    assert removed == 1
    assert result == [high]


def test_duplicate_kb_sources_removed_by_document_identity():
    first = _kb_source(source_id="doc1", content="same chunk", score=0.6)
    better = _kb_source(source_id="doc1", content="same chunk", score=0.95)
    other = _kb_source(source_id="doc2", content="different", score=0.5)
    result, removed = deduplicate([first, better, other])
    assert removed == 1
    assert len(result) == 2
    assert better in result and other in result


def test_web_before_kb_then_descending_score():
    web_low = _web_source(url="https://x.com/1", score=0.2)
    web_high = _web_source(url="https://x.com/2", score=0.8)
    kb = _kb_source(score=0.99)
    result, _ = deduplicate([kb, web_low, web_high])
    assert [s.source_type for s in result] == ["web", "web", "kb"]
    assert result[0].score == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Orchestration (both sources mocked at module boundary)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_searches(monkeypatch):
    calls = {"kb": [], "web": []}

    def kb(query, limit=5):
        calls["kb"].append(query)
        return [_kb_source(source_id=f"kb-{query[:8]}")]

    def web(query, limit=5):
        calls["web"].append(query)
        return [
            _web_source(url="https://example.com/shared"),
            _web_source(url=f"https://example.com/{query[:8]}"),
        ]

    monkeypatch.setattr(retrieval, "search_qdrant", kb)
    monkeypatch.setattr(retrieval, "search_tavily", web)
    return calls


def test_orchestration_processes_all_queries_both_sources(mock_searches):
    q1 = "applications of AI healthcare"
    q2 = "ethics of AI medicine"
    bundle = retrieve_evidence([q1, q2])
    assert sorted(mock_searches["kb"]) == [q1, q2]
    assert sorted(mock_searches["web"]) == [q1, q2]
    # 2 queries x (1 kb + 2 web); the "shared" URL dedupes to one across queries.
    assert bundle.retrieval_summary.total_sources == 5
    assert bundle.retrieval_summary.kb_sources == 2
    assert bundle.retrieval_summary.web_sources == 3
    assert bundle.retrieval_summary.deduplicated_sources == 1
    assert bundle.original_query == q1


def test_orchestration_deduplicates_across_queries(mock_searches):
    bundle = retrieve_evidence(["shared topic", "shared topic again"])
    # The shared URL appears once despite being returned for both queries.
    assert bundle.retrieval_summary.deduplicated_sources >= 1
    urls = [s.url for s in bundle.sources if s.source_type == "web"]
    assert len(urls) == len(set(normalize_url(u) for u in urls))


def test_orchestration_qdrant_down_web_continues(monkeypatch, mock_searches):
    def kb_fail(query, limit=5):
        raise qdrant_tool.QdrantUnavailableError("Qdrant search failed")

    monkeypatch.setattr(retrieval, "search_qdrant", kb_fail)
    bundle = retrieve_evidence(["some question"])
    assert bundle.retrieval_summary.kb_status == "unavailable"
    assert bundle.retrieval_summary.web_status == "ok"
    assert bundle.retrieval_summary.kb_sources == 0
    assert bundle.retrieval_summary.web_sources > 0


def test_orchestration_tavily_missing_key_kb_continues(monkeypatch):
    def web_fail(query, limit=5):
        raise tavily_tool.TavilyUnavailableError("Web search is not configured")

    monkeypatch.setattr(
        retrieval, "search_qdrant",
        lambda query, limit=5: [_kb_source()],
    )
    monkeypatch.setattr(retrieval, "search_tavily", web_fail)
    bundle = retrieve_evidence(["kb-only question"])
    assert bundle.retrieval_summary.web_status == "unavailable"
    assert bundle.retrieval_summary.kb_status == "ok"
    assert bundle.retrieval_summary.kb_sources == 1


def test_orchestration_empty_kb_reports_empty_not_crash(monkeypatch):
    monkeypatch.setattr(retrieval, "search_qdrant", lambda query, limit=5: [])
    monkeypatch.setattr(
        retrieval, "search_tavily",
        lambda query, limit=5: [_web_source()],
    )
    bundle = retrieve_evidence(["anything"])
    assert bundle.retrieval_summary.kb_status == "empty"
    assert bundle.retrieval_summary.total_sources == 1


def test_orchestration_both_unavailable_raises(monkeypatch):
    def kb_fail(query, limit=5):
        raise qdrant_tool.QdrantUnavailableError("down")

    def web_fail(query, limit=5):
        raise tavily_tool.TavilyUnavailableError("unconfigured")

    monkeypatch.setattr(retrieval, "search_qdrant", kb_fail)
    monkeypatch.setattr(retrieval, "search_tavily", web_fail)
    with pytest.raises(RetrievalError):
        retrieve_evidence(["question"])


def test_orchestration_blank_queries_rejected():
    with pytest.raises(ValueError):
        retrieve_evidence(["", "   "])


# ---------------------------------------------------------------------------
# Retriever agent service interface
# ---------------------------------------------------------------------------

def test_retrieve_for_plan_uses_plan_fields(monkeypatch):
    from agents.retriever import retrieve_for_plan
    from agents.schemas import ResearchPlan

    captured = {}

    def fake_retrieve(queries, original_query="", **kwargs):
        captured["queries"] = list(queries)
        captured["original_query"] = original_query
        return EvidenceBundle(
            original_query=original_query,
            sources=[],
            retrieval_summary=RetrievalSummary(
                total_sources=0, web_sources=0, kb_sources=0,
                deduplicated_sources=0, kb_status="empty", web_status="ok",
            ),
        )

    monkeypatch.setattr("agents.retriever.retrieve_evidence", fake_retrieve)
    plan = ResearchPlan(
        original_query="goal", sub_questions=["sq1", "sq2", "sq3"]
    )
    retrieve_for_plan(plan)
    assert captured["queries"] == ["sq1", "sq2", "sq3"]
    assert captured["original_query"] == "goal"
