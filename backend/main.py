import queue
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, field_validator

from agents.critic import CriticError
from agents.critic import MissingGroqKeyError as CriticMissingGroqKeyError
from agents.events import ResearchEvent, format_sse
from agents.orchestrator import run_research_pipeline
from agents.planner import MissingGroqKeyError, PlannerError, create_research_plan
from agents.schemas import (
    RETRIEVAL_MODES,
    EvidenceBundle,
    Report,
    ResearchPlan,
    ResearchRunResult,
    RetrievalMode,
)
from agents.writer import InsufficientEvidenceError, WriterError, write_report
from config import get_config
from kb import ingest as kb_ingest
from kb.models import KBDocumentList, KBDeleteResult, KBStatus, IngestResult
from tools.qdrant_tool import EMBEDDING_MODEL_NAME, QdrantUnavailableError
from tools.retrieval import RetrievalError, retrieve_evidence

app = FastAPI(
    title="AgentDesk",
    description="Multi-agent research assistant",
    version="1.0.0",
)

# CORS for direct-connect frontend development (production uses the
# frontend's reverse proxy and stays same-origin). Origins are
# configurable; never expose credentials cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


class PlanRequest(BaseModel):
    query: str

    @field_validator("query", mode="before")
    @classmethod
    def reject_empty(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("query must be a non-empty string")
        return value


class RetrieveRequest(BaseModel):
    queries: list[str]

    @field_validator("queries", mode="before")
    @classmethod
    def reject_empty(cls, value):
        if not isinstance(value, list) or not any(
            isinstance(q, str) and q.strip() for q in value
        ):
            raise ValueError("queries must contain at least one non-empty string")
        return value


class WriteRequest(BaseModel):
    query: str
    plan: ResearchPlan
    evidence: EvidenceBundle

    @field_validator("query", mode="before")
    @classmethod
    def reject_empty(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("query must be a non-empty string")
        return value


class ResearchRequest(BaseModel):
    query: str
    retrieval_mode: RetrievalMode = "both"

    @field_validator("query", mode="before")
    @classmethod
    def reject_empty(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("query must be a non-empty string")
        return value


class ReportPdfRequest(BaseModel):
    """PDF export request: the final report plus display metadata."""

    report: Report
    query: str = ""
    retrieval_mode: str = "both"
    final_status: str = ""


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "AgentDesk", "running": True}


@app.get("/runtime")
def runtime() -> dict:
    """Non-secret runtime settings for the Settings UI.

    Booleans only for provider configuration — never keys or tokens.
    """
    cfg = get_config()
    return {
        "model": cfg.groq_model,
        "groq_configured": cfg.has_groq_key,
        "tavily_configured": bool(cfg.tavily_api_key),
        "retrieval_modes": list(RETRIEVAL_MODES),
        "kb_collection": kb_ingest.DEFAULT_COLLECTION,
        "embedding_model": EMBEDDING_MODEL_NAME,
    }


@app.post("/plan", response_model=ResearchPlan)
def plan(request: PlanRequest) -> ResearchPlan:
    try:
        return create_research_plan(request.query)
    except MissingGroqKeyError:
        raise HTTPException(
            status_code=503,
            detail="LLM provider is not configured.",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except PlannerError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/retrieve", response_model=EvidenceBundle)
def retrieve(request: RetrieveRequest) -> EvidenceBundle:
    """Development-only retrieval endpoint (temporary until /query exists)."""
    queries = [q for q in request.queries if q.strip()]
    try:
        return retrieve_evidence(queries)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RetrievalError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/write", response_model=Report)
def write(request: WriteRequest) -> Report:
    """Development-only writer endpoint (temporary until /query exists)."""
    try:
        return write_report(request.plan, request.evidence)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except InsufficientEvidenceError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except WriterError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


def _run_research_or_error(query: str, retrieval_mode: str) -> ResearchRunResult:
    """Shared boundary for /research and /query: run pipeline, map errors."""
    try:
        return run_research_pipeline(
            query.strip(), retrieval_mode=retrieval_mode
        )
    except MissingGroqKeyError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except CriticMissingGroqKeyError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RetrievalError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except (PlannerError, WriterError, CriticError) as exc:
        if getattr(exc, "retry_after", None) is not None:
            raise HTTPException(
                status_code=429,
                detail="AI provider rate limit reached. Please try again later.",
            )
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/research", response_model=ResearchRunResult)
def research(request: ResearchRequest) -> ResearchRunResult:
    """Full pipeline (kept for backward compatibility).

    Prefer POST /query for new integrations; semantics are identical.
    """
    return _run_research_or_error(request.query, request.retrieval_mode)


@app.post("/query", response_model=ResearchRunResult)
def query(request: ResearchRequest) -> ResearchRunResult:
    """Authoritative synchronous research endpoint.

    Runs Planner -> Retriever -> Writer -> Critic (up to two revisions) and
    returns the final ResearchRunResult including evaluation and timing.
    For real-time progress use GET /research/stream instead.
    """
    return _run_research_or_error(request.query, request.retrieval_mode)


@app.post("/report/pdf")
def report_pdf(request: ReportPdfRequest) -> Response:
    """Render a final report as a downloadable PDF."""
    import pdf_export

    try:
        pdf_bytes = pdf_export.render_report_pdf(
            request.report,
            query=request.query,
            retrieval_mode=request.retrieval_mode,
            final_status=request.final_status,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    filename = pdf_export.slugify(request.report.title or "agentdesk-report")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}.pdf"'
        },
    )


# ---------------------------------------------------------------------------
# Knowledge base management
# ---------------------------------------------------------------------------

@app.post("/kb/upload", response_model=IngestResult)
async def kb_upload(file: UploadFile = File(...)) -> IngestResult:
    """Ingest a PDF/TXT/MD document into the Qdrant knowledge base."""
    filename = file.filename or ""
    try:
        data = await file.read()
    finally:
        await file.close()
    try:
        return kb_ingest.ingest_document(filename, data)
    except kb_ingest.KBIngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except QdrantUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base storage is unavailable.",
        )


@app.get("/kb/documents", response_model=KBDocumentList)
def kb_documents() -> KBDocumentList:
    try:
        return kb_ingest.list_documents()
    except QdrantUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base storage is unavailable.",
        )


@app.delete("/kb/documents/{document_id}", response_model=KBDeleteResult)
def kb_delete_document(document_id: str) -> KBDeleteResult:
    try:
        return kb_ingest.delete_document(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except QdrantUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base storage is unavailable.",
        )


@app.get("/kb/status", response_model=KBStatus)
def kb_status() -> KBStatus:
    return kb_ingest.kb_status()


# ---------------------------------------------------------------------------
# SSE streaming layer
# ---------------------------------------------------------------------------

_QUEUE_TIMEOUT_SECONDS = 20  # keep-alive interval while the worker runs
_QUEUE_MAXSIZE = 100         # events are few; bounded so the worker never floods


def _stream_research(query: str, retrieval_mode: str):
    """Generator bridging the synchronous pipeline to SSE frames.

    The pipeline runs once in a single bounded worker thread; progress events
    flow through a bounded queue. The endpoint is transport-only: the final
    `research_completed` event carries the authoritative orchestrator result.
    """
    events: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
    _DONE = object()

    def callback(event: ResearchEvent) -> None:
        events.put(("event", event))

    def work() -> None:
        try:
            run_research_pipeline(
                query, event_callback=callback, retrieval_mode=retrieval_mode
            )
        except Exception:
            # A safe research_failed event was already emitted by the
            # orchestrator; internal details are never re-raised to the client.
            pass
        finally:
            events.put((_DONE, None))

    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(work)
    try:
        while True:
            try:
                kind, payload = events.get(timeout=_QUEUE_TIMEOUT_SECONDS)
            except queue.Empty:
                # SSE comment frame keeps intermediaries from closing the stream.
                yield ": keep-alive\n\n"
                continue
            if kind == "event":
                yield format_sse(payload)
                continue
            break
    finally:
        executor.shutdown(wait=False)


@app.get("/research/stream")
def research_stream(
    query: str = Query(...),
    retrieval_mode: RetrievalMode = Query("both"),
) -> StreamingResponse:
    """Stream a full research run as Server-Sent Events.

    Emits structured ResearchEvent JSON payloads with valid SSE framing;
    terminates after research_completed or research_failed. Clients that
    omit retrieval_mode get "both" (backward compatible).
    """
    if not query or not query.strip():
        raise HTTPException(status_code=422, detail="query must be a non-empty string")
    return StreamingResponse(
        _stream_research(query.strip(), retrieval_mode),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
