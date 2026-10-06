# AgentDesk

AgentDesk is a multi-agent research assistant: give it a natural-language
research goal and a team of AI agents plans the research, retrieves real
evidence from a local Qdrant knowledge base and live web search, writes a
cited report, audits it, revises it when necessary, and streams every step
to a modern React workspace.

> Grounding metrics everywhere in this project are explicitly-labeled
> **heuristics** — they measure mechanical grounding signals (citations,
> coverage, source balance), not factual truth.

## Features

- **Multi-agent pipeline** — Planner → Retriever → Writer → Critic with a
  bounded, evidence-preserving self-correction loop (max 2 revisions)
- **Dual-source retrieval** with user-selectable modes: `both`, `web`
  (Tavily), or `kb` (Qdrant)
- **Knowledge base ingestion** — upload PDF/TXT/MD documents; they are
  extracted, normalized, chunked, embedded locally (all-MiniLM-L6-v2), and
  stored in Qdrant; list and delete documents
- **Deterministic citations** — the backend assigns source IDs, the Writer
  may only cite those, and every inline `[N]` is validated before a report
  is accepted; authoritative references are built from the real evidence
- **Real-time SSE progress** — structured progress events stream to the UI
- **Clickable citations** — every `[N]` in the report scrolls to and
  highlights the matching source; sources show type (WEB/KB), title, URL,
  relevance score, and a content preview
- **Evidence grounding indicator** — a documented heuristic badge
  (High/Medium/Low grounding) derived from citation coverage
- **Run summary** — sources, web/KB balance, citations, coverage, critic
  verdict, iterations, final status, stage timings
- **Evaluation system** — deterministic metrics computed after every run:
  citation coverage, valid-citation rate, plan coverage, retrieval balance,
  critic outcome, and a composite grounding score (formula documented in
  `backend/eval/scorer.py`)
- **Evaluation dashboard** — compare grounding metrics across saved runs
- **Query history** — completed runs saved in browser localStorage, reload
  reports anytime, export Markdown or PDF
- **PDF export** — server-side rendering of the final report (fpdf2)
- **Benchmark harness** — `python -m eval.benchmark` runs a fixed query set
  and persists a JSON quality snapshot
- **Full-stack Docker Compose** — Qdrant + FastAPI + nginx-served React with
  health checks and SSE-safe proxying

## Architecture

```
User
 ↓
React (Vite + Tailwind)  ←— SSE ——┐
 ↓  /api/*                        │
nginx (prod) / Vite proxy (dev)   │
 ↓                                │
FastAPI  ─────────────────────────┘
 ↓
Orchestrator  (the single pipeline; SSE is transport-only)
 ├── Planner   (Groq, strict structured output → ResearchPlan)
 ├── Retriever (real Python tools — the LLM never searches)
 │    ├── Qdrant KB  (local all-MiniLM-L6-v2 embeddings)
 │    └── Tavily web (advanced search)
 │    → dedup, normalize, EvidenceBundle (immutable for the whole run)
 ├── Writer    (Groq, strict structured output → grounded, cited Report)
 └── Critic    (Groq, strict structured output → CritiqueResult)
      └── revision loop, max 2, same EvidenceBundle
 ↓
ResearchRunResult (+ EvaluationResult + RunTiming)
```

Agents use **Groq-hosted `openai/gpt-oss-120b`** (configurable via
`GROQ_MODEL`) with native strict JSON-Schema structured outputs. Responses
are re-validated with Pydantic before they leave the agent layer. LLM
attempts are bounded (2 per stage) with provider-aware backoff on HTTP 429:
the wait honors `Retry-After` and Groq's `x-ratelimit-reset-tokens` reset
information, and no single sleep ever exceeds 60 seconds. Per-minute limits
(TPM) fail the current request after bounded retries; daily-token exhaustion
(TPD) is detected and stops benchmark runs cleanly so they can be resumed
later without losing completed work.

### Core guarantees

1. Retrieval is performed by deterministic Python tools; the LLM never
   fabricates search results or URLs.
2. The Writer sees only the numbered evidence; citation IDs are constrained
   to real sources and validated after generation.
3. Final references are constructed by the backend from actual
   `EvidenceSource` objects.
4. The Critic evaluates only against the supplied evidence and never
   substitutes its own world knowledge as a "correction".
5. Revisions reuse the same `EvidenceBundle` — no new retrieval — and never
   let an invalid report through (a failed revision preserves the last
   valid report, `final_status = "revision_failed"`).
6. The SSE endpoint is transport only; it never duplicates orchestration.
7. API secrets stay in `.env` and are never logged, returned, or shipped to
   the frontend.

## Setup

Requirements: Python 3.12+, Node.js 20+, Docker (for Qdrant / full stack).

### 1. Configure environment

```bash
cp .env.example .env
# fill in GROQ_API_KEY (required) and TAVILY_API_KEY (required for web mode)
```

| Variable | Purpose |
| --- | --- |
| `GROQ_API_KEY` | Groq API key (server-side only) |
| `GROQ_MODEL` | Groq model id, default `openai/gpt-oss-120b` |
| `TAVILY_API_KEY` | Tavily web search key (server-side only) |
| `QDRANT_URL` | Qdrant endpoint; `http://localhost:6333` locally, `http://qdrant:6333` in Docker |
| `VITE_API_BASE_URL` | Frontend: leave empty (same-origin `/api` proxy) |

### Local development

```bash
# Qdrant
docker compose up -d qdrant

# Backend
cd backend
python -m venv .venv
.venv\Scripts\activate          # bash: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (second terminal)
cd frontend
npm install
npm run dev                     # http://localhost:5173, /api proxied to :8000
```

### Docker Compose (full stack)

```bash
docker compose up --build
# frontend:  http://localhost:5173
# backend:   http://localhost:8000  (docs at /docs)
# qdrant:    http://localhost:6333
docker compose down          # keeps the qdrant_data volume
```

## API

| Endpoint | Description |
| --- | --- |
| `GET /health` | Liveness probe |
| `GET /runtime` | Non-secret runtime settings (model, provider booleans, KB collection) |
| `POST /plan` | Research goal → `ResearchPlan` |
| `POST /retrieve` | Dev-only: queries → `EvidenceBundle` |
| `POST /write` | Dev-only: plan + evidence → `Report` |
| `POST /research` | Full pipeline (backward compatibility; same as `/query`) |
| `POST /query` | **Authoritative synchronous research endpoint** → `ResearchRunResult` |
| `GET /research/stream` | **SSE**: full pipeline as structured events (`?query=...&retrieval_mode=both`) |
| `POST /kb/upload` | Multipart upload of PDF/TXT/MD → ingest → document metadata |
| `GET /kb/documents` | Unique documents (not chunks) |
| `DELETE /kb/documents/{document_id}` | Delete one document and all its chunks |
| `GET /kb/status` | KB availability, collection, document/chunk counts |
| `POST /report/pdf` | Render a final report as a downloadable PDF |

Request/response examples:

```
POST /query
{ "query": "Research the impact of AI on healthcare in India",
  "retrieval_mode": "both" }
→ ResearchRunResult { query, plan, evidence, report, critique,
                      iteration_count, final_status, retrieval_mode,
                      evaluation, timing }
```

`/query` vs `/research/stream`: `/query` returns the final result
synchronously; `/research/stream` streams the same pipeline as SSE events
(`research_started`, `planner_*`, `retrieval_*`, `writer_*`, `critic_*`,
`revision_*`, `research_completed`, `research_failed`) for real-time UIs.
Events never contain prompts, provider responses, or stack traces.

## Knowledge base

```
upload (.pdf/.txt/.md, ≤10 MB)
  → extract text   (pypdf for PDFs, UTF-8 for text)
  → normalize      (whitespace, paragraph-preserving)
  → chunk          (deterministic: ~1200 chars target, ~200 chars overlap,
                    paragraph-accumulating with sentence-boundary fallback —
                    sized to the embedding model's context window)
  → embed          (shared local all-MiniLM-L6-v2 — same model as retrieval)
  → Qdrant upsert  (deterministic UUID point IDs, payload: document_id,
                    document_name, title, content, chunk_index, total_chunks,
                    source_id, content_hash, file_type, created_at)
```

- **Identity:** `document_id` = first 16 hex chars of the SHA-256 content
  hash. **Duplicate policy:** re-uploading identical content REPLACES the
  existing document's chunks (delete + re-ingest, status `"replaced"`); the
  same filename with different content is a separate document.
- **Chunk `source_id`** is `document_id:chunk_index` — stable across
  re-ingestion so citation deduplication stays consistent.

## Retrieval modes

| Mode | Sources queried | Failure behavior |
| --- | --- | --- |
| `both` (default) | Qdrant + Tavily in parallel | Fails only if BOTH fail; one source failing/empty degrades gracefully and is reported in the retrieval summary (`kb_status` / `web_status` = `ok`/`empty`/`unavailable`/`skipped`) |
| `web` | Tavily only | Controlled failure if web is unavailable; KB reports `skipped` |
| `kb` | Qdrant only | Controlled failure if KB is unavailable; web reports `skipped` |

## Critic loop

```
Writer (iteration 0) → Critic audit
  approved            → final_status = "approved"
  revision_needed     → Writer revision 1 (same evidence) → Critic audit
                          → approved → "approved"
                          → revision_needed → revision 2 → audit
                          → still revision_needed → "max_revisions_reached"
  revision fails validation → keep last valid report, "revision_failed"
```

## Evaluation

Computed deterministically by `backend/eval/scorer.py` after every run —
no extra LLM calls. All values are heuristics:

- **Citation coverage** — share of "claim sentences" (6+ words, not
  headings/references/transition openers) carrying ≥1 valid inline citation
- **Valid citation rate** — valid `[N]` references ÷ total detected (1.0 for
  any accepted report)
- **Plan coverage** — share of planned sub-questions whose salient keywords
  appear in the report body (lexical overlap; synonyms score as uncovered —
  documented limitation)
- **Retrieval balance** — web/KB percentages of retrieved sources
- **Critic outcome + iteration count** — from the quality gate
- **Grounding score** — composite:
  `0.40·coverage + 0.25·valid_rate + 0.20·plan_coverage + 0.15·critic`
  (critic: approved 1.0, max_revisions_reached 0.5, revision_failed 0.0)

Benchmark harness:

```bash
cd backend
python -m eval.benchmark              # default query set → eval/results/*.json
python -m eval.benchmark --query "..." --mode kb
```

(Consumes live provider quota; automated tests use mocks.)

## Hallucination benchmark (ablation study)

An additive harness (`backend/eval/hallucination.py`) measures whether
AgentDesk reduces **unsupported factual claims** versus an ungrounded LLM.
It runs the SAME production pipeline at three escalation levels — no pipeline
behavior is modified:

| Variant | What runs |
|---|---|
| `baseline` | direct LLM call, question only. No retrieval, planner, citation constraints, critic, or revision. |
| `grounded` | Planner → Retriever → Writer (citation-constrained); **critic/revision bypassed** |
| `agentdesk` | the complete production pipeline incl. Critic + revision loop |

### Unsupported claim rate ≠ citation coverage

**Citation coverage** (above) is a mechanical measure: did a sentence carry a
citation? A report can cite a source for a claim the source never makes.
The hallucination benchmark instead extracts factual claims from each
generated answer and **classifies each claim against the question's
authoritative reference corpus** (from the dataset — never against the
answer's own citations):

```
unsupported_claim_rate =
    (unsupported + contradicted claims) / total evaluated claims
```

Labels: `supported` (corpus entails the claim), `unsupported` (on-topic but
not substantiated), `contradicted`, `unverifiable`. Per-claim verdicts,
justifications, and evidence quotes are preserved in every result for audit.

```
hallucination_reduction_percent =
    ((baseline_rate − variant_rate) / baseline_rate) × 100
```

If the baseline rate is zero (or either rate is undefined), the reduction is
reported as `null` — never as an invented value.

### How claims are evaluated — two tiers, never conflated

- **automated_heuristic_llm_judge (not ground truth):** claims are extracted
  by a documented heuristic (sentence split + filters), then judged in batch
  by the same configured LLM at temperature 0 with strict structured output.
  Judge rules explicitly forbid treating citation markers or topic overlap as
  support.
- **manual_verification:** only a human auditing the preserved per-claim
  records can certify numbers as verified. Every result file carries its
  `evaluation_tier`, so automated output can never silently pass as verified.

### Dataset

`backend/eval/benchmark_dataset.json` (v1.1.0) — 20 factual research
questions across 10 domains (health, climate, energy, transport, economy,
education, agriculture, technology, environment, demographics). Every
question is `source_curation: "curated"`: it pins exact authoritative
documents (WHO fact sheets, NOAA GML trend pages, World Bank API indicators,
IPCC, UN, UNEP, Our World in Data) and **inlines verbatim evidence excerpts**
(`reference_sources[].content`) verified by live fetch during curation, so
the judge always evaluates claims against real evidence. The dataset is
deterministic and hash-pinned (`dataset_sha256` is stored in every result).
A question whose corpus cannot be established is excluded and its metric is
left unpopulated — it is never scored against nothing.

### Running it

```bash
cd backend
python -m eval.benchmark --all                    # full ablation (live; uses quota)
python -m eval.benchmark --variant baseline       # single variant
python -m eval.benchmark --all --runs 2           # repeat runs per question
python -m eval.benchmark --all --dataset my.json --output out.json
python -m eval.benchmark --all --mock             # offline smoke run — NOT a measurement

# Resumable execution (recommended for the full 20-question run):
python -m eval.benchmark --all --resume eval/results/hallucination-partial.json
```

With `--resume`, completed question×variant cells are loaded from the JSONL
and SKIPPED (never re-run, never re-billed), every newly finished row is
appended to the file incrementally as it completes, and if the provider's
daily token limit (TPD) is hit the run stops cleanly — exit code 2 — printing
the exact resume command. Re-run the identical command after the quota reset
to continue from where it stopped. Failed cells are re-run on resume;
successful ones never are. Per-minute (TPM) rate limits are cell-scoped: the
run continues, the affected cell is recorded as failed with no metric, and a
later `--resume` retries it. Rate-limit waits always honor the provider's
reset information and never exceed 60 seconds.

Every run stores a machine-readable JSON (timestamp, benchmark version, model,
temperatures, dataset sha256, variant, run index, raw answer/report, retrieved
source IDs, per-claim verdicts, counts, citation metrics, critic status,
iterations, latencies, errors) under `backend/eval/results/`. API keys are
never recorded. The legacy `python -m eval.benchmark` quality-snapshot mode is
unchanged.

### Current benchmark status

The harness, curated v1.1.0 dataset, and resume mechanism are complete and
tested (294 tests total across both suites cover them). The full 20-question
× 3-variant live run is **partially complete** (resumable — quota-limited)
and **no hallucination-reduction numbers have been published**: automated
LLM-judge rates are explicitly not ground truth, and per the protocol above
nothing is citable until the run finishes and the preserved per-claim
verdicts pass manual audit. Result files are gitignored run artifacts, not
shipped with the repository.

### Limitations of the automated protocol

The judge is the same LLM being measured — automated rates are estimates with
known bias, not ground truth. Claim extraction is lexical (it can merge or
miss claims). Judge verdicts depend on the curated evidence excerpts shipped
in the dataset; any number must still survive manual audit of the preserved
per-claim records before it is citable. **No hallucination-reduction claim is
made until the real benchmark run has completed and produced manually
verified measurements.**

## Testing

```bash
# Backend (252 tests; all external services mocked)
cd backend
.venv\Scripts\python -m pytest tests -q

# Frontend (42 tests) + production build
cd frontend
npm run test
npm run build
```

Every test suite runs fully offline — LLM calls, Tavily, and Qdrant are
mocked; the benchmark suite additionally covers resume semantics (skip
completed cells, re-run failed ones, clean TPD stop) and TPM/TPD rate-limit
classification. `ruff check eval agents/groq_structured.py` is the lint gate
for the benchmark/LLM path.

## Repository layout

```
├── docker-compose.yml          # Qdrant + backend + frontend stack
├── backend/
│   ├── main.py                 # FastAPI app: /query, /research/stream, /kb/*, /report/pdf
│   ├── config.py               # single source of env access
│   ├── pdf_export.py           # fpdf2 report rendering
│   ├── agents/                 # planner, retriever, writer, critic, orchestrator,
│   │                           #   groq_structured (LLM client), citations, events, schemas
│   ├── tools/                  # retrieval (Tavily + Qdrant), tavily_tool, qdrant_tool
│   ├── kb/                     # knowledge-base ingestion (chunk → embed → upsert)
│   ├── eval/                   # scorer, hallucination ablation harness, CLI,
│   │                           #   benchmark_dataset.json (v1.1.0, curated + inlined evidence)
│   └── tests/                  # 252 offline tests
└── frontend/
    ├── src/                    # React workspace: streaming research UI, KB manager,
    │                           #   history, evaluation dashboard, settings
    └── ...                     # Vite config, nginx.conf, Dockerfile, 42 tests
```

## Contributing

Issues and pull requests are welcome. Please keep the project's core
guarantees intact: deterministic retrieval, citation validation, and honest
evaluation labeling (heuristics are heuristics — never presented as ground
truth). Run both test suites before submitting.

## License

No license has been chosen yet — all rights reserved by default. If you
intend to reuse this code, open an issue to discuss licensing.

## Security notes

- API keys live only in local `.env` (gitignored); `.env.example` contains
  placeholders only.
- The frontend bundle contains no provider keys; `/runtime` exposes
  configuration as booleans, never values.
- Uploads are validated (extension allowlist, 10 MB cap, empty/scanned-PDF
  rejection); `document_id`s are content-derived, filenames never touch the
  filesystem.
- Report markdown renders via react-markdown with no raw-HTML plugin, so
  remote markup cannot execute; web-source links use
  `rel="noopener noreferrer nofollow"`.
- PDF generation renders only sanitized plain text through a fixed
  whitelist of layout operations.

## Limitations

- Coverage/grounding metrics are lexical heuristics, not semantic truth.
- Groq free-tier rate limits are respected with bounded backoff, but large
  runs can still hit quota; failures surface as safe user-facing errors.
- Retrieval quality depends on Tavily and the local embedding model;
  there is no reranker.
- The KB is a single global collection without per-user isolation.
- PDF parsing cannot handle scanned/image-only PDFs (no OCR).
- Evaluation history is stored per browser (localStorage), not on a server.

## Implementation status

| Component | Status |
| --- | --- |
| Planner / Retriever / Writer / Critic + revision loop | ✅ |
| Deterministic citation validation + references | ✅ |
| SSE streaming + React UI | ✅ |
| Retrieval modes (both / web / kb) | ✅ |
| KB ingestion (PDF/TXT/MD) + list/delete/status | ✅ |
| Final `/query` endpoint + evaluation + timing | ✅ |
| KB UI, History, Evaluation dashboard, Settings | ✅ |
| Markdown + PDF export, benchmark harness | ✅ |
| Dockerized full stack (Qdrant + FastAPI + nginx) | ✅ |
| Authentication / multi-user | ⏳ Not planned for v1 |
