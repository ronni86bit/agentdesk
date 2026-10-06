import { useCallback, useMemo, useState } from "react";
import Layout from "./components/Layout";
import QueryInput from "./components/QueryInput";
import RetrievalModeToggle from "./components/RetrievalModeToggle";
import AgentProgress from "./components/AgentProgress";
import ReportView from "./components/ReportView";
import SourcePanel from "./components/SourcePanel";
import ConfidenceBadge from "./components/ConfidenceBadge";
import RunSummary from "./components/RunSummary";
import HistoryPanel from "./components/HistoryPanel";
import EvalDashboard from "./components/EvalDashboard";
import KnowledgeBase from "./components/KnowledgeBase";
import SettingsView from "./components/SettingsView";
import DownloadButtons from "./components/DownloadButtons";
import ErrorBanner from "./components/ErrorBanner";
import EmptyState from "./components/EmptyState";
import LoadingState from "./components/LoadingState";
import { useResearchStream } from "./hooks/useResearchStream";
import { useLocalHistory } from "./hooks/useLocalHistory";

function Section({ title, children }) {
  return (
    <section className="space-y-2">
      {title ? (
        <h2 className="text-sm font-semibold text-slate-700">{title}</h2>
      ) : null}
      {children}
    </section>
  );
}

/**
 * Build a ResearchRunResult-shaped object from a saved history record so
 * restored runs render through the exact same components as live runs.
 */
function resultFromHistory(record) {
  return {
    query: record.query,
    report: {
      title: record.title || record.query,
      markdown: record.reportMarkdown ?? "",
      references: record.references ?? [],
    },
    evidence: {
      sources: [],
      retrieval_summary: record.retrievalSummary ?? {
        total_sources: null,
        web_sources: null,
        kb_sources: null,
      },
    },
    evaluation: record.evaluation ?? null,
    timing: record.timing ?? null,
    iteration_count: record.iterationCount ?? 0,
    final_status: record.finalStatus,
    retrieval_mode: record.retrievalMode ?? "both",
  };
}

function ResearchView({
  stream,
  history,
  restoredRecord,
  onClearRestored,
  onNavigate,
}) {
  const {
    startResearch,
    stopResearch,
    status,
    stages,
    planInfo,
    retrieval,
    writerInfo,
    criticInfo,
    revision,
    result,
    report,
    error,
    isRunning,
  } = stream;

  const [selectedMode, setSelectedMode] = useState("both");
  const [activeSource, setActiveSource] = useState(null);
  const [savedId, setSavedId] = useState(null);

  const handleStart = useCallback(
    (nextQuery) => {
      onClearRestored();
      setSavedId(null);
      setActiveSource(null);
      startResearch(nextQuery, selectedMode);
    },
    [onClearRestored, selectedMode, startResearch],
  );

  const restored = useMemo(() => {
    if (!restoredRecord || isRunning || result) return null;
    return {
      record: restoredRecord,
      result: resultFromHistory(restoredRecord),
      report: {
        title: restoredRecord.title || restoredRecord.query,
        markdown: restoredRecord.reportMarkdown ?? "",
      },
      sources: restoredRecord.sources ?? [],
    };
  }, [restoredRecord, isRunning, result]);

  const displayResult = restored?.result ?? result;
  const displayReport = restored?.report ?? report;
  const displaySources = restored
    ? restored.sources
    : (result?.evidence?.sources ?? []).map((source) => ({
        ...source,
        preview: source.content ?? "",
      }));
  const citedIds = (
    displayResult?.report?.references ?? []
  ).map((ref) => ref.citation_id);

  const handleSave = useCallback(() => {
    if (!result) return;
    const record = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      query: result.query,
      timestamp: result.evaluation?.timestamp ?? new Date().toISOString(),
      title: result.report?.title ?? result.query,
      finalStatus: result.final_status,
      iterationCount: result.iteration_count,
      retrievalMode: result.retrieval_mode ?? selectedMode,
      evaluation: result.evaluation ?? null,
      timing: result.timing ?? null,
      retrievalSummary: result.evidence?.retrieval_summary ?? null,
      reportMarkdown: result.report?.markdown ?? "",
      references: result.report?.references ?? [],
      sources: (result.evidence?.sources ?? []).slice(0, 40).map((source) => ({
        source_type: source.source_type,
        title: source.title,
        url: source.url,
        document_name: source.document_name,
        score: source.score,
        preview: (source.content ?? "").slice(0, 280),
      })),
    };
    history.addRun(record);
    setSavedId(record.id);
  }, [history, result, selectedMode]);

  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-5 lg:col-span-2">
          <QueryInput
            onSubmit={handleStart}
            onStop={stopResearch}
            isRunning={isRunning}
          />
        </div>
        <div className="space-y-5">
          <RetrievalModeToggle
            value={selectedMode}
            onChange={setSelectedMode}
            disabled={isRunning}
          />
        </div>
      </div>

      {restoredRecord ? (
        <p className="flex flex-wrap items-center gap-3 rounded-lg border border-slate-200 bg-slate-100 px-4 py-2 text-xs text-slate-500">
          <span>
            Viewing a saved run from history
            {restoredRecord.timestamp
              ? ` (${new Date(restoredRecord.timestamp).toLocaleString()})`
              : ""}
            .
          </span>
          <button
            type="button"
            onClick={onClearRestored}
            className="font-medium text-indigo-600 underline-offset-2 hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400"
          >
            Back to new research
          </button>
        </p>
      ) : null}

      <ErrorBanner message={error} />

      <AgentProgress
        state={{ status, stages, planInfo, retrieval, writerInfo, criticInfo, revision }}
      />

      {isRunning && !report ? (
        <LoadingState label="Researching — planner, retriever, writer, and critic are working…" />
      ) : null}

      {displayResult ? (
        <div className="space-y-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <ConfidenceBadge
              coverage={displayResult.evaluation?.citation_coverage}
            />
            <div className="flex flex-wrap items-center gap-2">
              <DownloadButtons result={displayResult} />
              {!restored ? (
                <button
                  type="button"
                  onClick={handleSave}
                  disabled={savedId !== null}
                  className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 shadow-sm transition hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {savedId ? "✓ Saved to history" : "Save to history"}
                </button>
              ) : null}
            </div>
          </div>

          <Section title="Final report">
            <ReportView
              report={displayReport}
              activeSource={activeSource}
              onSelectSource={setActiveSource}
            />
          </Section>

          <Section title="Sources">
            <SourcePanel
              sources={displaySources}
              citedIds={citedIds}
              activeSource={activeSource}
              onSelectSource={(id) =>
                setActiveSource((current) => (current === id ? null : id))
              }
            />
          </Section>

          <RunSummary
            result={displayResult}
            evaluation={displayResult.evaluation}
            timing={displayResult.timing}
            retrieval={retrieval}
          />
        </div>
      ) : !isRunning && !error ? (
        <EmptyState
          title="Start a research run"
          hint="Enter a research goal above, choose where to retrieve evidence from, and the agent pipeline will plan, retrieve, write, and audit a grounded, cited report."
        >
          <button
            type="button"
            onClick={() => onNavigate("kb")}
            className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 shadow-sm transition hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400"
          >
            Add documents to your knowledge base first
          </button>
        </EmptyState>
      ) : null}
    </div>
  );
}

export default function App() {
  const [view, setView] = useState("research");
  const [restoredRecord, setRestoredRecord] = useState(null);
  const stream = useResearchStream();
  const history = useLocalHistory();

  const navigate = useCallback((id) => setView(id), []);

  const viewHistoryItem = useCallback((item) => {
    if (item) {
      setRestoredRecord(item);
      setView("research");
      window.scrollTo({ top: 0 });
    } else {
      setView("history");
    }
  }, []);

  return (
    <Layout view={view} onNavigate={navigate} status={stream.status}>
      {view === "research" ? (
        <ResearchView
          stream={stream}
          history={history}
          restoredRecord={restoredRecord}
          onClearRestored={() => setRestoredRecord(null)}
          onNavigate={navigate}
        />
      ) : null}
      {view === "kb" ? <KnowledgeBase /> : null}
      {view === "history" ? (
        <HistoryPanel
          history={history.history}
          onView={viewHistoryItem}
          onDelete={history.deleteRun}
          onClear={history.clearHistory}
        />
      ) : null}
      {view === "evaluation" ? (
        <EvalDashboard history={history.history} />
      ) : null}
      {view === "settings" ? <SettingsView /> : null}
    </Layout>
  );
}
