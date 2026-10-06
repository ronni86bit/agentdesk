import { formatDuration, formatPercent } from "../utils/formatting";

function Metric({ label, value }) {
  return (
    <div className="min-w-0">
      <dt className="text-[11px] font-medium uppercase tracking-wide text-slate-400">
        {label}
      </dt>
      <dd className="truncate text-sm font-semibold text-slate-800">
        {value ?? "—"}
      </dd>
    </div>
  );
}

const FINAL_STATUS_LABELS = {
  approved: "Approved",
  max_revisions_reached: "Max revisions reached",
  revision_failed: "Revision failed",
};

export default function RunSummary({ result, evaluation, timing, retrieval }) {
  if (!result) return null;

  const evidence = result.evidence;
  const summary = evaluation ?? {
    total_sources: evidence?.retrieval_summary?.total_sources,
    web_sources: evidence?.retrieval_summary?.web_sources,
    kb_sources: evidence?.retrieval_summary?.kb_sources,
  };

  return (
    <section aria-label="Run summary" className="space-y-2">
      <h2 className="text-sm font-semibold text-slate-700">Run summary</h2>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm sm:grid-cols-3 lg:grid-cols-4">
        <Metric label="Sources" value={summary.total_sources} />
        <Metric label="Web" value={summary.web_sources} />
        <Metric label="KB" value={summary.kb_sources} />
        <Metric
          label="Citations"
          value={result.report?.references?.length ?? result.report?.markdown?.match(/\[\d+\]/g)?.length ?? 0}
        />
        <Metric
          label="Coverage"
          value={
            typeof evaluation?.citation_coverage === "number"
              ? formatPercent(evaluation.citation_coverage)
              : undefined
          }
        />
        <Metric
          label="Critic"
          value={
            evaluation?.critic_status === "approved"
              ? "Approved"
              : evaluation?.critic_status === "revision_needed"
                ? "Revision needed"
                : evaluation?.critic_status
          }
        />
        <Metric label="Iterations" value={result.iteration_count} />
        <Metric
          label="Final status"
          value={FINAL_STATUS_LABELS[result.final_status] ?? result.final_status}
        />
        {typeof evaluation?.web_percentage === "number" ? (
          <Metric
            label="Web share"
            value={`${evaluation.web_percentage.toFixed(0)}%`}
          />
        ) : null}
        {typeof evaluation?.kb_percentage === "number" ? (
          <Metric
            label="KB share"
            value={`${evaluation.kb_percentage.toFixed(0)}%`}
          />
        ) : null}
        {typeof evaluation?.grounding_score === "number" ? (
          <Metric
            label="Grounding score"
            value={evaluation.grounding_score.toFixed(2)}
          />
        ) : null}
        <Metric
          label="Runtime"
          value={
            timing?.total_seconds != null
              ? formatDuration(timing.total_seconds)
              : undefined
          }
        />
      </dl>
      {retrieval && (retrieval.kbStatus === "unavailable" || retrieval.webStatus === "unavailable") ? (
        <p className="text-xs text-amber-600">
          {retrieval.kbStatus === "unavailable" ? "Knowledge base unavailable. " : ""}
          {retrieval.webStatus === "unavailable" ? "Web search unavailable." : ""}
        </p>
      ) : null}
    </section>
  );
}
