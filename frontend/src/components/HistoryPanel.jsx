import EmptyState from "./EmptyState";
import { formatDate } from "../utils/formatting";

function EvaluationChip({ label, value }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-medium text-slate-600">
      {label}: <span className="font-semibold text-slate-800">{value}</span>
    </span>
  );
}

/**
 * Local research history. Records live in localStorage only — they never
 * contain API keys or backend secrets.
 */
export default function HistoryPanel({
  history,
  onView,
  onDelete,
  onClear,
}) {
  if (!history.length) {
    return (
      <EmptyState
        title="No saved research yet"
        hint="Completed runs you save from the Research page appear here, stored only in this browser."
      />
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h1 className="text-xl font-bold tracking-tight text-slate-900">
          History
        </h1>
        <button
          type="button"
          onClick={onClear}
          className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-red-600 shadow-sm transition hover:bg-red-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-300"
        >
          Clear history
        </button>
      </div>
      <ul className="space-y-3">
        {history.map((item) => (
          <li
            key={item.id}
            className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm"
          >
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0 flex-1">
                <p className="text-sm font-semibold text-slate-800">
                  {item.title || item.query}
                </p>
                <p className="mt-0.5 truncate text-xs text-slate-400">
                  {item.query} · {formatDate(item.timestamp)} ·{" "}
                  {item.finalStatus ?? "unknown"}
                </p>
                {item.evaluation ? (
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    <EvaluationChip
                      label="Grounding"
                      value={item.evaluation.grounding_score?.toFixed?.(2) ?? "—"}
                    />
                    <EvaluationChip
                      label="Coverage"
                      value={
                        item.evaluation.citation_coverage != null
                          ? `${Math.round(item.evaluation.citation_coverage * 100)}%`
                          : "—"
                      }
                    />
                    <EvaluationChip
                      label="Critic"
                      value={
                        item.evaluation.critic_status === "approved"
                          ? "Approved"
                          : item.evaluation.critic_status ?? "—"
                      }
                    />
                    <EvaluationChip
                      label="Sources"
                      value={item.evaluation.total_sources ?? "—"}
                    />
                  </div>
                ) : null}
              </div>
              <div className="flex shrink-0 gap-2">
                <button
                  type="button"
                  onClick={() => onView?.(item)}
                  className="rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-indigo-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400"
                >
                  View report
                </button>
                <button
                  type="button"
                  onClick={() => onDelete?.(item.id)}
                  aria-label={`Delete history item: ${item.title || item.query}`}
                  className="rounded-lg border border-slate-200 px-2.5 py-1.5 text-xs font-medium text-red-600 transition hover:bg-red-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-300"
                >
                  Delete
                </button>
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
