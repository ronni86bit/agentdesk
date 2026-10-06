import EmptyState from "./EmptyState";
import { formatDate } from "../utils/formatting";

const CRITIC_LABELS = {
  approved: "Approved",
  revision_needed: "Revision needed",
};

function Bar({ value }) {
  const pct = Math.round(Math.min(1, Math.max(0, value ?? 0)) * 100);
  return (
    <span className="inline-flex items-center gap-2">
      <span className="h-1.5 w-20 overflow-hidden rounded-full bg-slate-200">
        <span
          className="block h-full rounded-full bg-indigo-500"
          style={{ width: `${pct}%` }}
        />
      </span>
      <span className="tabular-nums">{(value ?? 0).toFixed(2)}</span>
    </span>
  );
}

/**
 * Evaluation dashboard over locally saved runs (localStorage only).
 * All metrics are documented heuristics from the backend eval module.
 */
export default function EvalDashboard({ history }) {
  const evaluated = history.filter((item) => item.evaluation);
  if (!evaluated.length) {
    return (
      <div className="space-y-4">
        <h1 className="text-xl font-bold tracking-tight text-slate-900">
          Evaluation
        </h1>
        <EmptyState
          title="No evaluation data yet"
          hint="Save completed research runs to compare grounding metrics across queries."
        />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <header className="space-y-1">
        <h1 className="text-xl font-bold tracking-tight text-slate-900">
          Evaluation
        </h1>
        <p className="text-sm text-slate-500">
          Deterministic heuristic metrics over your saved runs ({evaluated.length}).
          These measure mechanical grounding signals — not factual accuracy.
        </p>
      </header>
      <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white shadow-sm">
        <table className="w-full min-w-[760px] text-sm">
          <caption className="sr-only">
            Saved research runs with evaluation metrics
          </caption>
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50 text-left text-[11px] uppercase tracking-wide text-slate-500">
              <th scope="col" className="px-4 py-2.5 font-semibold">Query</th>
              <th scope="col" className="px-3 py-2.5 font-semibold">Date</th>
              <th scope="col" className="px-3 py-2.5 font-semibold">Grounding</th>
              <th scope="col" className="px-3 py-2.5 font-semibold">Coverage</th>
              <th scope="col" className="px-3 py-2.5 font-semibold">Critic</th>
              <th scope="col" className="px-3 py-2.5 font-semibold">Iter.</th>
              <th scope="col" className="px-3 py-2.5 font-semibold">Sources</th>
              <th scope="col" className="px-3 py-2.5 font-semibold">Web/KB</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {evaluated.map((item) => {
              const ev = item.evaluation;
              return (
                <tr key={item.id} className="text-slate-700">
                  <td
                    className="max-w-[220px] truncate px-4 py-2.5 font-medium text-slate-800"
                    title={item.query}
                  >
                    {item.query}
                  </td>
                  <td className="whitespace-nowrap px-3 py-2.5 text-xs text-slate-500">
                    {formatDate(item.timestamp)}
                  </td>
                  <td className="px-3 py-2.5">
                    <Bar value={ev.grounding_score} />
                  </td>
                  <td className="px-3 py-2.5 tabular-nums">
                    {ev.citation_coverage != null
                      ? `${Math.round(ev.citation_coverage * 100)}%`
                      : "—"}
                  </td>
                  <td className="px-3 py-2.5">
                    <span
                      className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${
                        ev.critic_status === "approved"
                          ? "bg-emerald-100 text-emerald-700"
                          : "bg-amber-100 text-amber-700"
                      }`}
                    >
                      {CRITIC_LABELS[ev.critic_status] ?? ev.critic_status}
                    </span>
                  </td>
                  <td className="px-3 py-2.5 tabular-nums">
                    {item.iterationCount ?? ev.iteration_count ?? 0}
                  </td>
                  <td className="px-3 py-2.5 tabular-nums">
                    {ev.total_sources ?? "—"}
                  </td>
                  <td className="px-3 py-2.5 tabular-nums text-xs">
                    {ev.web_percentage != null && ev.kb_percentage != null
                      ? `${Math.round(ev.web_percentage)}/${Math.round(ev.kb_percentage)}`
                      : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
