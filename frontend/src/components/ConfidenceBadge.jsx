import { formatPercent } from "../utils/formatting";

const LEVELS = [
  {
    min: 0.85,
    label: "High grounding",
    classes: "bg-emerald-100 text-emerald-700",
    bar: "bg-emerald-500",
  },
  {
    min: 0.65,
    label: "Medium grounding",
    classes: "bg-amber-100 text-amber-700",
    bar: "bg-amber-500",
  },
  {
    min: 0,
    label: "Low grounding",
    classes: "bg-red-100 text-red-700",
    bar: "bg-red-400",
  },
];

/**
 * Evidence-grounding indicator (heuristic).
 *
 * Derived from citation coverage computed by the backend evaluation module —
 * the share of claim sentences carrying a valid inline citation. It is a
 * mechanical grounding signal, NOT model confidence and not a truth measure.
 */
export default function ConfidenceBadge({ coverage }) {
  if (typeof coverage !== "number" || Number.isNaN(coverage)) {
    return null;
  }
  const level = LEVELS.find((entry) => coverage >= entry.min) ?? LEVELS[2];
  return (
    <div
      className="inline-flex items-center gap-2"
      title="Heuristic: share of claim sentences with a valid citation. Not model confidence."
    >
      <span
        className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold ${level.classes}`}
      >
        {level.label}
      </span>
      <span
        className="hidden h-1.5 w-24 overflow-hidden rounded-full bg-slate-200 sm:block"
        role="img"
        aria-label={`Citation coverage ${formatPercent(coverage)}`}
      >
        <span
          className={`block h-full rounded-full ${level.bar}`}
          style={{ width: `${Math.round(Math.min(1, Math.max(0, coverage)) * 100)}%` }}
        />
      </span>
      <span className="text-xs text-slate-500">
        {formatPercent(coverage)} coverage
      </span>
    </div>
  );
}
