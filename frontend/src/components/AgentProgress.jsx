import { STAGES, STAGE_LABELS } from "../hooks/eventState";

const STAGE_ICONS = {
  waiting: { glyph: "○", classes: "text-slate-300" },
  running: { glyph: "↻", classes: "animate-spin text-indigo-500" },
  completed: { glyph: "✓", classes: "text-emerald-500" },
  failed: { glyph: "✕", classes: "text-red-500" },
};

function StageDetail({ stage, state }) {
  switch (stage) {
    case "planner":
      return state.planInfo?.subQuestions
        ? `${state.planInfo.subQuestions} sub-questions`
        : null;
    case "retriever":
      if (!state.retrieval) return null;
      return `${state.retrieval.total} sources · ${state.retrieval.web} Web / ${state.retrieval.kb} KB`;
    case "writer":
      return state.writerInfo?.citations != null
        ? `${state.writerInfo.citations} citations`
        : null;
    case "critic": {
      if (!state.criticInfo) return null;
      if (state.criticInfo.verdict === "approved") return "Approved";
      if (state.criticInfo.verdict === "revision_needed") {
        return state.criticInfo.issues
          ? `Revision needed · ${state.criticInfo.issues} issues`
          : "Revision needed";
      }
      return state.criticInfo.verdict;
    }
    default:
      return null;
  }
}

export default function AgentProgress({ state }) {
  const anyActivity =
    state.status === "running" ||
    STAGES.some((stage) => state.stages[stage] !== "waiting");
  if (!anyActivity) return null;

  return (
    <section aria-label="Agent pipeline" className="space-y-2">
      <h2 className="text-sm font-semibold text-slate-700">Agent pipeline</h2>
      <ol className="divide-y divide-slate-100 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
        {STAGES.map((stage) => {
          const stageState = state.stages[stage] ?? "waiting";
          const icon = STAGE_ICONS[stageState] ?? STAGE_ICONS.waiting;
          const detail = StageDetail({ stage, state });
          return (
            <li
              key={stage}
              className="flex items-center gap-3 px-4 py-2.5 text-sm"
            >
              <span
                aria-hidden="true"
                className={`w-5 text-center text-base ${icon.classes}`}
              >
                {icon.glyph}
              </span>
              <span className="font-medium text-slate-800">
                {STAGE_LABELS[stage]}
              </span>
              <span className="sr-only">{stageState}</span>
              {detail ? (
                <span className="ml-auto text-xs text-slate-500">{detail}</span>
              ) : null}
            </li>
          );
        })}
      </ol>
      {state.revision?.active ? (
        <p className="flex items-center gap-2 px-1 text-xs font-medium text-amber-600">
          <span aria-hidden="true">↻</span>
          Revision {state.revision.iteration ?? ""} in progress
        </p>
      ) : state.revision && !state.revision.active ? (
        <p className="px-1 text-xs text-slate-500">
          Revision {state.revision.iteration} completed
        </p>
      ) : null}
    </section>
  );
}
