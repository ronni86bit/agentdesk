import StatusHeader from "./StatusHeader";

const NAV_ITEMS = [
  { id: "research", label: "Research" },
  { id: "kb", label: "Knowledge Base" },
  { id: "history", label: "History" },
  { id: "evaluation", label: "Evaluation" },
  { id: "settings", label: "Settings" },
];

/**
 * Application shell: brand/status header + primary navigation + content.
 */
export default function Layout({ view, onNavigate, status, children }) {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <StatusHeader status={status} />
      <nav
        aria-label="Primary"
        className="border-b border-slate-200 bg-white/80 backdrop-blur"
      >
        <div className="mx-auto flex max-w-6xl gap-1 overflow-x-auto px-4 sm:px-6">
          {NAV_ITEMS.map((item) => {
            const selected = view === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => onNavigate(item.id)}
                aria-current={selected ? "page" : undefined}
                className={`relative whitespace-nowrap px-3 py-2.5 text-sm font-medium transition focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-indigo-400 ${
                  selected
                    ? "text-indigo-700"
                    : "text-slate-500 hover:text-slate-900"
                }`}
              >
                {item.label}
                {selected ? (
                  <span
                    aria-hidden="true"
                    className="absolute inset-x-2 bottom-0 h-0.5 rounded-full bg-indigo-600"
                  />
                ) : null}
              </button>
            );
          })}
        </div>
      </nav>
      <main className="mx-auto w-full max-w-6xl px-4 py-6 sm:px-6">
        {children}
      </main>
      <footer className="mx-auto max-w-6xl px-4 pb-8 pt-2 text-center text-xs text-slate-400 sm:px-6">
        AgentDesk · multi-agent research assistant · grounding metrics are
        heuristics, not measures of factual accuracy
      </footer>
    </div>
  );
}
