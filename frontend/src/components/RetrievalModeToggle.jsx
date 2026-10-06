const MODES = [
  { value: "both", label: "Web + Knowledge Base" },
  { value: "web", label: "Web only" },
  { value: "kb", label: "Knowledge Base only" },
];

export default function RetrievalModeToggle({ value, onChange, disabled = false }) {
  return (
    <fieldset className="min-w-0">
      <legend className="mb-1.5 text-xs font-medium uppercase tracking-wide text-slate-400">
        Retrieval mode
      </legend>
      <div
        role="radiogroup"
        aria-label="Retrieval mode"
        className="inline-flex max-w-full overflow-x-auto rounded-lg border border-slate-300 bg-slate-50 p-0.5"
      >
        {MODES.map((mode) => {
          const selected = value === mode.value;
          return (
            <button
              key={mode.value}
              type="button"
              role="radio"
              aria-checked={selected}
              disabled={disabled}
              onClick={() => onChange?.(mode.value)}
              className={`whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium transition focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 disabled:cursor-not-allowed disabled:opacity-50 ${
                selected
                  ? "bg-white text-slate-900 shadow-sm"
                  : "text-slate-500 hover:text-slate-800"
              }`}
            >
              {mode.label}
            </button>
          );
        })}
      </div>
    </fieldset>
  );
}
