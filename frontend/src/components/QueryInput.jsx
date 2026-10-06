import { useId, useRef, useState } from "react";

const MAX_QUERY_CHARS = 2000;

export default function QueryInput({
  onSubmit,
  isRunning = false,
  disabled = false,
  onStop,
}) {
  const [value, setValue] = useState("");
  const inputId = useId();
  const submittedRef = useRef(false);

  const trimmed = value.trim();
  const canSubmit = !disabled && !isRunning && trimmed.length > 0;

  function handleSubmit() {
    if (!canSubmit) return;
    submittedRef.current = true;
    onSubmit?.(trimmed);
  }

  function handleKeyDown(event) {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      handleSubmit();
    }
  }

  function handleClear() {
    setValue("");
  }

  return (
    <section aria-label="Research goal" className="space-y-3">
      <label
        htmlFor={inputId}
        className="block text-sm font-semibold text-slate-700"
      >
        Research goal
      </label>
      <textarea
        id={inputId}
        value={value}
        onChange={(event) => setValue(event.target.value.slice(0, MAX_QUERY_CHARS))}
        onKeyDown={handleKeyDown}
        rows={4}
        placeholder="e.g. Research the impact of AI on healthcare in India"
        disabled={disabled}
        className="w-full resize-y rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm text-slate-900 placeholder:text-slate-400 shadow-sm transition focus:border-indigo-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-300 disabled:cursor-not-allowed disabled:bg-slate-50 disabled:text-slate-400"
      />
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          {isRunning ? (
            <button
              type="button"
              onClick={onStop}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400"
            >
              Stop
            </button>
          ) : (
            <button
              type="button"
              onClick={handleSubmit}
              disabled={!canSubmit}
              className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-5 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500"
            >
              Research
            </button>
          )}
          {value && !isRunning ? (
            <button
              type="button"
              onClick={handleClear}
              className="rounded-lg px-2 py-2 text-sm text-slate-500 transition hover:text-slate-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400"
            >
              Clear
            </button>
          ) : null}
        </div>
        <p className="text-xs text-slate-400">
          <kbd className="rounded border border-slate-300 bg-slate-50 px-1 font-sans">
            Ctrl
          </kbd>{" "}
          +{" "}
          <kbd className="rounded border border-slate-300 bg-slate-50 px-1 font-sans">
            Enter
          </kbd>{" "}
          to submit · {value.length}/{MAX_QUERY_CHARS}
        </p>
      </div>
    </section>
  );
}
