import { useEffect, useRef } from "react";
import EmptyState from "./EmptyState";
import { truncate } from "../utils/formatting";

/**
 * Numbered evidence panel. Source N corresponds to citation [N]: the
 * backend assigns citation IDs in bundle order (1-based), which this
 * component mirrors. Web sources open in a new tab; KB sources show the
 * document name (they have no URL).
 */
export default function SourcePanel({
  sources,
  citedIds = [],
  activeSource = null,
  onSelectSource,
}) {
  const listRef = useRef(null);
  const itemRefs = useRef({});

  useEffect(() => {
    const node = itemRefs.current[activeSource];
    if (activeSource != null && node?.scrollIntoView) {
      node.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [activeSource]);

  if (!sources?.length) {
    return (
      <EmptyState
        title="No evidence sources yet"
        hint="Run a research query and retrieved sources will appear here."
      />
    );
  }

  const cited = new Set(citedIds);

  return (
    <section aria-label="Supporting evidence">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-700">
          Supporting evidence
        </h2>
        <span className="text-xs text-slate-400">
          {sources.length} source{sources.length === 1 ? "" : "s"}
        </span>
      </div>
      <ol ref={listRef} className="space-y-2">
        {sources.map((source, index) => {
          const number = index + 1;
          const isActive = activeSource === number;
          return (
            <li
              key={`${number}-${source.source_id ?? source.url ?? index}`}
              ref={(node) => {
                itemRefs.current[number] = node;
              }}
              className={`rounded-lg border px-4 py-3 transition ${
                isActive
                  ? "border-indigo-300 bg-indigo-50 ring-2 ring-indigo-200"
                  : "border-slate-200 bg-white"
              }`}
            >
              <div className="flex items-start gap-3">
                <button
                  type="button"
                  onClick={() => onSelectSource?.(number)}
                  aria-label={`Source ${number}: ${
                    source.title || source.document_name || "untitled"
                  }`}
                  className={`mt-0.5 inline-flex h-6 min-w-6 shrink-0 items-center justify-center rounded-md px-1.5 text-xs font-bold focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 ${
                    cited.has(number)
                      ? "bg-indigo-100 text-indigo-700 hover:bg-indigo-200"
                      : "bg-slate-100 text-slate-400 hover:bg-slate-200"
                  }`}
                >
                  {number}
                </button>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span
                      className={`rounded px-1.5 py-0.5 text-[10px] font-bold tracking-wide ${
                        source.source_type === "kb"
                          ? "bg-teal-100 text-teal-700"
                          : "bg-sky-100 text-sky-700"
                      }`}
                    >
                      {source.source_type === "kb" ? "KB" : "WEB"}
                    </span>
                    <p className="min-w-0 flex-1 truncate text-sm font-medium text-slate-800">
                      {source.title ||
                        source.document_name ||
                        source.url ||
                        "Untitled source"}
                    </p>
                    {typeof source.score === "number" ? (
                      <span
                        className="shrink-0 text-[11px] text-slate-400"
                        title="Retrieval relevance score"
                      >
                        {source.score.toFixed(2)}
                      </span>
                    ) : null}
                  </div>
                  {source.source_type === "web" && source.url ? (
                    <a
                      href={source.url}
                      target="_blank"
                      rel="noopener noreferrer nofollow"
                      className="mt-0.5 block truncate text-xs text-indigo-600 underline decoration-indigo-200 underline-offset-2 hover:text-indigo-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400"
                    >
                      {source.url}
                    </a>
                  ) : null}
                  {source.source_type === "kb" && source.document_name ? (
                    <p className="mt-0.5 truncate text-xs text-slate-500">
                      Document: {source.document_name}
                    </p>
                  ) : null}
                  {source.preview ? (
                    <p className="mt-1 text-xs leading-5 text-slate-500">
                      {truncate(source.preview)}
                    </p>
                  ) : null}
                </div>
              </div>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
