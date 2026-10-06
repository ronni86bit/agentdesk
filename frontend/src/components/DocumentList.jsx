import EmptyState from "./EmptyState";
import { formatDate, formatBytes } from "../utils/formatting";

const TYPE_LABELS = {
  pdf: "PDF",
  txt: "TXT",
  md: "MD",
};

export default function DocumentList({ documents, busy = false, onDelete }) {
  if (!documents.length) {
    return (
      <EmptyState
        title="No documents yet"
        hint="Upload a PDF, text, or markdown file to build your knowledge base."
      />
    );
  }

  return (
    <section aria-label="Indexed documents" className="space-y-2">
      <h2 className="text-sm font-semibold text-slate-700">
        Documents ({documents.length})
      </h2>
      <ul className="divide-y divide-slate-100 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
        {documents.map((doc) => (
          <li
            key={doc.document_id}
            className="flex flex-wrap items-center gap-3 px-4 py-3"
          >
            <span className="rounded bg-teal-100 px-1.5 py-0.5 text-[10px] font-bold tracking-wide text-teal-700">
              {TYPE_LABELS[doc.file_type] ?? doc.file_type?.toUpperCase() ?? "DOC"}
            </span>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium text-slate-800">
                {doc.document_name}
              </p>
              <p className="text-xs text-slate-400">
                {doc.chunk_count} chunks · {formatDate(doc.created_at)}
                {formatBytes(doc.size_bytes) ? ` · ${formatBytes(doc.size_bytes)}` : ""}
              </p>
            </div>
            <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-semibold text-emerald-700">
              {doc.status ?? "indexed"}
            </span>
            <button
              type="button"
              disabled={busy}
              onClick={() => onDelete?.(doc.document_id)}
              aria-label={`Delete ${doc.document_name}`}
              className="rounded-lg border border-slate-200 px-2.5 py-1 text-xs font-medium text-red-600 transition hover:bg-red-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-300 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Delete
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
