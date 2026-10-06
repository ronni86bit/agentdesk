import { useCallback, useEffect, useRef, useState } from "react";
import {
  deleteJSON,
  getJSON,
  uploadFile,
  ApiError,
} from "../services/api";
import DocumentList from "./DocumentList";
import ErrorBanner from "./ErrorBanner";
import LoadingState from "./LoadingState";

const ALLOWED_EXTENSIONS = [".pdf", ".txt", ".md"];

export default function KnowledgeBase() {
  const [documents, setDocuments] = useState(null);
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [notice, setNotice] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const refresh = useCallback(async () => {
    try {
      setError(null);
      const [docs, kbStatus] = await Promise.all([
        getJSON("/kb/documents"),
        getJSON("/kb/status"),
      ]);
      setDocuments(docs.documents ?? []);
      setStatus(kbStatus);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Could not load knowledge base state.",
      );
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const upload = useCallback(
    async (file) => {
      if (!file) return;
      const lower = file.name.toLowerCase();
      if (!ALLOWED_EXTENSIONS.some((ext) => lower.endsWith(ext))) {
        setError(
          `Unsupported file type. Allowed: ${ALLOWED_EXTENSIONS.join(", ")}`,
        );
        return;
      }
      setUploading(true);
      setNotice(null);
      setError(null);
      try {
        const result = await uploadFile("/kb/upload", file);
        setNotice(
          result.status === "replaced"
            ? `Replaced "${file.name}" — ${result.document.chunk_count} chunks`
            : `Indexed successfully — ${result.document.chunk_count} chunks`,
        );
        await refresh();
      } catch (err) {
        setError(
          err instanceof ApiError ? err.message : "Upload failed. Please retry.",
        );
      } finally {
        setUploading(false);
        if (fileInputRef.current) fileInputRef.current.value = "";
      }
    },
    [refresh],
  );

  const handleDelete = useCallback(
    async (documentId) => {
      setBusy(true);
      setError(null);
      try {
        await deleteJSON(`/kb/documents/${encodeURIComponent(documentId)}`);
        setNotice("Document deleted.");
        await refresh();
      } catch (err) {
        setError(
          err instanceof ApiError ? err.message : "Delete failed. Please retry.",
        );
      } finally {
        setBusy(false);
      }
    },
    [refresh],
  );

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <h1 className="text-xl font-bold tracking-tight text-slate-900">
          Knowledge Base
        </h1>
        <p className="text-sm text-slate-500">
          Upload PDF, text, or markdown documents. They are chunked, embedded
          locally, and stored in Qdrant for retrieval.
        </p>
      </header>

      <ErrorBanner message={error} onDismiss={() => setError(null)} />
      {notice ? (
        <p
          role="status"
          className="rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-800"
        >
          {notice}
        </p>
      ) : null}

      <section
        aria-label="Upload document"
        className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"
      >
        <label
          htmlFor="kb-file-input"
          className="block text-sm font-semibold text-slate-700"
        >
          Upload document
        </label>
        <p className="mt-0.5 text-xs text-slate-400">
          PDF, TXT, or MD · up to 10 MB · re-uploading identical content
          replaces the stored copy
        </p>
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <input
            ref={fileInputRef}
            id="kb-file-input"
            type="file"
            accept=".pdf,.txt,.md"
            disabled={uploading}
            onChange={(event) => upload(event.target.files?.[0])}
            className="block w-full max-w-sm text-sm text-slate-600 file:mr-3 file:cursor-pointer file:rounded-lg file:border-0 file:bg-indigo-600 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-white hover:file:bg-indigo-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 disabled:opacity-50"
          />
          {uploading ? <LoadingState label="Indexing document…" /> : null}
        </div>
      </section>

      <section aria-label="Knowledge base status" className="space-y-2">
        <h2 className="text-sm font-semibold text-slate-700">Status</h2>
        {status ? (
          <dl className="flex flex-wrap gap-x-8 gap-y-2 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm shadow-sm">
            <div>
              <dt className="text-[11px] uppercase tracking-wide text-slate-400">
                Store
              </dt>
              <dd
                className={`font-semibold ${
                  status.available ? "text-emerald-600" : "text-red-600"
                }`}
              >
                {status.available ? "Connected" : "Unavailable"}
              </dd>
            </div>
            <div>
              <dt className="text-[11px] uppercase tracking-wide text-slate-400">
                Documents
              </dt>
              <dd className="font-semibold text-slate-800">
                {status.documents}
              </dd>
            </div>
            <div>
              <dt className="text-[11px] uppercase tracking-wide text-slate-400">
                Chunks
              </dt>
              <dd className="font-semibold text-slate-800">{status.chunks}</dd>
            </div>
            <div>
              <dt className="text-[11px] uppercase tracking-wide text-slate-400">
                Collection
              </dt>
              <dd className="font-mono text-xs leading-6 text-slate-600">
                {status.collection}
              </dd>
            </div>
          </dl>
        ) : (
          <LoadingState label="Checking knowledge base…" />
        )}
      </section>

      {documents === null ? (
        <LoadingState label="Loading documents…" />
      ) : (
        <DocumentList
          documents={documents}
          busy={busy}
          onDelete={handleDelete}
        />
      )}
    </div>
  );
}
