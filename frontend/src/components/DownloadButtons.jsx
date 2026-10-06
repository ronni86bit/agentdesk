import { useState } from "react";
import { postForBlob } from "../services/api";
import { slugify } from "../utils/formatting";

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export function buildMarkdownDownload(result) {
  const lines = [];
  if (result.report?.title) {
    lines.push(`# ${result.report.title}`, "");
  }
  if (result.query) {
    lines.push(`> Research goal: ${result.query}`, "");
  }
  lines.push(result.report?.markdown ?? "");
  // Ensure authoritative references are present even if the markdown body
  // lacks a References section.
  const markdown = lines.join("\n");
  const refs = result.report?.references ?? [];
  if (refs.length && !/##\s*references/i.test(markdown)) {
    lines.push("", "## References", "");
    for (const ref of refs) {
      const locator =
        ref.source_type === "kb"
          ? `Document: ${ref.document_name ?? "knowledge base"}`
          : (ref.url ?? "web source");
      lines.push(`[${ref.citation_id}] ${ref.title} — ${locator}`);
    }
  }
  return lines.join("\n");
}

export default function DownloadButtons({ result, disabled = false }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  if (!result?.report) return null;

  const baseName = slugify(result.report.title || result.query);

  function downloadMarkdown() {
    const blob = new Blob([buildMarkdownDownload(result)], {
      type: "text/markdown;charset=utf-8",
    });
    triggerDownload(blob, `${baseName}.md`);
  }

  async function downloadPdf() {
    setBusy(true);
    setError(null);
    try {
      const blob = await postForBlob("/report/pdf", {
        report: result.report,
        query: result.query ?? "",
        retrieval_mode: result.retrieval_mode ?? "both",
        final_status: result.final_status ?? "",
      });
      triggerDownload(blob, `${baseName}.pdf`);
    } catch (err) {
      setError(err.message ?? "PDF export failed.");
    } finally {
      setBusy(false);
    }
  }

  const buttonClasses =
    "inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 shadow-sm transition hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 disabled:cursor-not-allowed disabled:opacity-50";

  return (
    <div className="flex flex-wrap items-center gap-2">
      <button type="button" onClick={downloadMarkdown} className={buttonClasses}>
        ↓ Markdown
      </button>
      <button
        type="button"
        onClick={downloadPdf}
        disabled={busy || disabled}
        className={buttonClasses}
      >
        {busy ? "Exporting…" : "↓ PDF"}
      </button>
      {error ? (
        <span role="alert" className="text-xs text-red-600">
          {error}
        </span>
      ) : null}
    </div>
  );
}
