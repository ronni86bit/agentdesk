import { useEffect, useState } from "react";
import { getJSON } from "../services/api";

function Row({ label, value, tone = "default" }) {
  const valueClasses =
    tone === "good"
      ? "text-emerald-600"
      : tone === "bad"
        ? "text-red-600"
        : "text-slate-800";
  return (
    <div className="flex items-center justify-between gap-4 border-b border-slate-100 py-2 last:border-b-0">
      <dt className="text-xs font-medium uppercase tracking-wide text-slate-400">
        {label}
      </dt>
      <dd className={`text-sm font-semibold ${valueClasses}`}>{value}</dd>
    </div>
  );
}

/**
 * Non-secret runtime settings. Provider configuration is shown as booleans
 * only — API keys never leave the backend and are never shown or stored in
 * the browser.
 */
export default function SettingsView() {
  const [runtime, setRuntime] = useState(null);
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([getJSON("/runtime"), getJSON("/health")])
      .then(([runtimeBody, healthBody]) => {
        if (cancelled) return;
        setRuntime(runtimeBody);
        setHealth(healthBody);
      })
      .catch(() => {
        if (!cancelled) setError("Could not reach the AgentDesk backend.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <div className="space-y-4">
        <h1 className="text-xl font-bold tracking-tight text-slate-900">
          Settings
        </h1>
        <p role="alert" className="text-sm text-red-600">
          {error}
        </p>
      </div>
    );
  }

  if (!runtime || !health) {
    return (
      <div className="space-y-4">
        <h1 className="text-xl font-bold tracking-tight text-slate-900">
          Settings
        </h1>
        <p className="animate-pulse text-sm text-slate-400">Loading…</p>
      </div>
    );
  }

  return (
    <div className="max-w-2xl space-y-6">
      <header className="space-y-1">
        <h1 className="text-xl font-bold tracking-tight text-slate-900">
          Settings
        </h1>
        <p className="text-sm text-slate-500">
          Runtime configuration. Secrets stay on the server: keys are never
          sent to, or stored by, this application.
        </p>
      </header>

      <section className="rounded-xl border border-slate-200 bg-white px-4 py-2 shadow-sm">
        <dl>
          <Row
            label="Backend"
            value={health.status === "ok" ? "Connected" : "Degraded"}
            tone={health.status === "ok" ? "good" : "bad"}
          />
          <Row label="LLM model" value={<span className="font-mono text-xs">{runtime.model}</span>} />
          <Row
            label="LLM provider key"
            value={runtime.groq_configured ? "Configured" : "Missing"}
            tone={runtime.groq_configured ? "good" : "bad"}
          />
          <Row
            label="Web search"
            value={runtime.tavily_configured ? "Configured" : "Not configured"}
            tone={runtime.tavily_configured ? "good" : "bad"}
          />
          <Row
            label="Retrieval modes"
            value={runtime.retrieval_modes?.join(" · ") ?? "—"}
          />
          <Row
            label="KB collection"
            value={<span className="font-mono text-xs">{runtime.kb_collection}</span>}
          />
          <Row
            label="Embedding model"
            value={<span className="font-mono text-xs">{runtime.embedding_model}</span>}
          />
        </dl>
      </section>

      <p className="text-xs text-slate-400">
        Retrieval mode, history, and saved evaluations are stored per browser;
        no account system exists and no research content is stored on the
        AgentDesk server beyond the knowledge base you upload.
      </p>
    </div>
  );
}
