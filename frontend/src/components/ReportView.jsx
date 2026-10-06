import ReactMarkdown from "react-markdown";
import CitationLink from "./CitationLink";
import { mapCitationTokens } from "../utils/citations";

/**
 * Render the final report markdown safely.
 *
 * react-markdown does NOT render raw HTML (no rehype-raw is configured), so
 * remote markup can never execute. Inline [N] citation markers inside any
 * text node become clickable CitationLink buttons.
 */
export default function ReportView({
  report,
  activeSource = null,
  onSelectSource,
}) {
  if (!report?.markdown) return null;

  const components = {
    p: ({ children }) => (
      <p className="text-sm leading-6 text-slate-700">
        {mapCitationTokens(children, (id, key) => (
          <CitationLink
            key={key}
            id={id}
            active={id === activeSource}
            onSelect={onSelectSource}
          />
        ))}
      </p>
    ),
    li: ({ children }) => (
      <li className="text-sm leading-6 text-slate-700 marker:text-slate-400">
        {mapCitationTokens(children, (id, key) => (
          <CitationLink
            key={key}
            id={id}
            active={id === activeSource}
            onSelect={onSelectSource}
          />
        ))}
      </li>
    ),
    h1: ({ children }) => (
      <h1 className="mt-1 text-xl font-bold tracking-tight text-slate-900">
        {children}
      </h1>
    ),
    h2: ({ children }) => (
      <h2 className="mt-5 border-b border-slate-100 pb-1 text-lg font-semibold text-slate-900">
        {children}
      </h2>
    ),
    h3: ({ children }) => (
      <h3 className="mt-4 text-base font-semibold text-slate-900">
        {children}
      </h3>
    ),
    a: ({ href, children }) => (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer nofollow"
        className="text-indigo-600 underline decoration-indigo-200 underline-offset-2 hover:text-indigo-500"
      >
        {children}
      </a>
    ),
    code: ({ children }) => (
      <code className="rounded bg-slate-100 px-1 py-0.5 font-mono text-[13px] text-slate-800">
        {children}
      </code>
    ),
    pre: ({ children }) => (
      <pre className="overflow-x-auto rounded-lg bg-slate-900 px-4 py-3 text-[13px] leading-6 text-slate-100">
        {children}
      </pre>
    ),
    table: ({ children }) => (
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-sm">{children}</table>
      </div>
    ),
    th: ({ children }) => (
      <th className="border-b border-slate-200 px-3 py-2 text-left font-semibold text-slate-700">
        {children}
      </th>
    ),
    td: ({ children }) => (
      <td className="border-b border-slate-100 px-3 py-2 align-top text-slate-600">
        {mapCitationTokens(children, (id, key) => (
          <CitationLink
            key={key}
            id={id}
            active={id === activeSource}
            onSelect={onSelectSource}
          />
        ))}
      </td>
    ),
    blockquote: ({ children }) => (
      <blockquote className="border-l-4 border-slate-200 pl-4 text-sm italic text-slate-600">
        {children}
      </blockquote>
    ),
  };

  return (
    <article aria-label="Final report" className="space-y-3">
      {report.title ? (
        <h2 className="text-lg font-bold tracking-tight text-slate-900">
          {report.title}
        </h2>
      ) : null}
      <ReactMarkdown components={components}>{report.markdown}</ReactMarkdown>
    </article>
  );
}
