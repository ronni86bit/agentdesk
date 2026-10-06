export default function EmptyState({ title, hint, children }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-slate-300 bg-slate-50 px-6 py-12 text-center">
      <p className="text-sm font-medium text-slate-600">{title}</p>
      {hint ? (
        <p className="mt-1 max-w-md text-xs text-slate-400">{hint}</p>
      ) : null}
      {children ? <div className="mt-4">{children}</div> : null}
    </div>
  );
}
