export default function LoadingState({ label = "Working…" }) {
  return (
    <div className="flex items-center gap-3 px-1 py-2 text-sm text-slate-500">
      <span
        aria-hidden="true"
        className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-slate-600"
      />
      <span role="status">{label}</span>
    </div>
  );
}
