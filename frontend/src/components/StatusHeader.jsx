const STATUS_META = {
  idle: { label: "Idle", classes: "bg-slate-100 text-slate-600" },
  running: { label: "Researching", classes: "bg-amber-100 text-amber-700" },
  completed: {
    label: "Completed",
    classes: "bg-emerald-100 text-emerald-700",
  },
  failed: { label: "Failed", classes: "bg-red-100 text-red-700" },
};

export default function StatusHeader({ status = "idle" }) {
  const meta = STATUS_META[status] ?? STATUS_META.idle;
  return (
    <header className="flex items-center justify-between border-b border-slate-200 bg-white px-6 py-4">
      <div className="flex items-center gap-2">
        <span className="text-lg font-semibold tracking-tight text-slate-900">
          AgentDesk
        </span>
        <span className="hidden text-sm text-slate-400 sm:inline">
          Multi-agent research assistant
        </span>
      </div>
      <span
        className={`rounded-full px-3 py-1 text-xs font-medium ${meta.classes}`}
      >
        {meta.label}
      </span>
    </header>
  );
}
