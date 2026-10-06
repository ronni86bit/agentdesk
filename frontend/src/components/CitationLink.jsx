export default function CitationLink({ id, active = false, onSelect, label }) {
  return (
    <button
      type="button"
      onClick={() => onSelect?.(id)}
      aria-label={`${label ?? "View source"} ${id}`}
      className={`mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded px-1 align-baseline text-[11px] font-semibold leading-none transition focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 ${
        active
          ? "bg-indigo-600 text-white"
          : "bg-indigo-100 text-indigo-700 hover:bg-indigo-200"
      }`}
    >
      {id}
    </button>
  );
}
