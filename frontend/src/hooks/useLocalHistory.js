// localStorage-backed research-run history. Malformed data is discarded
// gracefully; nothing sensitive (keys, tokens, raw logs) is ever stored.

import { useCallback, useEffect, useState } from "react";

const STORAGE_KEY = "agentdesk.history.v1";
const MAX_RECORDS = 50;

function isRecord(value) {
  return (
    value !== null &&
    typeof value === "object" &&
    typeof value.id === "string" &&
    typeof value.query === "string"
  );
}

export function loadHistory() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isRecord);
  } catch {
    return [];
  }
}

function persist(records) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(records));
  } catch {
    // Storage full/unavailable: history is best-effort by design.
  }
}

export function useLocalHistory() {
  const [history, setHistory] = useState(loadHistory);

  useEffect(() => {
    persist(history);
  }, [history]);

  const addRun = useCallback((record) => {
    if (!isRecord(record)) return null;
    setHistory((prev) => [record, ...prev].slice(0, MAX_RECORDS));
    return record;
  }, []);

  const deleteRun = useCallback((id) => {
    setHistory((prev) => prev.filter((item) => item.id !== id));
  }, []);

  const clearHistory = useCallback(() => {
    setHistory([]);
  }, []);

  return { history, addRun, deleteRun, clearHistory };
}
