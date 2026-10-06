import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useLocalHistory, loadHistory } from "./useLocalHistory";

function makeRecord(id) {
  return {
    id,
    query: `query ${id}`,
    timestamp: "2026-01-01T00:00:00Z",
    title: `title ${id}`,
    finalStatus: "approved",
    reportMarkdown: "# report",
    references: [],
    sources: [],
    evaluation: { grounding_score: 0.9, citation_coverage: 1, critic_status: "approved", total_sources: 3 },
  };
}

describe("useLocalHistory", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("adds, lists, and caps records at 50", () => {
    const { result } = renderHook(() => useLocalHistory());
    for (let i = 0; i < 55; i += 1) {
      act(() => {
        result.current.addRun(makeRecord(`id-${i}`));
      });
    }
    expect(result.current.history).toHaveLength(50);
    // newest first
    expect(result.current.history[0].id).toBe("id-54");
  });

  it("deletes one and clears all", () => {
    const { result } = renderHook(() => useLocalHistory());
    act(() => {
      result.current.addRun(makeRecord("a"));
      result.current.addRun(makeRecord("b"));
    });
    act(() => {
      result.current.deleteRun("a");
    });
    expect(result.current.history.map((r) => r.id)).toEqual(["b"]);
    act(() => {
      result.current.clearHistory();
    });
    expect(result.current.history).toEqual([]);
    expect(loadHistory()).toEqual([]);
  });

  it("persists to localStorage", () => {
    const { result } = renderHook(() => useLocalHistory());
    act(() => {
      result.current.addRun(makeRecord("persisted"));
    });
    expect(loadHistory()[0].id).toBe("persisted");
  });

  it("survives malformed localStorage data", () => {
    window.localStorage.setItem("agentdesk.history.v1", "{not json");
    expect(loadHistory()).toEqual([]);
    window.localStorage.setItem("agentdesk.history.v1", JSON.stringify({ nope: 1 }));
    expect(loadHistory()).toEqual([]);
    const { result } = renderHook(() => useLocalHistory());
    expect(result.current.history).toEqual([]);
    act(() => {
      result.current.addRun(makeRecord("after-garbage"));
    });
    expect(result.current.history[0].id).toBe("after-garbage");
  });

  it("rejects invalid records", () => {
    const { result } = renderHook(() => useLocalHistory());
    act(() => {
      result.current.addRun({ query: "missing id" });
    });
    expect(result.current.history).toEqual([]);
    expect(vi.isMockFunction(result.current.addRun)).toBe(false);
  });
});
