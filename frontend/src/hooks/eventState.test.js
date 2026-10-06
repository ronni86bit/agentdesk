import { describe, expect, it } from "vitest";
import {
  createInitialResearchState,
  reduceResearchEvent,
} from "./eventState";

function run(events) {
  return events.reduce(reduceResearchEvent, createInitialResearchState());
}

const RESULT = {
  query: "q",
  report: { title: "Title", markdown: "# Title\nClaim [1]." },
  final_status: "approved",
  iteration_count: 1,
  evaluation: { grounding_score: 0.91 },
};

describe("reduceResearchEvent", () => {
  it("starts idle and ignores malformed payloads", () => {
    const state = createInitialResearchState();
    expect(state.status).toBe("idle");
    expect(reduceResearchEvent(state, null)).toBe(state);
    expect(reduceResearchEvent(state, {})).toBe(state);
    expect(reduceResearchEvent(state, { nope: 1 })).toBe(state);
  });

  it("maps the full happy-path event sequence", () => {
    const state = run([
      { event: "research_started", data: { query: "q", retrieval_mode: "web" } },
      { event: "planner_started" },
      { event: "planner_completed", data: { sub_questions: 4 } },
      { event: "retrieval_started" },
      {
        event: "retrieval_completed",
        data: {
          total_sources: 20,
          web_sources: 18,
          kb_sources: 2,
          deduplicated_sources: 3,
          kb_status: "ok",
          web_status: "ok",
          retrieval_mode: "web",
        },
      },
      { event: "writer_started" },
      { event: "writer_completed", data: { citation_count: 13, report_title: "T" } },
      { event: "critic_started", iteration: 0 },
      { event: "critic_completed", data: { status: "approved", iteration: 0, issue_count: 0 } },
      { event: "research_completed", data: { final_status: "approved", iteration_count: 0, result: RESULT } },
    ]);

    expect(state.status).toBe("completed");
    expect(state.query).toBe("q");
    expect(state.retrievalMode).toBe("web");
    expect(state.stages).toEqual({
      planner: "completed",
      retriever: "completed",
      writer: "completed",
      critic: "completed",
    });
    expect(state.planInfo).toEqual({ subQuestions: 4 });
    expect(state.retrieval).toEqual({
      total: 20, web: 18, kb: 2, deduplicated: 3,
      kbStatus: "ok", webStatus: "ok", mode: "web",
    });
    expect(state.writerInfo).toEqual({ citations: 13, title: "T" });
    expect(state.criticInfo).toEqual({ verdict: "approved", iteration: 0, issues: 0 });
    expect(state.result).toEqual(RESULT);
    expect(state.report).toEqual(RESULT.report);
    expect(state.finalStatus).toBe("approved");
  });

  it("tracks revision cycles", () => {
    let state = createInitialResearchState();
    state = reduceResearchEvent(state, { event: "research_started" });
    state = reduceResearchEvent(state, {
      event: "revision_started",
      iteration: 1,
      data: { iteration: 1 },
    });
    expect(state.revision).toEqual({ iteration: 1, active: true });
    expect(state.stages.writer).toBe("running");
    state = reduceResearchEvent(state, {
      event: "revision_completed",
      iteration: 1,
    });
    expect(state.revision).toEqual({ iteration: 1, active: false });
    expect(state.stages.writer).toBe("completed");
  });

  it("failure state carries the safe message", () => {
    const state = run([
      { event: "research_started" },
      { event: "research_failed", data: { message: "Research failed." } },
    ]);
    expect(state.status).toBe("failed");
    expect(state.error).toBe("Research failed.");
  });

  it("unknown events are ignored (forward compatibility)", () => {
    const before = createInitialResearchState();
    const after = reduceResearchEvent(before, { event: "hologram_started" });
    expect(after).toBe(before);
  });

  it("stop semantics: connection error only fails a running stream", () => {
    const state = run([
      { event: "research_started" },
      { event: "research_failed", data: { message: "boom" } },
    ]);
    expect(state.status).toBe("failed");
  });
});
