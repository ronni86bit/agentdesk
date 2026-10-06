import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import AgentProgress from "./AgentProgress";
import { createInitialResearchState, reduceResearchEvent } from "../hooks/eventState";

function runState(events) {
  return events.reduce(reduceResearchEvent, createInitialResearchState());
}

const PROGRESS_EVENTS = [
  { event: "research_started", data: { query: "q" } },
  { event: "planner_started" },
  { event: "planner_completed", data: { sub_questions: 3 } },
  { event: "retrieval_started" },
  {
    event: "retrieval_completed",
    data: { total_sources: 20, web_sources: 18, kb_sources: 2 },
  },
  { event: "writer_started" },
  { event: "writer_completed", data: { citation_count: 13 } },
  { event: "critic_started" },
  {
    event: "critic_completed",
    data: { status: "approved", iteration: 0, issue_count: 0 },
  },
];

describe("AgentProgress", () => {
  it("renders nothing before any activity", () => {
    const { container } = render(
      <AgentProgress state={createInitialResearchState()} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("maps SSE events to stage summaries", () => {
    render(<AgentProgress state={runState(PROGRESS_EVENTS)} />);
    expect(screen.getByText("Planner")).toBeInTheDocument();
    expect(screen.getByText("Retriever")).toBeInTheDocument();
    expect(screen.getByText("Writer")).toBeInTheDocument();
    expect(screen.getByText("Critic")).toBeInTheDocument();
    expect(screen.getByText("3 sub-questions")).toBeInTheDocument();
    expect(screen.getByText("20 sources · 18 Web / 2 KB")).toBeInTheDocument();
    expect(screen.getByText("13 citations")).toBeInTheDocument();
    expect(screen.getByText("Approved")).toBeInTheDocument();
  });

  it("shows an active revision", () => {
    const state = runState([
      ...PROGRESS_EVENTS,
      { event: "revision_started", iteration: 2, data: { iteration: 2 } },
    ]);
    render(<AgentProgress state={state} />);
    expect(screen.getByText(/Revision 2 in progress/)).toBeInTheDocument();
  });

  it("shows revision-needed verdict with issue count", () => {
    const state = runState([
      ...PROGRESS_EVENTS.slice(0, 8),
      {
        event: "critic_completed",
        data: { status: "revision_needed", iteration: 0, issue_count: 4 },
      },
    ]);
    render(<AgentProgress state={state} />);
    expect(screen.getByText(/Revision needed · 4 issues/)).toBeInTheDocument();
  });
});
