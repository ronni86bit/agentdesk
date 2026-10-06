import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { vi } from "vitest";
import EvalDashboard from "./EvalDashboard";
import ConfidenceBadge from "./ConfidenceBadge";
import ReportView from "./ReportView";
import userEvent from "@testing-library/user-event";

const HISTORY = [
  {
    id: "a",
    query: "AI in healthcare",
    timestamp: "2026-01-01T10:00:00Z",
    evaluation: {
      grounding_score: 0.88,
      citation_coverage: 0.92,
      critic_status: "approved",
      iteration_count: 1,
      total_sources: 25,
      web_percentage: 72,
      kb_percentage: 28,
    },
  },
  {
    id: "b",
    query: "EV adoption",
    timestamp: "2026-01-02T10:00:00Z",
    evaluation: {
      grounding_score: 0.61,
      citation_coverage: 0.7,
      critic_status: "revision_needed",
      iteration_count: 2,
      total_sources: 12,
      web_percentage: 100,
      kb_percentage: 0,
    },
  },
  { id: "c", query: "no evaluation stored", timestamp: "2026-01-03T10:00:00Z" },
];

describe("EvalDashboard", () => {
  it("renders evaluation rows from history", () => {
    render(<EvalDashboard history={HISTORY} />);
    expect(screen.getByText("AI in healthcare")).toBeInTheDocument();
    expect(screen.getByText("Approved")).toBeInTheDocument();
    expect(screen.getByText("Revision needed")).toBeInTheDocument();
    expect(screen.getAllByText("92%").length).toBeGreaterThan(0);
  });

  it("handles the empty state and skips records without evaluation", () => {
    render(<EvalDashboard history={[HISTORY[2]]} />);
    expect(screen.getByText(/No evaluation data yet/i)).toBeInTheDocument();
  });
});

describe("ConfidenceBadge", () => {
  it("maps coverage to grounding levels", () => {
    render(<ConfidenceBadge coverage={0.92} />);
    expect(screen.getByText("High grounding")).toBeInTheDocument();
  });

  it("labels weak grounding clearly and documents the heuristic", () => {
    render(<ConfidenceBadge coverage={0.48} />);
    expect(screen.getByText("Low grounding")).toBeInTheDocument();
    expect(screen.getByText("48% coverage")).toBeInTheDocument();
  });

  it("renders nothing without data", () => {
    const { container } = render(<ConfidenceBadge coverage={undefined} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("ReportView citation linking", () => {
  it("renders markdown and turns [N] into clickable citations", async () => {
    const onSelectSource = vi.fn();
    render(
      <ReportView
        report={{
          title: "Report",
          markdown:
            "# Title\n\nClaim one [1] and claim two [2].\n\n<script>alert(1)</script>",
        }}
        onSelectSource={onSelectSource}
      />,
    );
    const cite1 = screen.getByRole("button", { name: "View source 1" });
    expect(
      screen.getByRole("button", { name: "View source 2" }),
    ).toBeInTheDocument();
    await userEvent.click(cite1);
    expect(onSelectSource).toHaveBeenCalledWith(1);
    // Raw HTML must NOT be rendered as markup.
    expect(document.querySelector("script")).toBeNull();
    expect(screen.getByText("<script>alert(1)</script>")).toBeInTheDocument();
  });
});
