import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import SourcePanel from "./SourcePanel";

const SOURCES = [
  {
    source_type: "web",
    title: "Web source",
    url: "https://example.com/a",
    score: 0.91,
    preview: "Web preview text",
    source_id: "https://example.com/a",
  },
  {
    source_type: "kb",
    title: "",
    document_name: "notes.pdf",
    score: 0.42,
    preview: "KB preview text",
    source_id: "doc1:0",
  },
];

describe("SourcePanel", () => {
  it("renders numbered sources with type badges", () => {
    render(<SourcePanel sources={SOURCES} />);
    expect(screen.getByText("WEB")).toBeInTheDocument();
    expect(screen.getByText("KB")).toBeInTheDocument();
    expect(screen.getByText("Web source")).toBeInTheDocument();
    expect(screen.getByText(/Document: notes\.pdf/)).toBeInTheDocument();
    expect(screen.getByText("Web preview text")).toBeInTheDocument();
  });

  it("web links open safely in a new tab", () => {
    render(<SourcePanel sources={SOURCES} />);
    const link = screen.getByRole("link", { name: /https:\/\/example\.com\/a/ });
    expect(link).toHaveAttribute("target", "_blank");
    expect(link.getAttribute("rel")).toContain("noopener");
  });

  it("clicking a source reports its number", async () => {
    const onSelectSource = vi.fn();
    render(<SourcePanel sources={SOURCES} onSelectSource={onSelectSource} />);
    await userEvent.click(screen.getByRole("button", { name: /Source 2/ }));
    expect(onSelectSource).toHaveBeenCalledWith(2);
  });

  it("highlights the active source", () => {
    const { container } = render(
      <SourcePanel sources={SOURCES} activeSource={2} />,
    );
    const active = container.querySelector(".ring-indigo-200");
    expect(active).not.toBeNull();
    expect(active.textContent).toContain("notes.pdf");
  });

  it("shows an empty state without sources", () => {
    render(<SourcePanel sources={[]} />);
    expect(screen.getByText(/No evidence sources yet/i)).toBeInTheDocument();
  });
});
