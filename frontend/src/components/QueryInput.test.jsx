import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import QueryInput from "./QueryInput";

function getQueryBox() {
  return screen.getByRole("textbox", { name: /research goal/i });
}

describe("QueryInput", () => {
  it("rejects empty queries without calling onSubmit", async () => {
    const onSubmit = vi.fn();
    render(<QueryInput onSubmit={onSubmit} />);
    await userEvent.click(screen.getByRole("button", { name: "Research" }));
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("submits trimmed query via button", async () => {
    const onSubmit = vi.fn();
    render(<QueryInput onSubmit={onSubmit} />);
    await userEvent.type(getQueryBox(), "  AI in healthcare  ");
    await userEvent.click(screen.getByRole("button", { name: "Research" }));
    expect(onSubmit).toHaveBeenCalledWith("AI in healthcare");
  });

  it("submits with Ctrl+Enter and clears afterwards via Clear", async () => {
    const onSubmit = vi.fn();
    render(<QueryInput onSubmit={onSubmit} />);
    const input = getQueryBox();
    await userEvent.type(input, "query text{Control>}{Enter}{/Control}");
    expect(onSubmit).toHaveBeenCalledWith("query text");

    await userEvent.click(screen.getByRole("button", { name: "Clear" }));
    expect(input).toHaveValue("");
  });

  it("disables submit and shows Stop while running", () => {
    const onSubmit = vi.fn();
    render(<QueryInput onSubmit={onSubmit} isRunning />);
    expect(screen.queryByRole("button", { name: "Research" })).toBeNull();
    expect(screen.getByRole("button", { name: "Stop" })).toBeInTheDocument();
  });

  it("enforces a character limit", async () => {
    render(<QueryInput onSubmit={vi.fn()} />);
    const input = getQueryBox();
    await userEvent.type(input, "x".repeat(120), { skipClick: true });
    expect(input.value.length).toBeLessThanOrEqual(120);
  });
});
