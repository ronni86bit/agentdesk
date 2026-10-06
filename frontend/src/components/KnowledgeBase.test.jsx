import React from "react";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import KnowledgeBase from "./KnowledgeBase";

vi.mock("../services/api", () => {
  class ApiError extends Error {
    constructor(message, status) {
      super(message);
      this.status = status;
    }
  }
  return {
    ApiError,
    getJSON: vi.fn(),
    uploadFile: vi.fn(),
    deleteJSON: vi.fn(),
  };
});

import { ApiError, deleteJSON, getJSON, uploadFile } from "../services/api";

function mockKbState(documents = []) {
  getJSON.mockImplementation((path) => {
    if (path === "/kb/documents") {
      return Promise.resolve({ documents, total: documents.length });
    }
    if (path === "/kb/status") {
      return Promise.resolve({
        available: true,
        collection: "agentdesk_knowledge",
        documents: documents.length,
        chunks: documents.reduce((sum, d) => sum + (d.chunk_count ?? 0), 0),
      });
    }
    return Promise.reject(new Error(`unexpected path ${path}`));
  });
}

function getFileInput() {
  return screen.getByLabelText(/upload document/i, { selector: "input" });
}

const DOC = {
  document_id: "abc123",
  document_name: "study.pdf",
  file_type: "pdf",
  chunk_count: 42,
  created_at: "2026-01-01T00:00:00Z",
  content_hash: "hash",
  size_bytes: 2048,
  status: "indexed",
};

describe("KnowledgeBase", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockKbState([DOC]);
  });

  it("shows KB status and document list", async () => {
    render(<KnowledgeBase />);
    expect(await screen.findByText("Connected")).toBeInTheDocument();
    expect(await screen.findByText("study.pdf")).toBeInTheDocument();
    expect(screen.getByText(/42 chunks/)).toBeInTheDocument();
  });

  it("uploads a document and shows chunk-count success", async () => {
    uploadFile.mockResolvedValue({
      status: "created",
      document: { ...DOC, chunk_count: 42 },
    });
    render(<KnowledgeBase />);
    await userEvent.upload(
      getFileInput(),
      new File(["content"], "notes.txt", { type: "text/plain" }),
    );
    await waitFor(() => {
      expect(
        screen.getByText(/Indexed successfully — 42 chunks/),
      ).toBeInTheDocument();
    });
    expect(uploadFile).toHaveBeenCalledWith("/kb/upload", expect.any(File));
  });

  it("rejects unsupported extensions client-side", async () => {
    render(<KnowledgeBase />);
    // userEvent.upload honors accept=".pdf,.txt,.md" and would skip the
    // file, so bypass it deliberately to exercise the component's own
    // extension validation.
    fireEvent.change(getFileInput(), {
      target: {
        files: [new File(["MZ"], "bad.exe", { type: "application/octet-stream" })],
      },
    });
    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(
        /Unsupported file type/,
      );
    });
    expect(uploadFile).not.toHaveBeenCalled();
  });

  it("deletes a document", async () => {
    deleteJSON.mockResolvedValue({ status: "deleted", document_id: "abc123" });
    render(<KnowledgeBase />);
    await screen.findByText("study.pdf");
    await userEvent.click(
      screen.getByRole("button", { name: /Delete study\.pdf/ }),
    );
    await waitFor(() => {
      expect(deleteJSON).toHaveBeenCalledWith("/kb/documents/abc123");
    });
  });

  it("surfaces storage unavailability safely", async () => {
    getJSON.mockRejectedValue(
      new ApiError("Knowledge base storage is unavailable.", 503),
    );
    render(<KnowledgeBase />);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      /storage is unavailable/i,
    );
  });
});
