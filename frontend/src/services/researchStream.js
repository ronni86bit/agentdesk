import { apiUrl } from "./api";

export const RESEARCH_EVENT_NAMES = [
  "research_started",
  "planner_started",
  "planner_completed",
  "retrieval_started",
  "retrieval_completed",
  "writer_started",
  "writer_completed",
  "critic_started",
  "critic_completed",
  "revision_started",
  "revision_completed",
  "research_completed",
  "research_failed",
];

const TERMINAL_EVENTS = new Set(["research_completed", "research_failed"]);

/**
 * Open an SSE connection to the backend research stream.
 *
 * Returns a handle with close(). The connection closes itself after a
 * terminal event (research_completed / research_failed); the caller may also
 * close it early (user cancellation, unmount).
 *
 * @param {object} opts
 * @param {string} opts.query      research goal (must be non-empty)
 * @param {string} [opts.mode]     retrieval mode: "both" | "web" | "kb"
 * @param {(event: object|null) => void} opts.onEvent
 *        called with each parsed JSON payload; null for malformed payloads.
 * @param {(message: string) => void} opts.onConnectionError
 *        called once if the SSE connection fails before completing.
 */
export function openResearchStream({ query, mode = "both", onEvent, onConnectionError }) {
  const params = new URLSearchParams({
    query: query.trim(),
    retrieval_mode: mode,
  });
  const url = apiUrl(`/research/stream?${params.toString()}`);
  let closed = false;
  let failed = false;

  const source = new EventSource(url);

  function shutdown() {
    if (!closed) {
      closed = true;
      source.close();
    }
  }

  function makeHandler(name) {
    return (e) => {
      if (closed) return;
      let payload = null;
      try {
        payload = JSON.parse(e.data);
      } catch {
        // Malformed frame: surface nothing user-facing, keep stream alive.
        onEvent(null);
        return;
      }
      onEvent(payload);
      if (TERMINAL_EVENTS.has(name)) shutdown();
    };
  }

  for (const name of RESEARCH_EVENT_NAMES) {
    source.addEventListener(name, makeHandler(name));
  }

  source.onerror = () => {
    if (closed || failed) return;
    if (source.readyState === EventSource.CLOSED) {
      // The browser gave up reconnecting (backend down / network failure).
      failed = true;
      shutdown();
      onConnectionError(
        "Could not reach the AgentDesk backend. Is it running?",
      );
    }
    // readyState === CONNECTING: transient drop; the browser retries
    // automatically, so we stay quiet unless it fully closes.
  };

  return { close: shutdown };
}
