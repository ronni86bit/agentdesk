import { useCallback, useEffect, useRef, useState } from "react";
import { openResearchStream } from "../services/researchStream";
import {
  createInitialResearchState,
  reduceResearchEvent,
} from "./eventState";

const MAX_EVENT_LOG = 100;

/**
 * Manage the SSE research lifecycle:
 * open on startResearch, close on terminal event / stop / unmount.
 */
export function useResearchStream() {
  const [state, setState] = useState(createInitialResearchState);
  const [events, setEvents] = useState([]);
  const sourceRef = useRef(null);

  const closeStream = useCallback(() => {
    if (sourceRef.current) {
      sourceRef.current.close();
      sourceRef.current = null;
    }
  }, []);

  const startResearch = useCallback(
    (query, mode = "both") => {
      const cleanQuery = (query ?? "").trim();
      if (!cleanQuery) return;

      closeStream();
      setState({ ...createInitialResearchState(), status: "running" });
      setEvents([]);

      sourceRef.current = openResearchStream({
        query: cleanQuery,
        mode,
        onEvent: (payload) => {
          if (payload === null) return; // malformed frame, ignore safely
          setState((prev) => reduceResearchEvent(prev, payload));
          setEvents((prev) => {
            const next = [...prev, payload];
            return next.length > MAX_EVENT_LOG
              ? next.slice(next.length - MAX_EVENT_LOG)
              : next;
          });
        },
        onConnectionError: (message) => {
          setState((prev) =>
            prev.status === "running"
              ? { ...prev, status: "failed", error: message }
              : prev,
          );
        },
      });
    },
    [closeStream],
  );

  const stopResearch = useCallback(() => {
    closeStream();
    // User cancellation is not an error: return to idle.
    setState((prev) =>
      prev.status === "running"
        ? { ...createInitialResearchState(), status: "idle" }
        : prev,
    );
  }, [closeStream]);

  // Close the stream when the owning component unmounts.
  useEffect(() => closeStream, [closeStream]);

  return {
    startResearch,
    stopResearch,
    events,
    latestEvent: events.length > 0 ? events[events.length - 1] : null,
    status: state.status, // idle | running | completed | failed
    query: state.query,
    retrievalMode: state.retrievalMode,
    stages: state.stages,
    planInfo: state.planInfo,
    retrieval: state.retrieval,
    writerInfo: state.writerInfo,
    criticInfo: state.criticInfo,
    revision: state.revision,
    result: state.result,
    report: state.report,
    finalStatus: state.finalStatus,
    iterationCount: state.iterationCount,
    error: state.error,
    isRunning: state.status === "running",
  };
}
