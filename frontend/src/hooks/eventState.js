// Pure event -> UI state mapping. Kept free of React so it can be unit
// tested directly. State is derived ONLY from SSE event names — never from
// message text.

export const STAGES = ["planner", "retriever", "writer", "critic"];

export const STAGE_LABELS = {
  planner: "Planner",
  retriever: "Retriever",
  writer: "Writer",
  critic: "Critic",
};

const STAGE_BY_EVENT_PREFIX = {
  planner_: "planner",
  retrieval_: "retriever",
  writer_: "writer",
  critic_: "critic",
};

export function createInitialResearchState() {
  return {
    status: "idle", // idle | running | completed | failed
    query: null,
    retrievalMode: null, // both | web | kb
    stages: {
      planner: "waiting", // waiting | running | completed | failed
      retriever: "waiting",
      writer: "waiting",
      critic: "waiting",
    },
    planInfo: null, // { subQuestions }
    retrieval: null, // { total, web, kb, deduplicated, kbStatus, webStatus }
    writerInfo: null, // { citations, title }
    criticInfo: null, // { verdict, iteration, issues }
    revision: null, // { iteration, active }
    result: null, // full ResearchRunResult from research_completed
    report: null, // { title, markdown }
    finalStatus: null, // approved | max_revisions_reached | revision_failed
    iterationCount: null,
    error: null, // user-safe error message or null
  };
}

function setStage(stages, stage, value) {
  return { ...stages, [stage]: value };
}

/**
 * Apply one parsed ResearchEvent payload to the state.
 * Unknown events are ignored (forward compatibility).
 */
export function reduceResearchEvent(state, payload) {
  if (!payload || typeof payload.event !== "string") {
    return state;
  }

  const name = payload.event;
  const data = payload.data ?? {};

  switch (name) {
    case "research_started":
      return {
        ...state,
        status: "running",
        query: data.query ?? state.query,
        retrievalMode: data.retrieval_mode ?? state.retrievalMode,
      };

    case "planner_started":
      return { ...state, stages: setStage(state.stages, "planner", "running") };
    case "planner_completed":
      return {
        ...state,
        stages: setStage(state.stages, "planner", "completed"),
        planInfo: {
          subQuestions: typeof data.sub_questions === "number"
            ? data.sub_questions
            : null,
        },
      };

    case "retrieval_started":
      return {
        ...state,
        stages: setStage(state.stages, "retriever", "running"),
      };
    case "retrieval_completed":
      return {
        ...state,
        stages: setStage(state.stages, "retriever", "completed"),
        retrieval: {
          total: data.total_sources ?? 0,
          web: data.web_sources ?? 0,
          kb: data.kb_sources ?? 0,
          deduplicated: data.deduplicated_sources ?? 0,
          kbStatus: data.kb_status ?? null,
          webStatus: data.web_status ?? null,
          mode: data.retrieval_mode ?? state.retrievalMode,
        },
      };

    case "writer_started":
      return { ...state, stages: setStage(state.stages, "writer", "running") };
    case "writer_completed":
      return {
        ...state,
        stages: setStage(state.stages, "writer", "completed"),
        writerInfo: {
          citations: typeof data.citation_count === "number"
            ? data.citation_count
            : null,
          title: data.report_title ?? null,
        },
      };

    case "critic_started":
      // Re-entered on each post-revision audit.
      return { ...state, stages: setStage(state.stages, "critic", "running") };
    case "critic_completed":
      return {
        ...state,
        stages: setStage(state.stages, "critic", "completed"),
        criticInfo: {
          verdict: data.status ?? null, // approved | revision_needed
          iteration: data.iteration ?? null,
          issues: typeof data.issue_count === "number" ? data.issue_count : null,
        },
      };

    case "revision_started":
      return {
        ...state,
        stages: setStage(state.stages, "writer", "running"),
        revision: { iteration: data.iteration ?? null, active: true },
      };
    case "revision_completed":
      return {
        ...state,
        stages: setStage(state.stages, "writer", "completed"),
        revision: {
          iteration: payload.iteration ?? data.iteration ?? null,
          active: false,
        },
      };

    case "research_completed": {
      const result = data.result ?? null;
      const report = result?.report ?? null;
      return {
        ...state,
        status: "completed",
        revision:
          state.revision === null
            ? null
            : { ...state.revision, active: false },
        result,
        report: report ? { title: report.title, markdown: report.markdown } : null,
        finalStatus: data.final_status ?? result?.final_status ?? null,
        iterationCount: data.iteration_count ?? result?.iteration_count ?? null,
      };
    }

    case "research_failed":
      return {
        ...state,
        status: "failed",
        revision:
          state.revision === null
            ? null
            : { ...state.revision, active: false },
        error: data.message || "Research failed.",
      };

    default:
      // Also tolerate any "<stage>_started"/"<stage>_completed" additions.
      for (const prefix in STAGE_BY_EVENT_PREFIX) {
        if (name.startsWith(prefix)) {
          const stage = STAGE_BY_EVENT_PREFIX[prefix];
          const value = name.endsWith("_started")
            ? "running"
            : name.endsWith("_completed")
              ? "completed"
              : null;
          if (value && stage in state.stages) {
            return { ...state, stages: setStage(state.stages, stage, value) };
          }
        }
      }
      return state;
  }
}
