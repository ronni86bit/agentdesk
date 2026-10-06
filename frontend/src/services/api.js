// Central API configuration + typed helpers.
//
// Routing model: the browser always requests `<API_BASE>/api/...`.
// - Development: Vite's dev proxy forwards /api/* to the backend (rewrite
//   strips the /api prefix), so no localhost URLs are hardcoded.
// - Production: nginx serves the SPA and reverse-proxies /api/* to the
//   backend container (SSE-safe: buffering disabled).
// VITE_API_BASE_URL stays EMPTY except for exotic direct-connect setups.
const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/+$/, "");

export function apiUrl(path) {
  return `${API_BASE}/api${path}`;
}

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function parseError(response) {
  let detail = "";
  try {
    const body = await response.json();
    detail = body?.detail ?? "";
  } catch {
    // non-JSON error body; fall through with generic text
  }
  const fallbacks = {
    404: "Not found.",
    413: "File is too large.",
    422: detail || "The request was rejected.",
    429: "AI provider rate limit reached. Please try again later.",
    502: "The backend pipeline failed. Please try again.",
    503:
      detail ||
      "A required backend service is unavailable. Is the AgentDesk backend running?",
  };
  return new ApiError(detail || fallbacks[response.status] || `Request failed (${response.status}).`, response.status);
}

async function request(path, options) {
  let response;
  try {
    response = await fetch(apiUrl(path), options);
  } catch {
    throw new ApiError(
      "Could not reach the AgentDesk backend. Is it running?",
      0,
    );
  }
  if (!response.ok) throw await parseError(response);
  return response;
}

export function getJSON(path) {
  return request(path).then((r) => r.json());
}

export function postJSON(path, body) {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  }).then((r) => r.json());
}

export function deleteJSON(path) {
  return request(path, { method: "DELETE" }).then((r) => r.json());
}

export async function uploadFile(path, file) {
  const form = new FormData();
  form.append("file", file);
  const response = await request(path, { method: "POST", body: form });
  return response.json();
}

export async function postForBlob(path, body) {
  const response = await request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  return response.blob();
}
