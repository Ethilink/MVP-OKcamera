import type { DetectorControl, Report, StartedResponse, Status } from "./types"

// Backend serves on :8000, Vite dev on :5173 — cross-origin (matches T04 CORS).
// All requests + streamUrl are absolute against BASE so MSW (node + browser)
// intercepts them by URL. Exported so fixtures/handlers match the same origin.
export const BASE: string =
  import.meta.env.VITE_API_BASE ?? "http://localhost:8000"

/** Thrown on any non-2xx response. `detail` comes from FastAPI's `{detail}`. */
export class ApiError extends Error {
  readonly status: number
  readonly detail: string
  constructor(status: number, detail: string) {
    super(`API ${status}: ${detail}`)
    this.name = "ApiError"
    this.status = status
    this.detail = detail
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init)
  if (!res.ok) {
    // FastAPI wrong-phase errors are 409 with {"detail": "..."}. Fall back to
    // statusText if the body isn't the expected JSON shape.
    let detail = res.statusText
    try {
      const body = (await res.json()) as { detail?: unknown }
      if (typeof body?.detail === "string") detail = body.detail
    } catch {
      // non-JSON error body — keep statusText
    }
    throw new ApiError(res.status, detail)
  }
  return (await res.json()) as T
}

export const api = {
  status: (signal?: AbortSignal) => request<Status>("/status", { signal }),
  // Start preserves the approved roster and begins recording.
  startRecording: () =>
    request<StartedResponse>("/recording/start", { method: "POST" }),
  // POST /stop returns the same body shape as GET /report (contract §/stop).
  stopRecording: () =>
    request<Report>("/recording/stop", { method: "POST" }),
  report: () => request<Report>("/report"),
  // T11/B5: runtime detection-confidence — an advanced setup control. Returns the
  // updated detector_control; a changed value restarts enrolment backend-side.
  setDetectionConfidence: (confidence: number) =>
    request<DetectorControl>("/settings/detection-confidence", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confidence }),
    }),
  // Deliberately rebuild catalogue links from the masks now visible in setup.
  // New Recording never calls this, so its previous successful links persist.
  relinkCurrentMasks: () =>
    request<DetectorControl>("/settings/relink", { method: "POST" }),
  streamUrl: `${BASE}/stream`, // for <img src={api.streamUrl}>
}
