import { http, HttpResponse } from "msw"
import { BASE } from "@/api/client"
import type {
  Detection,
  DetectorControl,
  InstrumentReport,
  Report,
  Status,
} from "@/api/types"

// Interactive dev backend for `npm run dev:msw` (and Chrome verification): a
// stateful stand-in that actually responds to Prepare/Start/Stop/confidence, so
// the real App can be driven setup → recording → report before the backend
// exists. Unlike the component-test fixtures (deterministic, no clock) this one
// uses the wall clock so timers tick and the exact-catalog gate (T11) cycles
// through its states on its own. Dev-only.

const MODEL = "scenario-0.1"

// The fixed catalog colours the real backend derives per specimen (backend/
// render.py CATALOG_PALETTE, keyed by index in the sorted catalog {1..8}, so
// slot = id - 1). Mirrored so `dev:msw` shows the same swatch↔mask pairing the
// real overlay does. Unknown/recognising → neutral gray (UNKNOWN_COLOUR).
const CATALOG_PALETTE = [
  "#4285f4",
  "#34a853",
  "#fbbc04",
  "#ea4335",
  "#a142f4",
  "#24c1e0",
  "#ff6d01",
  "#f538a0",
]
const UNKNOWN_COLOUR = "#9ca3af"
const catalogColour = (id: number) => CATALOG_PALETTE[(id - 1) % CATALOG_PALETTE.length]

// Distinct, GENERATED swatch "crops" so `dev:msw` can demonstrate the per-
// instrument cut-outs without a camera. Each is a self-contained coloured SVG
// data URI — a valid generated preview source for the dev-only fixture.
function fakeCrop(id: number, tint: string): string {
  const svg =
    `<svg xmlns='http://www.w3.org/2000/svg' width='96' height='96'>` +
    `<rect width='96' height='96' fill='${tint}'/>` +
    `<rect x='30' y='18' width='36' height='60' rx='10' fill='rgba(255,255,255,0.16)'/>` +
    `<text x='48' y='58' font-family='sans-serif' font-size='34' font-weight='600' ` +
    `fill='rgba(255,255,255,0.92)' text-anchor='middle'>${id}</text></svg>`
  return `data:image/svg+xml,${encodeURIComponent(svg)}`
}

const CATALOG = [1, 2, 3, 4, 5, 6, 7, 8]
const FOREIGN_ID = 1042

// Recording choreography (seconds into the recording): instrument 1 is picked up
// and returned (present, closed window); instrument 3 leaves and never returns
// (missing, open window); the rest stay on the table.
const SCRIPT: Record<number, { off: number; on: number | null }[]> = {
  1: [{ off: 5, on: 12 }],
  3: [{ off: 8, on: null }],
}

let phase: Status["phase"] = "setup"
let setupStart = Date.now()
let recStart = 0
let frozenReport: Report | null = null
let confidence = 0.5

const nowS = (from: number) => (Date.now() - from) / 1000

function detectorControl(): DetectorControl {
  return {
    confidence,
    default_confidence: 0.5,
    minimum: 0.3,
    maximum: 0.9,
    step: 0.05,
  }
}

// The scripted foreign object: appears 4 s into a setup pass and leaves at 9 s,
// reading "recognising" (spinner) for its first 1.5 s, then settled "unknown".
// The gate is permissive: while it resolves it reads "recognising"; once settled
// as Unknown it no longer blocks, so the gate cycles hold_steady → ready →
// recognising → hold_steady → ready while the Unknown tile still shows.
const FOREIGN_APPEAR = 4
const FOREIGN_SETTLE = 5.5
const FOREIGN_LEAVE = 9

function buildSetup(elapsed: number) {
  const detections: Detection[] = CATALOG.map((id) => ({
    tracker_id: id,
    state: "recognised" as const,
    label: `Instrument ${id}`,
    colour: catalogColour(id),
    thumbnail: fakeCrop(id, catalogColour(id)),
  }))

  const foreignPresent = elapsed >= FOREIGN_APPEAR && elapsed < FOREIGN_LEAVE
  const foreignResolving = elapsed >= FOREIGN_APPEAR && elapsed < FOREIGN_SETTLE
  if (foreignPresent) {
    detections.push(
      foreignResolving
        ? {
            tracker_id: FOREIGN_ID,
            state: "recognising",
            label: "",
            colour: UNKNOWN_COLOUR,
            thumbnail: null,
          }
        : {
            tracker_id: FOREIGN_ID,
            state: "unknown",
            label: "Unknown",
            colour: UNKNOWN_COLOUR,
            thumbnail: fakeCrop(0, UNKNOWN_COLOUR),
          },
    )
  }

  // The id-set is unchanged since the last appear/leave edge — that anchors the
  // stability clock, exactly as the backend's readiness does.
  const stableSince = !foreignPresent && elapsed >= FOREIGN_LEAVE ? FOREIGN_LEAVE : 0
  const idsetChangedAt = foreignPresent ? FOREIGN_APPEAR : stableSince
  const stable_for_s = Math.max(0, elapsed - idsetChangedAt)

  const recognised_count = detections.filter((d) => d.state === "recognised").length
  const resolving_count = detections.filter((d) => d.state === "recognising").length
  const unknown_count = detections.filter((d) => d.state === "unknown").length
  const ready =
    recognised_count > 0 &&
    resolving_count === 0 &&
    stable_for_s >= 2

  let blocking_reason: string | null = null
  if (!ready) {
    if (resolving_count > 0) blocking_reason = "recognising"
    else if (recognised_count === 0) blocking_reason = "missing_instruments"
    else blocking_reason = "hold_steady"
  }

  return {
    detected_count: detections.length,
    expected_count: CATALOG.length,
    recognised_count,
    resolving_count,
    unknown_count,
    stable_for_s,
    ready,
    blocking_reason,
    detections,
  }
}

function liveInstrument(id: number, t: number) {
  const windows = SCRIPT[id] ?? []
  const open = windows.find((w) => t >= w.off && (w.on === null || t < w.on))
  return {
    tracker_id: id,
    label: `Instrument ${id}`,
    on_table: !open,
    // Live crop only while visible (on the table); off-table falls back to the
    // last-seen crop client-side, mirroring the real backend.
    thumbnail: open ? null : fakeCrop(id, catalogColour(id)),
    // Keyed off the id and the fixed catalog, so a returning instrument keeps its hue.
    colour: catalogColour(id),
  }
}

function buildReport(duration: number): Report {
  const instruments: InstrumentReport[] = CATALOG.map((id) => {
    const usage = (SCRIPT[id] ?? [])
      .filter((w) => w.off < duration)
      .map((w) => ({
        off_s: w.off,
        on_s: w.on !== null && w.on <= duration ? w.on : null,
      }))
    const last = usage[usage.length - 1]
    return {
      tracker_id: id,
      label: `Instrument ${id}`,
      completeness: last && last.on_s === null ? "missing" : "present",
      usage,
    }
  })
  return {
    started_at: new Date(recStart).toISOString(),
    stopped_at: new Date().toISOString(),
    duration_s: duration,
    model_version: MODEL,
    instruments,
  }
}

export const devHandlers = [
  http.get(`${BASE}/status`, () => {
    if (phase === "recording") {
      const t = nowS(recStart)
      const instruments = CATALOG.map((id) => liveInstrument(id, t))
      return HttpResponse.json<Status>({
        phase,
        capture_health: "ok",
        model_version: MODEL,
        setup: null,
        recording: {
          started_at: new Date(recStart).toISOString(),
          elapsed_s: t,
          instruments,
        },
        detector_control: detectorControl(),
      })
    }
    // setup / finished: the gate cycles on the wall clock since initialisation or
    // the last changed confidence (the only action that resets enrolment).
    return HttpResponse.json<Status>({
      phase,
      capture_health: "ok",
      model_version: MODEL,
      setup: buildSetup(nowS(setupStart)),
      recording: null,
      detector_control: detectorControl(),
    })
  }),
  http.post(`${BASE}/recording/start`, () => {
    // Server-enforced exact-catalog gate (D3): refuse unless setup is ready.
    if (phase === "recording") {
      return HttpResponse.json({ detail: "already recording" }, { status: 409 })
    }
    if (!buildSetup(nowS(setupStart)).ready) {
      return HttpResponse.json({ detail: "setup is not ready" }, { status: 409 })
    }
    phase = "recording"
    recStart = Date.now()
    frozenReport = null
    return HttpResponse.json({ started_at: new Date(recStart).toISOString() })
  }),
  http.post(`${BASE}/recording/stop`, () => {
    frozenReport = buildReport(nowS(recStart))
    phase = "finished"
    return HttpResponse.json(frozenReport)
  }),
  http.get(`${BASE}/report`, () =>
    frozenReport
      ? HttpResponse.json(frozenReport)
      : HttpResponse.json({ detail: "no finished recording" }, { status: 409 }),
  ),
  http.patch(`${BASE}/settings/detection-confidence`, async ({ request }) => {
    const body = (await request.json()) as { confidence: number }
    if (body.confidence < 0.3 || body.confidence > 0.9) {
      return HttpResponse.json({ detail: "out of range" }, { status: 422 })
    }
    if (phase === "recording") {
      return HttpResponse.json(
        { detail: "cannot change confidence during recording" },
        { status: 409 },
      )
    }
    // A changed value restarts enrolment (re-arm the gate); a no-op returns as-is.
    if (body.confidence !== confidence) {
      confidence = body.confidence
      setupStart = Date.now()
    }
    return HttpResponse.json(detectorControl())
  }),
]
