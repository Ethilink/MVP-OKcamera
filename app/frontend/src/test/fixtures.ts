import { http, HttpResponse, type HttpHandler } from "msw"
import { BASE } from "@/api/client"
import type { Detection, DetectorControl, Report, SetupStatus, Status } from "@/api/types"

// Scripted Status/Report fixtures mirroring api-contract.md (T11 contract). Used
// by component tests AND the pre-backend dev page. Times are fixed strings — no
// wall clock — so fixtures stay deterministic.

const STARTED_AT = "2026-07-20T10:31:04+02:00"
const STOPPED_AT = "2026-07-20T10:36:40+02:00"

const MODEL = "scenario-0.1"

// A 1×1 transparent GIF data URI — a valid <img src> for tests that assert the
// tiles render real previews, without shipping a real cutout into the fixtures.
const PIXEL =
  "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"

// The fixed catalog mask colours the backend hands out (T11/D5): a palette slot
// per specimen, keyed by its index in the *sorted catalog* (not the roster) — so
// the tile/panel swatch is always the same hex the video overlay draws that
// instrument's mask with, stable across partial rosters. Mirrors
// `backend/render.py`'s CATALOG_PALETTE. The hexes are a backend TUNABLE — tests
// must read a colour off the fixture rather than hard-coding one, or they break
// the next time the palette is retuned.
export const CATALOG_PALETTE = [
  "#4285f4",
  "#34a853",
  "#fbbc04",
  "#ea4335",
  "#a142f4",
  "#24c1e0",
  "#ff6d01",
  "#f538a0",
]
// Neutral gray for recognising/unknown detections (T11/D4). Mirrors
// `backend/render.py`'s UNKNOWN_COLOUR.
export const UNKNOWN_COLOUR = "#9ca3af"

/**
 * The catalog colour for a specimen id, mirroring `catalog_colour(...)`: the
 * palette slot at its index in the sorted catalog. The demo catalog is `{1..8}`,
 * so the index is simply `id - 1`. Colour follows the *id*, which is what lets a
 * returned instrument regain its original colour.
 */
export function catalogColour(specimenId: number): string {
  return CATALOG_PALETTE[(specimenId - 1) % CATALOG_PALETTE.length]
}

// Per-item detection builders under the single identity policy (T11/B6):
// recognised → Instrument N + catalog colour + crop; recognising → "" + gray +
// spinner; unknown → "Unknown" + gray. A raw id is NEVER rendered as a label.
export const recognised = (id: number, thumbnail: string | null = PIXEL): Detection => ({
  tracker_id: id,
  state: "recognised",
  label: `Instrument ${id}`,
  colour: catalogColour(id),
  thumbnail,
})
export const recognising = (id: number, thumbnail: string | null = null): Detection => ({
  tracker_id: id,
  state: "recognising",
  label: "",
  colour: UNKNOWN_COLOUR,
  thumbnail,
})
export const unknown = (id: number, thumbnail: string | null = PIXEL): Detection => ({
  tracker_id: id,
  state: "unknown",
  label: "Unknown",
  colour: UNKNOWN_COLOUR,
  thumbnail,
})

/** The eight recognised specimens — the normal demo tray. */
export const eightRecognised: Detection[] = [1, 2, 3, 4, 5, 6, 7, 8].map((id) =>
  recognised(id),
)

// The runtime detection-confidence control — always present on /status (T11/B5).
export const DETECTOR_CONTROL: DetectorControl = {
  confidence: 0.5,
  default_confidence: 0.5,
  minimum: 0.3,
  maximum: 0.9,
  step: 0.05,
}

/** Build a setup block from its detections + readiness, deriving the counts so a
 *  fixture can never disagree with itself. */
function setupBlock(
  detections: Detection[],
  opts: { stable_for_s: number; ready: boolean; blocking_reason: string | null },
): SetupStatus {
  const recognised_count = detections.filter((d) => d.state === "recognised").length
  const resolving_count = detections.filter((d) => d.state === "recognising").length
  const unknown_count = detections.filter((d) => d.state === "unknown").length
  return {
    detected_count: detections.length,
    expected_count: 8,
    recognised_count,
    resolving_count,
    unknown_count,
    stable_for_s: opts.stable_for_s,
    ready: opts.ready,
    blocking_reason: opts.blocking_reason,
    detections,
  }
}

function setupStatus(
  detections: Detection[],
  opts: {
    stable_for_s: number
    ready: boolean
    blocking_reason: string | null
    capture_health?: Status["capture_health"]
    phase?: "setup" | "finished"
  },
): Status {
  return {
    phase: opts.phase ?? "setup",
    capture_health: opts.capture_health ?? "ok",
    model_version: MODEL,
    setup: setupBlock(detections, opts),
    recording: null,
    detector_control: DETECTOR_CONTROL,
  }
}

// --- setup fixtures, one per readiness state (T11/D3) ---

/** 8/8 recognised, stable ≥ 2 s, healthy → the server's Start verdict is ready. */
export const setupStable: Status = setupStatus(eightRecognised, {
  stable_for_s: 3.2,
  ready: true,
  blocking_reason: null,
})

/** 8/8 recognised but not yet stable → blocked on hold_steady. */
export const setupUnstable: Status = setupStatus(eightRecognised, {
  stable_for_s: 0.6,
  ready: false,
  blocking_reason: "hold_steady",
})

/** ready:true while the stability clock reads < 2 s. Impossible from the real
 *  backend (ready implies ≥ 2 s stable), but it proves the frontend gates on
 *  `ready` ALONE — the old local `stable_for_s >= 2` reconstruction would have
 *  DISABLED Track here (F1 / §8-1). */
export const setupReadyUnstableClock: Status = setupStatus(eightRecognised, {
  stable_for_s: 0.4,
  ready: true,
  blocking_reason: null,
})

/** One specimen still resolving (spinner) → blocked on recognising. */
export const setupRecognising: Status = setupStatus(
  [...[1, 2, 3, 4, 5, 6, 7].map((id) => recognised(id)), recognising(1042)],
  { stable_for_s: 1.0, ready: false, blocking_reason: "recognising" },
)

/** Only seven of eight catalog specimens present → blocked on missing. */
export const setupMissing: Status = setupStatus(
  [1, 2, 3, 4, 5, 6, 7].map((id) => recognised(id)),
  { stable_for_s: 3.0, ready: false, blocking_reason: "missing_instruments" },
)

/** Eight recognised plus one settled foreign object → blocked on unknown. */
export const setupUnknownObject: Status = setupStatus(
  [...eightRecognised, unknown(1042)],
  { stable_for_s: 3.0, ready: false, blocking_reason: "unknown_objects" },
)

/** Capture frozen: perfect tray but ready is false and blocking_reason is null —
 *  health is folded into `ready` only. The frontend shows the stalled banner. */
export const captureStalled: Status = setupStatus(eightRecognised, {
  stable_for_s: 3.2,
  ready: false,
  blocking_reason: null,
  capture_health: "stalled",
})

/**
 * finished phase: setup block PRESENT & ready (D15 — table still observed for
 * run 2), recording null. Drives the run-2 gate.
 */
export const finishedStatus: Status = setupStatus(eightRecognised, {
  stable_for_s: 3.2,
  ready: true,
  blocking_reason: null,
  phase: "finished",
})

// --- recording fixtures ---

/** recording, everything on the table. */
export const recordingAllOn: Status = {
  phase: "recording",
  capture_health: "ok",
  model_version: MODEL,
  setup: null,
  recording: {
    started_at: STARTED_AT,
    elapsed_s: 12.0,
    instruments: [1, 2, 3, 4, 5].map((id) => ({
      tracker_id: id,
      label: `Instrument ${id}`,
      on_table: true,
      thumbnail: PIXEL,
      colour: catalogColour(id),
    })),
  },
  detector_control: DETECTOR_CONTROL,
}

/** recording, instrument 3 off the table. Analytics remain report-only. */
export const recordingOneOff: Status = {
  phase: "recording",
  capture_health: "ok",
  model_version: MODEL,
  setup: null,
  recording: {
    started_at: STARTED_AT,
    elapsed_s: 74.3,
    instruments: [
      { tracker_id: 1, label: "Instrument 1", on_table: true, thumbnail: PIXEL, colour: catalogColour(1) },
      { tracker_id: 2, label: "Instrument 2", on_table: true, thumbnail: PIXEL, colour: catalogColour(2) },
      // Off the table this frame → no live crop (the app shows its last-seen one).
      // Its colour is unchanged by the absence: the backend keys it off the id and
      // the fixed catalog, so the swatch still matches the mask when it returns.
      { tracker_id: 3, label: "Instrument 3", on_table: false, thumbnail: null, colour: catalogColour(3) },
      { tracker_id: 4, label: "Instrument 4", on_table: true, thumbnail: PIXEL, colour: catalogColour(4) },
      { tracker_id: 5, label: "Instrument 5", on_table: true, thumbnail: PIXEL, colour: catalogColour(5) },
    ],
  },
  detector_control: DETECTOR_CONTROL,
}

/**
 * Canned report: instrument 1 picked up and returned (present, closed window);
 * instrument 3 left and never came back (missing, open window); instrument 2
 * never picked up (present, empty usage). Exercises both completeness values,
 * an open (on_s: null) window, and the empty-usage case.
 */
export const demoReport: Report = {
  started_at: STARTED_AT,
  stopped_at: STOPPED_AT,
  duration_s: 336.0,
  model_version: MODEL,
  instruments: [
    {
      tracker_id: 1,
      label: "Instrument 1",
      completeness: "present",
      usage: [{ off_s: 61.0, on_s: 84.5 }],
    },
    {
      tracker_id: 2,
      label: "Instrument 2",
      completeness: "present",
      usage: [],
    },
    {
      tracker_id: 3,
      label: "Instrument 3",
      completeness: "missing",
      usage: [{ off_s: 190.2, on_s: null }],
    },
  ],
}

/**
 * MSW handlers that step a `sequence` of Status payloads one-per-poll and hold
 * the last entry thereafter (AC5). start/stop/report resolve against `report`;
 * the confidence PATCH resolves trivially so wiring that calls it doesn't 404.
 * A fresh call gives a fresh counter, so tests are isolated.
 */
export function scriptedHandlers(
  sequence: Status[],
  report: Report = demoReport,
): HttpHandler[] {
  let i = 0
  return [
    http.get(`${BASE}/status`, () => {
      const idx = Math.min(i, sequence.length - 1)
      i += 1
      return HttpResponse.json(sequence[idx])
    }),
    http.post(`${BASE}/recording/start`, () =>
      HttpResponse.json({ started_at: report.started_at }),
    ),
    http.post(`${BASE}/recording/stop`, () => HttpResponse.json(report)),
    http.get(`${BASE}/report`, () => HttpResponse.json(report)),
    http.patch(`${BASE}/settings/detection-confidence`, async ({ request }) => {
      const body = (await request.json()) as { confidence: number }
      return HttpResponse.json({ ...DETECTOR_CONTROL, confidence: body.confidence })
    }),
  ]
}
