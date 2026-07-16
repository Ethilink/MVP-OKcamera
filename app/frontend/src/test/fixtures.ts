import { http, HttpResponse, type HttpHandler } from "msw"
import { BASE } from "@/api/client"
import type { Report, Status } from "@/api/types"

// Scripted Status/Report fixtures mirroring api-contract.md. Used by T06/T07
// component tests AND the pre-backend dev page (AC6). Times are fixed strings —
// no wall clock — so fixtures stay deterministic.

const STARTED_AT = "2026-07-20T10:31:04+02:00"
const STOPPED_AT = "2026-07-20T10:36:40+02:00"

const MODEL = "scenario-0.1"

// A 1×1 transparent GIF data URI — a valid <img src> for tests that assert the
// tiles render real crops, without shipping a real JPEG into the fixtures.
const PIXEL =
  "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"

// The fixed mask colours the backend hands out (T10): a palette slot per
// instrument, keyed by its position in the frozen roster, so the panel swatch is
// always the same hex the video overlay draws that instrument's mask with.
// Mirrors `backend/render.py`'s ROSTER_PALETTE. The hexes themselves are a
// backend TUNABLE — tests must read a row's colour off the fixture rather than
// hard-coding one, or they break the next time the palette is retuned.
const MASK_PALETTE = [
  "#4285f4",
  "#34a853",
  "#fbbc04",
  "#ea4335",
  "#a142f4",
  "#24c1e0",
  "#ff6d01",
  "#f538a0",
]

/**
 * The mask colour for a roster instrument, mirroring `roster_colour(...)`: the
 * palette slot at its index in the sorted roster. The fixtures roster ids 1..n
 * contiguously, so the index is simply `id - 1`. Colour follows the *id*, which
 * is what lets a returned instrument regain its original colour.
 */
function maskColour(trackerId: number): string {
  return MASK_PALETTE[(trackerId - 1) % MASK_PALETTE.length]
}

/** Five detections with real thumbnails — exercises the crop-tile path. */
const fiveDetections = [1, 2, 3, 4, 5].map((id) => ({
  tracker_id: id,
  label: `Instrument ${id}`,
  thumbnail: PIXEL,
}))

/** setup phase, id-set not yet stable → Start must stay disabled. */
export const setupUnstable: Status = {
  phase: "setup",
  capture_health: "ok",
  model_version: MODEL,
  setup: { detected_count: 5, stable_for_s: 0.6, detections: fiveDetections },
  recording: null,
}

/** setup phase, stable ≥ 2 s → Start enabled. */
export const setupStable: Status = {
  phase: "setup",
  capture_health: "ok",
  model_version: MODEL,
  setup: { detected_count: 5, stable_for_s: 3.2, detections: fiveDetections },
  recording: null,
}

/** capture frozen → Start disabled + banner (T06 AC1). */
export const captureStalled: Status = {
  phase: "setup",
  capture_health: "stalled",
  model_version: MODEL,
  setup: { detected_count: 5, stable_for_s: 3.2, detections: fiveDetections },
  recording: null,
}

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
      colour: maskColour(id),
    })),
  },
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
      { tracker_id: 1, label: "Instrument 1", on_table: true, thumbnail: PIXEL, colour: maskColour(1) },
      { tracker_id: 2, label: "Instrument 2", on_table: true, thumbnail: PIXEL, colour: maskColour(2) },
      // Off the table this frame → no live crop (the app shows its last-seen one).
      // Its colour is unchanged by the absence: the backend keys it off the id,
      // so the swatch still matches the mask when it comes back (T10).
      { tracker_id: 3, label: "Instrument 3", on_table: false, thumbnail: null, colour: maskColour(3) },
      { tracker_id: 4, label: "Instrument 4", on_table: true, thumbnail: PIXEL, colour: maskColour(4) },
      { tracker_id: 5, label: "Instrument 5", on_table: true, thumbnail: PIXEL, colour: maskColour(5) },
    ],
  },
}

/**
 * finished phase: setup block PRESENT & stable (D15 — table still observed for
 * run 2), recording null. Drives the run-2 gate; T06 AC4b needs it.
 */
export const finishedStatus: Status = {
  phase: "finished",
  capture_health: "ok",
  model_version: MODEL,
  setup: { detected_count: 5, stable_for_s: 3.2, detections: fiveDetections },
  recording: null,
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
 * the last entry thereafter (AC5). start/stop/report resolve against `report`.
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
  ]
}
