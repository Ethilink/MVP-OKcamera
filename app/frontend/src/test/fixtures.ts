import { http, HttpResponse, type HttpHandler } from "msw"
import { BASE } from "@/api/client"
import type { Report, Status } from "@/api/types"

// Scripted Status/Report fixtures mirroring api-contract.md. Used by T06/T07
// component tests AND the pre-backend dev page (AC6). Times are fixed strings —
// no wall clock — so fixtures stay deterministic.

const STARTED_AT = "2026-07-20T10:31:04+02:00"
const STOPPED_AT = "2026-07-20T10:36:40+02:00"

const MODEL = "scenario-0.1"

/** setup phase, id-set not yet stable → Start must stay disabled. */
export const setupUnstable: Status = {
  phase: "setup",
  capture_health: "ok",
  model_version: MODEL,
  setup: { detected_count: 5, stable_for_s: 0.6 },
  recording: null,
}

/** setup phase, stable ≥ 2 s → Start enabled. */
export const setupStable: Status = {
  phase: "setup",
  capture_health: "ok",
  model_version: MODEL,
  setup: { detected_count: 5, stable_for_s: 3.2 },
  recording: null,
}

/** capture frozen → Start disabled + banner (T06 AC1). */
export const captureStalled: Status = {
  phase: "setup",
  capture_health: "stalled",
  model_version: MODEL,
  setup: { detected_count: 5, stable_for_s: 3.2 },
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
    on_table_count: 5,
    instruments: [1, 2, 3, 4, 5].map((id) => ({
      tracker_id: id,
      label: `Instrument ${id}`,
      on_table: true,
      off_since_s: null,
      pickup_count: 0,
    })),
  },
}

/** recording, instrument 3 off the table with off_since_s ticking. */
export const recordingOneOff: Status = {
  phase: "recording",
  capture_health: "ok",
  model_version: MODEL,
  setup: null,
  recording: {
    started_at: STARTED_AT,
    elapsed_s: 74.3,
    on_table_count: 4,
    instruments: [
      { tracker_id: 1, label: "Instrument 1", on_table: true, off_since_s: null, pickup_count: 1 },
      { tracker_id: 2, label: "Instrument 2", on_table: true, off_since_s: null, pickup_count: 0 },
      { tracker_id: 3, label: "Instrument 3", on_table: false, off_since_s: 13.2, pickup_count: 2 },
      { tracker_id: 4, label: "Instrument 4", on_table: true, off_since_s: null, pickup_count: 0 },
      { tracker_id: 5, label: "Instrument 5", on_table: true, off_since_s: null, pickup_count: 0 },
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
  setup: { detected_count: 5, stable_for_s: 3.2 },
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
