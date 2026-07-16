import { http, HttpResponse } from "msw"
import { BASE } from "@/api/client"
import type { InstrumentReport, Report, Status } from "@/api/types"

// Interactive dev backend for `npm run dev:msw` (and Chrome verification): a
// stateful stand-in that actually responds to Start/Stop, so the real App can be
// driven setup → recording → report before the backend exists. Unlike the
// component-test fixtures (deterministic, no clock) this one uses the wall clock
// so timers tick and the Start gate becomes stable after ~2 s. Dev-only.

const MODEL = "scenario-0.1"

// Distinct, GENERATED swatch "crops" so `dev:msw` can demonstrate the per-
// instrument cut-outs without a camera (the real crops come from
// `orc-demo --fake`). Each is a self-contained coloured SVG data-URI — a real
// asset, not an empty/placeholder <img src>.
const CROP_TINTS = ["#4b7a63", "#7a4b52", "#4b5f7a", "#7a6a4b", "#5f4b7a"]
function fakeCrop(id: number): string {
  const tint = CROP_TINTS[(id - 1) % CROP_TINTS.length]
  const svg =
    `<svg xmlns='http://www.w3.org/2000/svg' width='96' height='96'>` +
    `<rect width='96' height='96' fill='${tint}'/>` +
    `<rect x='30' y='18' width='36' height='60' rx='10' fill='rgba(255,255,255,0.16)'/>` +
    `<text x='48' y='58' font-family='sans-serif' font-size='34' font-weight='600' ` +
    `fill='rgba(255,255,255,0.92)' text-anchor='middle'>${id}</text></svg>`
  return `data:image/svg+xml,${encodeURIComponent(svg)}`
}

// Scripted choreography (seconds into the recording): instrument 1 is picked up
// and returned (present, closed window); instrument 3 leaves and never returns
// (missing, open window); the rest stay on the table.
// The fixed mask colours the real backend derives per roster instrument
// (backend/render.py ROSTER_PALETTE, keyed by index in the sorted roster —
// contiguous ids here, so slot = id - 1). Mirrored so `dev:msw` shows the same
// swatch↔mask pairing the real overlay does.
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
const maskColour = (id: number) => MASK_PALETTE[(id - 1) % MASK_PALETTE.length]

const SCRIPT: Record<number, { off: number; on: number | null }[]> = {
  1: [{ off: 5, on: 12 }],
  2: [],
  3: [{ off: 8, on: null }],
  4: [],
  5: [],
}
const IDS = [1, 2, 3, 4, 5]

let phase: Status["phase"] = "setup"
let setupStart = Date.now()
let recStart = 0
let frozenReport: Report | null = null

const nowS = (from: number) => (Date.now() - from) / 1000

function liveInstrument(id: number, t: number) {
  // The currently-open off window, if any, at time t.
  const open = SCRIPT[id].find((w) => t >= w.off && (w.on === null || t < w.on))
  return {
    tracker_id: id,
    label: `Instrument ${id}`,
    on_table: !open,
    // Live crop only while visible (on the table); off-table falls back to the
    // last-seen crop client-side, mirroring the real backend.
    thumbnail: open ? null : fakeCrop(id),
    // Keyed off the id, so an instrument that leaves and returns keeps its hue.
    colour: maskColour(id),
  }
}

function buildReport(duration: number): Report {
  const instruments: InstrumentReport[] = IDS.map((id) => {
    const usage = SCRIPT[id]
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
      const instruments = IDS.map((id) => liveInstrument(id, t))
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
      })
    }
    // setup / finished: the id-set is "stable" once 2 s have passed here.
    // Generated swatch crops stand in for the real camera crops (which come from
    // `orc-demo --fake`), so the constellation + list show distinct cut-outs.
    return HttpResponse.json<Status>({
      phase,
      capture_health: "ok",
      model_version: MODEL,
      setup: {
        detected_count: 5,
        stable_for_s: nowS(setupStart),
        detections: IDS.map((id) => ({
          tracker_id: id,
          label: `Instrument ${id}`,
          thumbnail: fakeCrop(id),
        })),
      },
      recording: null,
    })
  }),
  http.post(`${BASE}/recording/start`, () => {
    phase = "recording"
    recStart = Date.now()
    frozenReport = null
    return HttpResponse.json({ started_at: new Date(recStart).toISOString() })
  }),
  http.post(`${BASE}/recording/stop`, () => {
    frozenReport = buildReport(nowS(recStart))
    phase = "finished"
    setupStart = Date.now() // re-arm the gate for run 2
    return HttpResponse.json(frozenReport)
  }),
  http.get(`${BASE}/report`, () =>
    frozenReport
      ? HttpResponse.json(frozenReport)
      : HttpResponse.json({ detail: "no finished recording" }, { status: 409 })
  ),
]
