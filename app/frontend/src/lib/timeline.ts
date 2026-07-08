import type { UsageWindow } from "@/api/types"
import { formatClock } from "./format"

export interface Segment {
  leftPct: number
  widthPct: number
  open: boolean
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v))
}

/** A time (s) as a [0,100] percentage of the recording. Never NaN (dur 0 → 0). */
function toPct(t: number, duration_s: number): number {
  if (duration_s <= 0) return 0
  return clamp((t / duration_s) * 100, 0, 100)
}

/**
 * Off-table windows → positioned percentage segments on the recording track.
 * `on_s == null` → an open segment running to 100% ("never returned"). Backend
 * guarantees windows are sorted + non-overlapping (api-contract §/report); we
 * assert that in a dev guard rather than re-sorting/merging. Degenerate input
 * (duration 0, a window past duration) clamps into [0,100] and never yields NaN.
 */
export function usageSegments(
  duration_s: number,
  usage: UsageWindow[]
): Segment[] {
  if (import.meta.env.DEV) {
    for (let i = 1; i < usage.length; i++) {
      const prevEnd = usage[i - 1].on_s ?? Infinity
      if (usage[i].off_s < prevEnd) {
        console.warn(
          "usageSegments: windows must be sorted & non-overlapping (contract §/report invariants)",
          usage
        )
      }
    }
  }

  return usage.map((w) => {
    const leftPct = toPct(w.off_s, duration_s)
    if (w.on_s === null) {
      return { leftPct, widthPct: Math.max(0, 100 - leftPct), open: true }
    }
    const rightPct = toPct(w.on_s, duration_s)
    return { leftPct, widthPct: Math.max(0, rightPct - leftPct), open: false }
  })
}

// {1,2,5}×10ⁿ seconds, ascending. The range spans sub-second to ~decades so a
// fitting step always exists at BOTH ends: the {1,2,5} gaps are ≤2.5×, which is
// inside the 3× window (dur/6, dur/2] where a step yields 3–6 ticks — so the
// smallest step giving ≤6 ticks automatically gives ≥3 too (T07 frozen 3–6).
function niceSteps(): number[] {
  const out: number[] = []
  for (let exp = -3; exp <= 9; exp++) {
    for (const m of [1, 2, 5]) out.push(m * 10 ** exp)
  }
  return out
}

/** Smallest nice step whose 0-based grid yields ≤ 6 ticks across `duration_s`. */
function niceStep(duration_s: number): number {
  const steps = niceSteps()
  for (const step of steps) {
    if (Math.floor(duration_s / step) + 1 <= 6) return step
  }
  return steps[steps.length - 1]
}

/**
 * "Nice" mm:ss axis ticks: a {1,2,5}×10ⁿ s step chosen so there are ≤ 6 ticks
 * (typically 3–6). First tick at 0% ("0:00"); last is the final nice multiple
 * that fits (≤ 100%, need not hit it). Strictly monotonic.
 */
export function axisTicks(
  duration_s: number
): { pct: number; label: string }[] {
  if (duration_s <= 0) return [{ pct: 0, label: formatClock(0) }]

  const step = niceStep(duration_s)
  const ticks: { pct: number; label: string }[] = []
  // +ε guards against a float landing a hair under an exact final multiple.
  for (let t = 0; t <= duration_s + 1e-9; t += step) {
    ticks.push({ pct: toPct(t, duration_s), label: formatClock(t) })
  }
  return ticks
}
