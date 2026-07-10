import { expect, test } from "vitest"
import { axisTicks, usageSegments } from "./timeline"

// --- AC1: usageSegments ---

test("AC1 closed window → left/width percentages", () => {
  expect(usageSegments(100, [{ off_s: 20, on_s: 35 }])).toEqual([
    { leftPct: 20, widthPct: 15, open: false },
  ])
})

test("AC1 open window (on_s null) → width to 100, open:true", () => {
  expect(usageSegments(100, [{ off_s: 60, on_s: null }])).toEqual([
    { leftPct: 60, widthPct: 40, open: true },
  ])
})

test("AC1 empty usage → []", () => {
  expect(usageSegments(336, [])).toEqual([])
})

test("AC1 degenerate input clamps into [0,100], never NaN", () => {
  const zeroDur = usageSegments(0, [{ off_s: 5, on_s: 9 }])
  const past = usageSegments(100, [{ off_s: 150, on_s: 200 }])
  for (const seg of [...zeroDur, ...past]) {
    expect(Number.isNaN(seg.leftPct)).toBe(false)
    expect(Number.isNaN(seg.widthPct)).toBe(false)
    expect(seg.leftPct).toBeGreaterThanOrEqual(0)
    expect(seg.leftPct).toBeLessThanOrEqual(100)
    expect(seg.widthPct).toBeGreaterThanOrEqual(0)
    expect(seg.widthPct).toBeLessThanOrEqual(100)
  }
})

// --- AC2: axisTicks ---

function assertTicks(duration: number, expectedLabels: string[]) {
  const ticks = axisTicks(duration)
  expect(ticks.map((t) => t.label)).toEqual(expectedLabels)
  // 3–6 ticks, strictly monotonic, first 0%, last ≤ 100%
  expect(ticks.length).toBeGreaterThanOrEqual(3)
  expect(ticks.length).toBeLessThanOrEqual(6)
  expect(ticks[0]).toEqual({ pct: 0, label: "0:00" })
  expect(ticks[ticks.length - 1].pct).toBeLessThanOrEqual(100)
  for (let i = 1; i < ticks.length; i++) {
    expect(ticks[i].pct).toBeGreaterThan(ticks[i - 1].pct)
  }
}

test("AC2 awkward 13 s → step 5 (3 ticks)", () => {
  assertTicks(13, ["0:00", "0:05", "0:10"])
})

test("AC2 100 s → step 20 (6 ticks), mm:ss labels incl 1:20", () => {
  assertTicks(100, ["0:00", "0:20", "0:40", "1:00", "1:20", "1:40"])
})

test("AC2 336 s → step 100 (4 ticks)", () => {
  assertTicks(336, ["0:00", "1:40", "3:20", "5:00"])
})

// Regression: the frozen 3–6 bound must hold at BOTH extremes, not just the
// realistic middle — short positive durations and absurdly large ones alike.
test("AC2 stays within 3–6 ticks for extreme durations", () => {
  for (const d of [0.5, 1, 1.9, 2, 7, 3600, 30_000_000]) {
    const ticks = axisTicks(d)
    expect(ticks.length).toBeGreaterThanOrEqual(3)
    expect(ticks.length).toBeLessThanOrEqual(6)
    expect(ticks[0].pct).toBe(0)
    expect(ticks[ticks.length - 1].pct).toBeLessThanOrEqual(100)
    for (let i = 1; i < ticks.length; i++) {
      expect(ticks[i].pct).toBeGreaterThan(ticks[i - 1].pct)
    }
  }
})
