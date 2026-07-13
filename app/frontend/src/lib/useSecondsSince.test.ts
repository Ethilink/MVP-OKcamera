import { renderHook } from "@testing-library/react"
import { afterEach, expect, test, vi } from "vitest"
import { useSecondsSince } from "./useSecondsSince"

afterEach(() => vi.restoreAllMocks())

test("returns ~0 right after the anchor changes, then accrues wall-clock", () => {
  let now = 1000
  vi.spyOn(performance, "now").mockImplementation(() => now)

  const anchorA = { poll: 1 }
  const { result, rerender } = renderHook(
    ({ anchor }) => useSecondsSince(anchor, true),
    { initialProps: { anchor: anchorA } }
  )
  expect(result.current).toBe(0)

  // same anchor identity → delta accrues with the clock
  now = 1500
  rerender({ anchor: anchorA })
  expect(result.current).toBeCloseTo(0.5, 5)

  // a fresh poll (new object) → re-anchors back to 0
  now = 2000
  rerender({ anchor: { poll: 2 } })
  expect(result.current).toBe(0)
})

test("never returns a negative delta", () => {
  let now = 5000
  vi.spyOn(performance, "now").mockImplementation(() => now)
  const anchor = {}
  const { result, rerender } = renderHook(() => useSecondsSince(anchor, true))
  now = 4000 // clock went backwards (shouldn't happen, but must not go negative)
  rerender()
  expect(result.current).toBe(0)
})
