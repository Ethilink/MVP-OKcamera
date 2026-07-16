import { expect, test } from "vitest"
import { formatClock } from "./format"

test("formatClock renders mm:ss, flooring fractional seconds", () => {
  expect(formatClock(0)).toBe("0:00")
  expect(formatClock(9)).toBe("0:09")
  expect(formatClock(74.3)).toBe("1:14")
  expect(formatClock(336)).toBe("5:36")
})

test("formatClock never goes negative", () => {
  expect(formatClock(-5)).toBe("0:00")
})
