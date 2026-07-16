import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, expect, test, vi } from "vitest"
import { VideoFeed } from "./VideoFeed"

afterEach(() => vi.useRealTimers())

// D16 / T06 AC7 (RTL half): MSW can't intercept the MJPEG <img> load, so the
// panel is what renders once the <img> errors. Assert the fallback, not a frame.
test("swaps in the 'no stream (dev mode)' panel when the <img> errors", () => {
  render(<VideoFeed />)
  const img = screen.getByAltText("live camera feed")
  fireEvent.error(img)
  expect(screen.getByLabelText("no stream (dev mode)")).toBeInTheDocument()
  expect(screen.queryByAltText("live camera feed")).not.toBeInTheDocument()
})

test("automatically retries the stream with a cache-busting URL", async () => {
  vi.useFakeTimers()
  render(<VideoFeed />)
  const firstSrc = screen.getByAltText("live camera feed").getAttribute("src")

  fireEvent.error(screen.getByAltText("live camera feed"))
  expect(screen.getByLabelText("no stream (dev mode)")).toBeInTheDocument()

  await act(async () => {
    await vi.advanceTimersByTimeAsync(1_000)
  })

  const retry = screen.getByAltText("live camera feed")
  expect(retry.getAttribute("src")).not.toBe(firstSrc)
  expect(retry.getAttribute("src")).toMatch(/[?&]retry=1(?:&|$)/)
})
