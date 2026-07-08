import { fireEvent, render, screen } from "@testing-library/react"
import { expect, test } from "vitest"
import { VideoFeed } from "./VideoFeed"

// D16 / T06 AC7 (RTL half): MSW can't intercept the MJPEG <img> load, so the
// panel is what renders once the <img> errors. Assert the fallback, not a frame.
test("swaps in the 'no stream (dev mode)' panel when the <img> errors", () => {
  render(<VideoFeed />)
  const img = screen.getByAltText("live camera feed")
  fireEvent.error(img)
  expect(screen.getByLabelText("no stream (dev mode)")).toBeInTheDocument()
  expect(screen.queryByAltText("live camera feed")).not.toBeInTheDocument()
})
