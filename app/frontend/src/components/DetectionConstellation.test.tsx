import { render, screen } from "@testing-library/react"
import { expect, test } from "vitest"
import type { Detection } from "@/api/types"
import { DetectionConstellation } from "./DetectionConstellation"

const PIXEL =
  "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"

function detection(
  id: number,
  thumbnail: string | null,
  state: Detection["state"] = "recognised",
): Detection {
  return {
    tracker_id: id,
    state,
    label: state === "recognised" ? `Instrument ${id}` : state === "unknown" ? "Unknown" : "",
    colour: state === "recognised" ? "#4285f4" : "#9ca3af",
    thumbnail,
  }
}

const base = { ready: false, stalled: false, connecting: false }

test("renders one crop image per detection thumbnail, labelled", () => {
  const detections = [1, 2, 3].map((id) => detection(id, PIXEL))
  render(<DetectionConstellation {...base} detectedCount={3} detections={detections} />)

  const imgs = screen.getAllByRole("img")
  expect(imgs).toHaveLength(3)
  expect(screen.getByAltText("Instrument 2")).toHaveAttribute("src", PIXEL)
})

test("falls back to icons (no imgs) when there are no detections", () => {
  render(<DetectionConstellation {...base} detectedCount={4} />)

  expect(screen.queryAllByRole("img")).toHaveLength(0)
})

test("a null thumbnail falls back to an icon for that one tile", () => {
  const detections = [detection(1, PIXEL), detection(2, null), detection(3, PIXEL)]
  render(<DetectionConstellation {...base} detectedCount={3} detections={detections} />)

  // two tiles have crops, the middle one falls back → only two <img>s.
  expect(screen.getAllByRole("img")).toHaveLength(2)
})

test("badge shows detectedCount even when it differs from the tile count", () => {
  // one-frame skew: session says 8, snapshot carried 3 crops.
  const detections = [1, 2, 3].map((id) => detection(id, PIXEL))
  render(<DetectionConstellation {...base} detectedCount={8} detections={detections} />)

  expect(screen.getByRole("group")).toHaveAttribute(
    "aria-label",
    "8 instruments detected"
  )
  expect(screen.getAllByRole("img")).toHaveLength(3)
})

test("renders all eight catalog instruments around the constellation", () => {
  const detections = Array.from({ length: 8 }, (_, i) => detection(i + 1, PIXEL))
  render(<DetectionConstellation {...base} detectedCount={8} detections={detections} />)

  expect(screen.getAllByTestId("constellation-tile")).toHaveLength(8)
})

test("does not add unknown or resolving detections to the constellation", () => {
  const detections = [
    ...Array.from({ length: 8 }, (_, i) => detection(i + 1, PIXEL)),
    detection(1042, PIXEL, "unknown"),
    detection(1043, null, "recognising"),
  ]
  render(<DetectionConstellation {...base} detectedCount={10} detections={detections} />)

  expect(screen.getAllByTestId("constellation-tile")).toHaveLength(8)
  expect(screen.queryByText("Unknown")).toBeNull()
  expect(screen.queryByText("Recognising…")).toBeNull()
})
