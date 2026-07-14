import { render, screen, within } from "@testing-library/react"
import { expect, test } from "vitest"
import type { Status } from "@/api/types"
import {
  captureStalled,
  recordingOneOff,
  setupStable,
  setupUnstable,
} from "@/test/fixtures"
import { LiveScreen } from "./LiveScreen"

function renderLive(status: Status, overrides: Partial<Parameters<typeof LiveScreen>[0]> = {}) {
  return render(
    <LiveScreen
      status={status}
      error={null}
      showBackToReport={false}
      onBackToReport={() => {}}
      {...overrides}
    />
  )
}

// --- AC1: the Start gate (setup layout) ---

test("AC1 unstable id-set → Start disabled with a reason", () => {
  renderLive(setupUnstable)
  const start = screen.getByRole("button", { name: /^track$/i })
  expect(start).toBeDisabled()
  expect(screen.getByText(/hold steady/i)).toBeInTheDocument()
})

test("AC1 stable ≥ 2 s → Start enabled", () => {
  renderLive(setupStable)
  expect(screen.getByRole("button", { name: /^track$/i })).toBeEnabled()
})

test("AC1 capture_health stalled → Start disabled + destructive banner", () => {
  renderLive(captureStalled)
  expect(screen.getByRole("button", { name: /^track$/i })).toBeDisabled()
  expect(screen.getByRole("alert")).toHaveTextContent(/stalled/i)
})

// --- AC3: the recording layout + instrument panel ---

test("AC3 recordingOneOff → per-row crop + ON/OFF TABLE; Stop present; no count/off-secs/pickups", () => {
  renderLive(recordingOneOff)

  // A Stop control is present; the on-table count subtitle, the off-since
  // seconds and the pickups column are all gone.
  expect(screen.getByRole("button", { name: /^stop$/i })).toBeInTheDocument()
  expect(screen.queryByText(/\d+ on table/i)).toBeNull()
  expect(screen.queryByText(/pickups/i)).toBeNull()
  expect(screen.queryByText(/^13s$/)).toBeNull()

  const items = screen.getAllByRole("listitem")
  expect(items).toHaveLength(5)

  // Instrument 3 is off the table this frame (no live crop) → OFF TABLE + the
  // icon fallback (no <img>).
  const three = items.find((li) => within(li).queryByText("Instrument 3"))!
  expect(within(three).getByText("OFF TABLE")).toBeInTheDocument()
  expect(within(three).queryByRole("img")).toBeNull()

  // An on-table instrument shows ON TABLE and its live crop.
  const one = items.find((li) => within(li).queryByText("Instrument 1"))!
  expect(within(one).getByText("ON TABLE")).toBeInTheDocument()
  expect(within(one).getByRole("img", { name: "Instrument 1" })).toBeInTheDocument()
})

test("AC3 rows are sorted by tracker_id regardless of payload order", () => {
  const shuffled: Status = {
    ...recordingOneOff,
    recording: {
      ...recordingOneOff.recording!,
      instruments: [...recordingOneOff.recording!.instruments].reverse(),
    },
  }
  renderLive(shuffled)
  const labels = screen
    .getAllByRole("listitem")
    .map((li) => within(li).getByText(/^Instrument \d$/).textContent)
  expect(labels).toEqual([
    "Instrument 1",
    "Instrument 2",
    "Instrument 3",
    "Instrument 4",
    "Instrument 5",
  ])
})
