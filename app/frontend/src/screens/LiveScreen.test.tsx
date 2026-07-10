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
  const start = screen.getByRole("button", { name: /^start$/i })
  expect(start).toBeDisabled()
  expect(screen.getByText(/waiting for stable detections/i)).toBeInTheDocument()
})

test("AC1 stable ≥ 2 s → Start enabled", () => {
  renderLive(setupStable)
  expect(screen.getByRole("button", { name: /^start$/i })).toBeEnabled()
})

test("AC1 capture_health stalled → Start disabled + destructive banner", () => {
  renderLive(captureStalled)
  expect(screen.getByRole("button", { name: /^start$/i })).toBeDisabled()
  expect(screen.getByRole("alert")).toHaveTextContent(/stalled/i)
})

// --- AC3: the recording layout + instrument panel ---

test("AC3 recordingOneOff → off instrument shows OFF TABLE, off_since, pickups; others ON TABLE; header count", () => {
  renderLive(recordingOneOff)

  // header shows on_table_count and a Stop control
  expect(screen.getByText(/4 on table/i)).toBeInTheDocument()
  expect(screen.getByRole("button", { name: /^stop$/i })).toBeInTheDocument()

  const items = screen.getAllByRole("listitem")
  expect(items).toHaveLength(5)

  const three = items.find((li) => within(li).queryByText("Instrument 3"))!
  expect(within(three).getByText("OFF TABLE")).toBeInTheDocument()
  expect(within(three).getByText("13s")).toBeInTheDocument() // 13.2 floored
  expect(within(three).getByText(/2 pickups/)).toBeInTheDocument()

  const one = items.find((li) => within(li).queryByText("Instrument 1"))!
  expect(within(one).getByText("ON TABLE")).toBeInTheDocument()
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
