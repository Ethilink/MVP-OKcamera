import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { http } from "msw"
import { expect, test } from "vitest"
import { BASE } from "@/api/client"
import type { Status } from "@/api/types"
import {
  captureStalled,
  recordingOneOff,
  setupMissing,
  setupReadyUnstableClock,
  setupRecognising,
  setupStable,
  setupUnknownObject,
  setupUnstable,
} from "@/test/fixtures"
import { server } from "@/test/server"
import { LiveScreen } from "./LiveScreen"

function renderLive(
  status: Status | null,
  overrides: Partial<Parameters<typeof LiveScreen>[0]> = {},
) {
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

const track = () => screen.getByRole("button", { name: /^track$/i })

// --- F1: the Start gate is the backend's `setup.ready`, not a local calc ---

test("F1 ready:true → Track enabled", () => {
  renderLive(setupStable)
  expect(track()).toBeEnabled()
})

test("F1 ready:false disables Track even with a stable count present", () => {
  // detected_count 9, stable_for_s 3.0 — the OLD local gate (count≥1 ∧ stable≥2s)
  // would ENABLE this. The backend says ready:false, and that must win (F1).
  renderLive(setupUnknownObject)
  expect(track()).toBeDisabled()
})

test("F1 ready:true enables Track even when the stability clock reads < 2 s", () => {
  // The converse discriminator: the OLD local gate (stable_for_s >= 2) would
  // DISABLE this (clock 0.4 s). Track follows `ready` alone, so it is enabled —
  // an `setup.ready && oldLocalGate` implementation would fail this.
  renderLive(setupReadyUnstableClock)
  expect(track()).toBeEnabled()
})

test("F1 stalled camera disables Track and shows the destructive banner", () => {
  renderLive(captureStalled)
  expect(track()).toBeDisabled()
  expect(screen.getByRole("alert")).toHaveTextContent(/stalled/i)
})

// --- F1: each blocking reason renders its operator copy under Track ---

test.each([
  ["recognising", setupRecognising, /recognising instruments/i],
  ["missing_instruments", setupMissing, /recognised 7 of 8 instruments/i],
  ["unknown_objects", setupUnknownObject, /remove 1 unknown object/i],
  ["hold_steady", setupUnstable, /hold the tray steady/i],
] as const)("F1 blocker %s renders its reason", (_name, status, re) => {
  renderLive(status)
  expect(screen.getByText(re)).toBeInTheDocument()
})

// --- Setup constellation renders (its own behaviour is covered by
//     DetectionConstellation.test); here we just assert it mounts in setup ---

test("setup shows the detection constellation", () => {
  renderLive(setupStable)
  // The ring's count pill announces the detected instruments (aria-label).
  expect(
    screen.getByRole("group", { name: /instruments? detected/i }),
  ).toBeInTheDocument()
})

// --- F3/D2: a confidence reset holds "Recognising" over the stale-ready gap ---

test("F3/D2 awaitingReset holds Recognising and disables Track despite a stale ready", () => {
  // App passes awaitingReset while the pre-reset (still-ready) status is held.
  renderLive(setupStable, { awaitingReset: true })
  expect(screen.getByText(/recognising instruments/i)).toBeInTheDocument()
  expect(screen.queryByText(/^ready$/i)).toBeNull()
  expect(track()).toBeDisabled()
})

// --- F4: Track held during a confidence PATCH; Advanced absent while recording ---

test("F4/9 Track is disabled while a confidence PATCH is pending", async () => {
  server.use(
    http.patch(`${BASE}/settings/detection-confidence`, () => new Promise(() => {})),
  )
  // Current confidence differs from default so Reset-to-default is available.
  const status: Status = {
    ...setupStable,
    detector_control: { ...setupStable.detector_control, confidence: 0.65 },
  }
  renderLive(status)
  expect(track()).toBeEnabled()

  fireEvent.click(screen.getByRole("button", { name: /advanced/i }))
  fireEvent.click(await screen.findByRole("button", { name: /reset to default/i }))

  await waitFor(() => expect(track()).toBeDisabled())
})

test("F4 the Advanced control is absent during recording", () => {
  renderLive(recordingOneOff)
  expect(screen.queryByRole("button", { name: /advanced/i })).toBeNull()
})

// --- recording layout (unchanged from T06/T07) ---

test("recordingOneOff → per-row crop + ON/OFF TABLE; Stop present; no count/off-secs/pickups", () => {
  renderLive(recordingOneOff)

  expect(screen.getByRole("button", { name: /^stop$/i })).toBeInTheDocument()
  expect(screen.queryByText(/\d+ on table/i)).toBeNull()
  expect(screen.queryByText(/pickups/i)).toBeNull()

  const items = screen.getAllByRole("listitem")
  expect(items).toHaveLength(5)

  const three = items.find((li) => within(li).queryByText("Instrument 3"))!
  expect(within(three).getByText("OFF TABLE")).toBeInTheDocument()
  expect(within(three).queryByRole("img")).toBeNull()

  const one = items.find((li) => within(li).queryByText("Instrument 1"))!
  expect(within(one).getByText("ON TABLE")).toBeInTheDocument()
  expect(within(one).getByRole("img", { name: "Instrument 1" })).toBeInTheDocument()
})

test("recording rows are sorted by tracker_id regardless of payload order", () => {
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
