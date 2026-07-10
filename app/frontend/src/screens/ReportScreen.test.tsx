import { fireEvent, render, screen, within } from "@testing-library/react"
import { http, HttpResponse } from "msw"
import { expect, test, vi } from "vitest"
import { api } from "@/api/client"
import { BASE } from "@/api/client"
import { demoReport } from "@/test/fixtures"
import { server } from "@/test/server"
import { ReportScreen } from "./ReportScreen"

function serveReport(report = demoReport) {
  server.use(http.get(`${BASE}/report`, () => HttpResponse.json(report)))
}

// AC3: one row per instrument; missing → MISSING + flagged row + open bar;
// present-with-window → closed bar; never-picked-up → "never picked up".
test("AC3 renders one row per instrument with the right timeline + badge", async () => {
  serveReport()
  render(<ReportScreen onNewRecording={() => {}} />)

  // Instrument 1: present, closed window
  const one = (await screen.findByText("Instrument 1")).closest("div")!
  expect(within(one).getByText("PRESENT")).toBeInTheDocument()
  expect(within(one).getByLabelText(/off 1:01–1:24/)).toBeInTheDocument()

  // Instrument 2: present, never picked up
  const two = screen.getByText("Instrument 2").closest("div")!
  expect(within(two).getByText("PRESENT")).toBeInTheDocument()
  expect(within(two).getByText(/never picked up/i)).toBeInTheDocument()

  // Instrument 3: missing, open (never returned) bar
  const three = screen.getByText("Instrument 3").closest("div")!
  expect(within(three).getByText("MISSING")).toBeInTheDocument()
  expect(within(three).getByLabelText(/never returned/i)).toBeInTheDocument()

  // exactly three instrument rows
  expect(screen.getAllByText(/^Instrument \d$/)).toHaveLength(3)
})

// AC4: summary + New recording invokes the prop once and does NOT start a recording.
test("AC4 summary shows counts/model; New recording calls the prop, not the API", async () => {
  serveReport()
  const onNewRecording = vi.fn()
  const startSpy = vi.spyOn(api, "startRecording")
  render(<ReportScreen onNewRecording={onNewRecording} />)

  await screen.findByText("Instrument 1")
  // duration 336 → 5:36, 3 instruments, 1 missing, model_version
  expect(screen.getByText("5:36")).toBeInTheDocument()
  expect(screen.getByText(/3 instruments/)).toBeInTheDocument()
  expect(screen.getByText(/1 missing/)).toBeInTheDocument()
  expect(screen.getByText(demoReport.model_version)).toBeInTheDocument()

  fireEvent.click(screen.getByRole("button", { name: /new recording/i }))
  expect(onNewRecording).toHaveBeenCalledTimes(1)
  expect(startSpy).not.toHaveBeenCalled()
  startSpy.mockRestore()
})

// AC5: report fetch failing → non-crashing error state, for both 409 and 500.
test("AC5 409 → 'no finished recording' error state, still shows New recording", async () => {
  server.use(
    http.get(`${BASE}/report`, () =>
      HttpResponse.json({ detail: "no finished recording" }, { status: 409 })
    )
  )
  render(<ReportScreen onNewRecording={() => {}} />)
  expect(
    await screen.findByText(/no finished recording/i)
  ).toBeInTheDocument()
  expect(
    screen.getByRole("button", { name: /new recording/i })
  ).toBeInTheDocument()
})

test("AC5 generic failure → non-crashing error state", async () => {
  server.use(http.get(`${BASE}/report`, () => HttpResponse.error()))
  render(<ReportScreen onNewRecording={() => {}} />)
  expect(await screen.findByText(/could not load the report/i)).toBeInTheDocument()
})
