import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { http, HttpResponse } from "msw"
import { expect, test, vi } from "vitest"
import { BASE } from "@/api/client"
import type { Status } from "@/api/types"
import {
  catalogColour,
  demoReport,
  finishedStatus,
  recordingAllOn,
  recordingOneOff,
  setupStable,
} from "@/test/fixtures"
import App from "./App"
import { server } from "./test/server"

const STARTED = { started_at: "2026-07-20T10:31:04+02:00" }

// Poll fast so poll-driven transitions resolve well inside waitFor's window.
const POLL = 15

test("MSW intercepts a fetch through the test server", async () => {
  server.use(
    http.get(`${BASE}/status`, () => HttpResponse.json({ phase: "setup" })),
  )
  const res = await fetch(`${BASE}/status`)
  expect(res.status).toBe(200)
})

// AC2: Start POSTs /recording/start; the recording layout appears ONLY after a
// poll returns phase:"recording" (no optimistic flip). Backend modelled as
// setup-until-start.
test("AC2 Start posts /recording/start and switches on the recording poll", async () => {
  const startSpy = vi.fn()
  let started = false
  server.use(
    http.get(`${BASE}/status`, () =>
      HttpResponse.json(started ? recordingAllOn : setupStable),
    ),
    http.post(`${BASE}/recording/start`, () => {
      startSpy()
      started = true
      return HttpResponse.json(STARTED)
    }),
  )

  render(<App pollMs={POLL} />)

  const start = await screen.findByRole("button", { name: /^track$/i })
  await waitFor(() => expect(start).toBeEnabled())
  expect(screen.queryByRole("button", { name: /^stop$/i })).toBeNull()

  fireEvent.click(start)
  await waitFor(() => expect(startSpy).toHaveBeenCalledTimes(1))

  await waitFor(() =>
    expect(screen.getByRole("button", { name: /^stop$/i })).toBeInTheDocument(),
  )
  expect(screen.queryByRole("button", { name: /^track$/i })).toBeNull()
})

// AC4: Stop POSTs /recording/stop; on the finished poll (flag false) App routes
// to ReportScreen and LiveScreen is gone.
test("AC4 Stop routes to ReportScreen after the finished poll", async () => {
  const stopSpy = vi.fn()
  let stopped = false
  server.use(
    http.get(`${BASE}/status`, () =>
      HttpResponse.json(stopped ? finishedStatus : recordingAllOn),
    ),
    http.post(`${BASE}/recording/stop`, () => {
      stopSpy()
      stopped = true
      return HttpResponse.json(demoReport)
    }),
    http.get(`${BASE}/report`, () => HttpResponse.json(demoReport)),
  )

  render(<App pollMs={POLL} />)

  const stop = await screen.findByRole("button", { name: /^stop$/i })
  fireEvent.click(stop)
  await waitFor(() => expect(stopSpy).toHaveBeenCalledTimes(1))

  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: /new recording/i }),
    ).toBeInTheDocument(),
  )
  expect(screen.queryByRole("button", { name: /^stop$/i })).toBeNull()
})

// AC4b + AC4c: from finished/ReportScreen, "New recording" opens the run-2 setup
// layout (flag only — it does NOT call the API); it shows "Back to report"; the
// gate behaves like setup; Start restarts and clears the flag.
test("AC4b run-2 restart; AC4c Back to report", async () => {
  let started = false
  server.use(
    http.get(`${BASE}/status`, () =>
      HttpResponse.json(started ? recordingAllOn : finishedStatus),
    ),
    http.post(`${BASE}/recording/start`, () => {
      started = true
      return HttpResponse.json(STARTED)
    }),
    http.get(`${BASE}/report`, () => HttpResponse.json(demoReport)),
  )

  render(<App pollMs={POLL} />)

  // finished + no flag → ReportScreen
  const newRec = await screen.findByRole("button", { name: /new recording/i })
  fireEvent.click(newRec)

  // The run-2 setup layout appears (gate enabled — the finished payload is
  // ready), with Back to report present.
  const start = await screen.findByRole("button", { name: /^track$/i })
  await waitFor(() => expect(start).toBeEnabled())
  expect(
    screen.getByRole("button", { name: /back to report/i }),
  ).toBeInTheDocument()

  // AC4c: Back to report clears the flag → ReportScreen again (report preserved)
  fireEvent.click(screen.getByRole("button", { name: /back to report/i }))
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: /new recording/i }),
    ).toBeInTheDocument(),
  )

  // AC4b: back into setup, Start restarts; recording poll clears the flag → recording
  fireEvent.click(screen.getByRole("button", { name: /new recording/i }))
  fireEvent.click(await screen.findByRole("button", { name: /^track$/i }))
  await waitFor(() =>
    expect(screen.getByRole("button", { name: /^stop$/i })).toBeInTheDocument(),
  )
})

// T11/F2/5: a recognised instrument shows its fixed catalog colour on the
// recording row. The backend keys colour off the id and the frozen roster, so
// pressing Track never recolours it — the swatch is the same hex the mask uses.
test("F2/5 a recognised instrument shows its catalog colour on the recording row", async () => {
  let started = false
  server.use(
    http.get(`${BASE}/status`, () =>
      HttpResponse.json(started ? recordingAllOn : setupStable),
    ),
    http.post(`${BASE}/recording/start`, () => {
      started = true
      return HttpResponse.json(STARTED)
    }),
  )

  render(<App pollMs={POLL} />)

  const start = await screen.findByRole("button", { name: /^track$/i })
  await waitFor(() => expect(start).toBeEnabled())
  fireEvent.click(start)
  await waitFor(() =>
    expect(screen.getByRole("button", { name: /^stop$/i })).toBeInTheDocument(),
  )

  const row = screen
    .getAllByRole("listitem")
    .find((li) => within(li).queryByText("Instrument 1"))!
  const recordingColour = within(row).getByTestId("instrument-swatch").style
    .backgroundColor
  // It is the catalog colour for id 1 (never a literal — the palette is the
  // backend's to retune; read it through a probe element).
  const probe = document.createElement("span")
  probe.style.backgroundColor = catalogColour(1)
  expect(recordingColour).toBe(probe.style.backgroundColor)
})

// AC4c negative: a genuine phase:"setup" (run 1, no report) shows no Back-to-report.
test("AC4c no Back to report in a genuine setup", async () => {
  server.use(
    http.get(`${BASE}/status`, () => HttpResponse.json(setupStable)),
  )
  render(<App pollMs={POLL} />)
  await screen.findByRole("button", { name: /^track$/i })
  expect(
    screen.queryByRole("button", { name: /back to report/i }),
  ).toBeNull()
})

// AC5: poll failure mid-recording → banner + last panel stays; recovery clears it.
test("AC5 poll failure shows a banner, keeps the panel, clears on recovery", async () => {
  let n = 0
  server.use(
    http.get(`${BASE}/status`, () => {
      n += 1
      if (n === 2) return HttpResponse.error() // one failed poll mid-recording
      return HttpResponse.json(recordingAllOn)
    }),
  )

  render(<App pollMs={POLL} />)

  // panel is present. getAllByText: the experimental match-debug table
  // (feat/matching-tests) also renders the instrument label, so the name is
  // no longer unique on screen — assert it appears at all, not exactly once.
  await waitFor(() =>
    expect(screen.getAllByText("Instrument 1").length).toBeGreaterThan(0),
  )
  // banner appears on the failed poll, panel still there
  await waitFor(() =>
    expect(screen.getByRole("alert")).toHaveTextContent(/lost connection/i),
  )
  expect(screen.getAllByText("Instrument 1").length).toBeGreaterThan(0)
  // recovers → banner clears
  await waitFor(() => expect(screen.queryByRole("alert")).toBeNull())
})

// T10: an instrument that leaves the table and comes back keeps its original
// mask colour. The backend derives the colour from the id and the frozen roster
// and the linker re-emits the original id, so "regains its colour" holds across
// the absence; the panel's job is to stay a pure pass-through of that field
// (no caching it by row, no re-deriving it on a fresh poll).
test("T10 an instrument that goes absent and returns keeps its swatch colour", async () => {
  let status: Status = recordingAllOn
  server.use(http.get(`${BASE}/status`, () => HttpResponse.json(status)))

  const swatchOfThree = () => {
    const row = screen
      .getAllByRole("listitem")
      .find((li) => within(li).queryByText("Instrument 3"))!
    return within(row).getByTestId("instrument-swatch").style.backgroundColor
  }
  const badgeOfThree = (text: RegExp) =>
    waitFor(() =>
      expect(
        screen
          .getAllByRole("listitem")
          .find((li) => within(li).queryByText("Instrument 3"))!,
      ).toHaveTextContent(text),
    )

  render(<App pollMs={POLL} />)

  // On the table: note the colour the API actually served (never a literal —
  // the palette is the backend's to retune). getAllByText: the experimental
  // match-debug table (feat/matching-tests) also renders the label.
  await waitFor(() => expect(screen.getAllByText("Instrument 3").length).toBeGreaterThan(0))
  const before = swatchOfThree()
  expect(before).not.toBe("")

  // Picked up: the row stays, greys nothing about its identity.
  status = recordingOneOff
  await badgeOfThree(/OFF TABLE/)
  const whileOff = swatchOfThree()

  // Returned.
  status = recordingAllOn
  await badgeOfThree(/ON TABLE/)

  expect(whileOff).toBe(before)
  expect(swatchOfThree()).toBe(before)
})

// AC6: a 409 on Start surfaces non-fatally; no crash; polling continues.
test("AC6 409 on Start shows an inline error and keeps polling", async () => {
  server.use(
    http.get(`${BASE}/status`, () => HttpResponse.json(setupStable)),
    http.post(`${BASE}/recording/start`, () =>
      HttpResponse.json({ detail: "already recording" }, { status: 409 }),
    ),
  )

  render(<App pollMs={POLL} />)

  const start = await screen.findByRole("button", { name: /^track$/i })
  await waitFor(() => expect(start).toBeEnabled())
  fireEvent.click(start)

  await waitFor(() =>
    expect(screen.getByText(/already recording/i)).toBeInTheDocument(),
  )
  // still on the setup screen, still polling (Start remains)
  expect(screen.getByRole("button", { name: /^track$/i })).toBeInTheDocument()
})
