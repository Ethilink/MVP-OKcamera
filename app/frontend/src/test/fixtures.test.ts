import { describe, expect, test } from "vitest"
import {
  demoReport,
  finishedStatus,
  recordingOneOff,
  scriptedHandlers,
  setupStable,
  setupUnstable,
} from "./fixtures"
import { api } from "@/api/client"
import { server } from "./server"

describe("fixtures", () => {
  test("scriptedHandlers steps one response per poll, then holds the last (AC5)", async () => {
    server.use(...scriptedHandlers([setupUnstable, setupStable, recordingOneOff]))
    expect(await api.status()).toEqual(setupUnstable)
    expect(await api.status()).toEqual(setupStable)
    expect(await api.status()).toEqual(recordingOneOff)
    expect(await api.status()).toEqual(recordingOneOff) // holds last thereafter
  })

  test("scriptedHandlers serves start/stop/report from the report (AC5)", async () => {
    server.use(...scriptedHandlers([finishedStatus], demoReport))
    expect(await api.stopRecording()).toEqual(demoReport)
    expect(await api.report()).toEqual(demoReport)
    expect(await api.startRecording()).toEqual({
      started_at: demoReport.started_at,
    })
  })

  test("demoReport covers both completeness values, an open window, empty usage (AC5)", () => {
    const completeness = demoReport.instruments.map((i) => i.completeness)
    expect(completeness).toContain("present")
    expect(completeness).toContain("missing")

    const missing = demoReport.instruments.find(
      (i) => i.completeness === "missing",
    )
    expect(missing?.usage.at(-1)?.on_s).toBeNull() // open (never-returned) window

    const neverPickedUp = demoReport.instruments.find(
      (i) => i.usage.length === 0,
    )
    expect(neverPickedUp).toBeDefined()
  })
})
