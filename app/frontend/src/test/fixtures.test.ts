import { describe, expect, test } from "vitest"
import {
  demoReport,
  finishedStatus,
  recordingAllOn,
  recordingOneOff,
  scriptedHandlers,
  setupStable,
  setupUnknownObject,
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

  test("scriptedHandlers serves the confidence PATCH (T11)", async () => {
    server.use(...scriptedHandlers([setupStable]))
    expect(await api.setDetectionConfidence(0.6)).toMatchObject({ confidence: 0.6 })
  })

  // T11/§8-10: the fixtures (and, at compile time, the generated schema they are
  // typed against) carry every new required field. This runtime check fails loudly
  // if a fixture is missing one; the schema half is enforced by `tsc` — `types.ts`
  // maps these fields off `schema.d.ts`, so a stale schema is a type error.
  test("fixtures carry every new T11 field (readiness, per-item identity, detector_control)", () => {
    const setup = setupUnknownObject.setup!
    expect(setup).toMatchObject({
      detected_count: expect.any(Number),
      expected_count: expect.any(Number),
      recognised_count: expect.any(Number),
      resolving_count: expect.any(Number),
      unknown_count: expect.any(Number),
      stable_for_s: expect.any(Number),
      ready: expect.any(Boolean),
      blocking_reason: "unknown_objects",
    })
    expect(setup.detections[0]).toMatchObject({
      tracker_id: expect.any(Number),
      state: expect.any(String),
      label: expect.any(String),
      colour: expect.any(String),
    })
    expect(setupStable.detector_control).toMatchObject({
      confidence: expect.any(Number),
      default_confidence: expect.any(Number),
      minimum: expect.any(Number),
      maximum: expect.any(Number),
      step: expect.any(Number),
    })
    expect(recordingAllOn.recording!.instruments[0].colour).toEqual(
      expect.any(String),
    )
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
