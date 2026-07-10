import { http, HttpResponse } from "msw"
import { describe, expect, test } from "vitest"
import { ApiError, api, BASE } from "./client"
import { server } from "@/test/server"
import { demoReport, setupStable } from "@/test/fixtures"

// AC2: api.* methods hit the right method+path and parse bodies; 409 → ApiError.
describe("api client", () => {
  test("status() GETs /status and returns the parsed body", async () => {
    server.use(http.get(`${BASE}/status`, () => HttpResponse.json(setupStable)))
    expect(await api.status()).toEqual(setupStable)
  })

  test("startRecording() POSTs /recording/start", async () => {
    let method = ""
    server.use(
      http.post(`${BASE}/recording/start`, ({ request }) => {
        method = request.method
        return HttpResponse.json({ started_at: setupStable.model_version })
      }),
    )
    await api.startRecording()
    expect(method).toBe("POST")
  })

  test("stopRecording() POSTs /recording/stop and returns a Report", async () => {
    server.use(
      http.post(`${BASE}/recording/stop`, () => HttpResponse.json(demoReport)),
    )
    expect(await api.stopRecording()).toEqual(demoReport)
  })

  test("report() GETs /report", async () => {
    server.use(http.get(`${BASE}/report`, () => HttpResponse.json(demoReport)))
    expect(await api.report()).toEqual(demoReport)
  })

  test("a 409 rejects with ApiError carrying detail", async () => {
    server.use(
      http.get(`${BASE}/report`, () =>
        HttpResponse.json({ detail: "no finished recording" }, { status: 409 }),
      ),
    )
    await expect(api.report()).rejects.toMatchObject({
      name: "ApiError",
      status: 409,
      detail: "no finished recording",
    })
    await expect(api.report()).rejects.toBeInstanceOf(ApiError)
  })

  test("streamUrl is absolute against BASE", () => {
    expect(api.streamUrl).toBe(`${BASE}/stream`)
  })
})
