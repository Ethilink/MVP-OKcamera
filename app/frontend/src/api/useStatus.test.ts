import { act, renderHook } from "@testing-library/react"
import { http, HttpResponse } from "msw"
import { afterEach, describe, expect, test, vi } from "vitest"
import { useStatus } from "./useStatus"
import { api, BASE } from "./client"
import { server } from "@/test/server"
import { setupStable, setupUnstable } from "@/test/fixtures"

afterEach(() => vi.useRealTimers())

const tick = (ms: number) =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })

describe("useStatus", () => {
  test("polls at the interval and exposes the latest payload (AC3)", async () => {
    vi.useFakeTimers()
    const seq = [setupUnstable, setupStable]
    let i = 0
    server.use(
      http.get(`${BASE}/status`, () => {
        const s = seq[Math.min(i, seq.length - 1)]
        i += 1
        return HttpResponse.json(s)
      }),
    )
    const { result } = renderHook(() => useStatus(500))
    await tick(0) // immediate first poll
    expect(result.current.status).toEqual(setupUnstable)
    await tick(500)
    expect(result.current.status).toEqual(setupStable)
  })

  test("does not stack overlapping requests when a response is slow (AC3)", async () => {
    vi.useFakeTimers()
    let calls = 0
    let release!: () => void
    const gate = new Promise<void>((r) => (release = r))
    server.use(
      http.get(`${BASE}/status`, async () => {
        calls += 1
        await gate // never resolves during the assertions
        return HttpResponse.json(setupStable)
      }),
    )
    renderHook(() => useStatus(500))
    await tick(0) // first poll starts and hangs
    await tick(500)
    await tick(500)
    expect(calls).toBe(1) // no overlap while one is in flight
    release()
  })

  test("a failed poll keeps last good status + sets error; recovery clears it (AC4)", async () => {
    vi.useFakeTimers()
    const outcomes = ["ok", "fail", "ok"]
    let i = 0
    server.use(
      http.get(`${BASE}/status`, () => {
        const o = outcomes[Math.min(i, outcomes.length - 1)]
        i += 1
        return o === "ok"
          ? HttpResponse.json(setupStable)
          : HttpResponse.json({ detail: "boom" }, { status: 500 })
      }),
    )
    const { result } = renderHook(() => useStatus(500))
    await tick(0)
    expect(result.current.status).toEqual(setupStable)
    expect(result.current.error).toBeNull()

    await tick(500) // failed poll
    expect(result.current.error).not.toBeNull()
    expect(result.current.status).toEqual(setupStable) // last good kept

    await tick(500) // recovery
    expect(result.current.error).toBeNull()
  })

  test("times out a hung request, retries, and aborts pending work on cleanup", async () => {
    vi.useFakeTimers()
    const signals: AbortSignal[] = []
    const statusSpy = vi.spyOn(api, "status").mockImplementation((signal) => {
      if (signal) signals.push(signal)
      return new Promise((_, reject) => {
        signal?.addEventListener(
          "abort",
          () => reject(new DOMException("Aborted", "AbortError")),
          { once: true },
        )
      })
    })

    const { result, unmount } = renderHook(() => useStatus(500, 250))
    await tick(0)
    expect(statusSpy).toHaveBeenCalledTimes(1)

    await tick(250)
    expect(result.current.error).not.toBeNull()

    await tick(250)
    expect(statusSpy).toHaveBeenCalledTimes(2)
    expect(signals[1]?.aborted).toBe(false)

    unmount()
    expect(signals[1]?.aborted).toBe(true)
  })
})
