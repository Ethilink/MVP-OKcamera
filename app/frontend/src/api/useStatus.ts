import { useEffect, useState } from "react"
import { api, type ApiError } from "./client"
import type { Status } from "./types"

export interface UseStatusResult {
  status: Status | null // last good payload, kept across errors
  error: ApiError | Error | null // non-null after a failed poll, cleared on success
}

/**
 * Polls `GET /status` every `intervalMs` (default 500 → ~2 Hz, contract D4).
 * - keeps the last good `status` even while `error` is set (AC4)
 * - never stacks overlapping requests: if a poll is still in flight when the
 *   next tick fires, the tick is skipped (AC3)
 * - aborts a request that exceeds `requestTimeoutMs`, allowing later polls to
 *   recover, and aborts pending work when the hook unmounts
 */
export function useStatus(
  intervalMs = 500,
  requestTimeoutMs = 5_000,
): UseStatusResult {
  const [status, setStatus] = useState<Status | null>(null)
  const [error, setError] = useState<ApiError | Error | null>(null)

  useEffect(() => {
    let cancelled = false
    let inFlight = false
    let activeController: AbortController | null = null

    async function poll() {
      if (inFlight) return // a previous (slow) request is still running
      inFlight = true
      const controller = new AbortController()
      activeController = controller
      const timeoutId = window.setTimeout(
        () => controller.abort(),
        requestTimeoutMs,
      )
      try {
        const next = await api.status(controller.signal)
        if (cancelled) return
        setStatus(next)
        setError(null)
      } catch (err) {
        if (cancelled) return
        setError(err as ApiError | Error) // keep last good status untouched
      } finally {
        window.clearTimeout(timeoutId)
        if (activeController === controller) activeController = null
        inFlight = false
      }
    }

    poll() // fire immediately, don't wait a full interval for first paint
    const id = setInterval(poll, intervalMs)
    return () => {
      cancelled = true
      clearInterval(id)
      activeController?.abort()
    }
  }, [intervalMs, requestTimeoutMs])

  return { status, error }
}
