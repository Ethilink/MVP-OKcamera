import { useEffect, useRef, useState } from "react"
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
 */
export function useStatus(intervalMs = 500): UseStatusResult {
  const [status, setStatus] = useState<Status | null>(null)
  const [error, setError] = useState<ApiError | Error | null>(null)
  const inFlight = useRef(false)

  useEffect(() => {
    let cancelled = false

    async function poll() {
      if (inFlight.current) return // a previous (slow) request is still running
      inFlight.current = true
      try {
        const next = await api.status()
        if (cancelled) return
        setStatus(next)
        setError(null)
      } catch (err) {
        if (cancelled) return
        setError(err as ApiError | Error) // keep last good status untouched
      } finally {
        inFlight.current = false
      }
    }

    poll() // fire immediately, don't wait a full interval for first paint
    const id = setInterval(poll, intervalMs)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [intervalMs])

  return { status, error }
}
