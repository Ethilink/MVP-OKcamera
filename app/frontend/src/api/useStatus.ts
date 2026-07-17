import { useCallback, useEffect, useRef, useState } from "react"
import { api, type ApiError } from "./client"
import type { Status } from "./types"

export interface UseStatusResult {
  status: Status | null // last good payload, kept across errors
  error: ApiError | Error | null // non-null after a failed poll, cleared on success
  /** Waits for any older poll, then fetches a response started after the caller's
   *  mutation. Used to close the stale-ready window after reset operations. */
  refresh: () => Promise<Status | null>
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
  const refreshRef = useRef<() => Promise<Status | null>>(async () => null)
  const refresh = useCallback(() => refreshRef.current(), [])

  useEffect(() => {
    let cancelled = false
    let inFlight: Promise<Status | null> | null = null
    let activeController: AbortController | null = null
    const recoveryWaiters = new Set<(status: Status | null) => void>()

    function resolveRecoveryWaiters(next: Status | null) {
      for (const resolve of recoveryWaiters) resolve(next)
      recoveryWaiters.clear()
    }

    async function requestStatus(): Promise<Status | null> {
      const controller = new AbortController()
      activeController = controller
      const timeoutId = window.setTimeout(
        () => controller.abort(),
        requestTimeoutMs,
      )
      try {
        const next = await api.status(controller.signal)
        if (cancelled) return null
        setStatus(next)
        setError(null)
        resolveRecoveryWaiters(next)
        return next
      } catch (err) {
        if (cancelled) return null
        setError(err as ApiError | Error) // keep last good status untouched
        return null
      } finally {
        window.clearTimeout(timeoutId)
        if (activeController === controller) activeController = null
      }
    }

    function poll(): Promise<Status | null> {
      if (inFlight) return inFlight // regular ticks never overlap
      const request = requestStatus()
      inFlight = request
      void request.finally(() => {
        if (inFlight === request) inFlight = null
      })
      return request
    }

    async function refreshAfterCurrentPoll(): Promise<Status | null> {
      // A poll that started before prepare/PATCH may still carry the old Ready
      // payload. Let it finish, then start one request that is definitely newer.
      const olderPoll = inFlight
      if (olderPoll) await olderPoll
      if (cancelled) return null
      const freshStatus = await poll()
      if (freshStatus || cancelled) return freshStatus
      // The mutation-fresh request failed. Keep the caller's safety hold armed
      // until ordinary polling recovers, then resolve it with that first success.
      return new Promise((resolve) => recoveryWaiters.add(resolve))
    }

    refreshRef.current = refreshAfterCurrentPoll

    void poll() // fire immediately, don't wait a full interval for first paint
    const id = setInterval(() => void poll(), intervalMs)
    return () => {
      cancelled = true
      clearInterval(id)
      activeController?.abort()
      resolveRecoveryWaiters(null)
      if (refreshRef.current === refreshAfterCurrentPoll) {
        refreshRef.current = async () => null
      }
    }
  }, [intervalMs, requestTimeoutMs])

  return { status, error, refresh }
}
