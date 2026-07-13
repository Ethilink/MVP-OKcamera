import { useEffect, useState } from "react"

/**
 * Seconds elapsed since `anchor` (by identity) last changed — 0 right after a
 * change. The live screen anchors on the freshest `/status` object so every
 * server counter in that payload (elapsed_s, each off_since_s) can be shown as
 * `serverValue + delta`, smoothly filling the ½-second gaps between 2 Hz polls;
 * the next poll hands back a new object → delta re-anchors to 0 and the backend
 * value is authoritative again (T06 "Ticking ownership").
 *
 * `running` gates the repaint loop: no timer churns while paused (setup/finished).
 *
 * Re-anchoring uses the React-blessed "adjust state during render" pattern (a
 * setState guarded by an identity check, which re-renders immediately and
 * discards the in-progress paint) rather than mutating a ref in render — so an
 * abandoned concurrent render can never leave a committed anchor pointing at the
 * wrong poll.
 */
export function useSecondsSince(anchor: unknown, running = true): number {
  const [start, setStart] = useState(() => performance.now())
  const [seenAnchor, setSeenAnchor] = useState(anchor)
  if (seenAnchor !== anchor) {
    setSeenAnchor(anchor)
    setStart(performance.now())
  }

  const [, forceRepaint] = useState(0)
  useEffect(() => {
    if (!running) return
    const id = setInterval(() => forceRepaint((n) => n + 1), 200)
    return () => clearInterval(id)
  }, [running])

  return Math.max(0, (performance.now() - start) / 1000)
}
