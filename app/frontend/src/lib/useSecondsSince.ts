import { useEffect, useRef, useState } from "react"

/**
 * Seconds elapsed since `anchor` (by identity) last changed — 0 right after a
 * change. The live screen anchors on the freshest `/status` object so every
 * server counter in that payload (elapsed_s, each off_since_s) can be shown as
 * `serverValue + delta`, smoothly filling the ½-second gaps between 2 Hz polls;
 * the next poll hands back a new object → delta re-anchors to 0 and the backend
 * value is authoritative again (T06 "Ticking ownership").
 *
 * `running` gates the repaint loop: no timer churns while paused (setup/finished).
 */
export function useSecondsSince(anchor: unknown, running = true): number {
  const anchorRef = useRef(anchor)
  const startRef = useRef(performance.now())
  // Re-anchor during render when the identity changes (the "adjust state on prop
  // change" pattern) so the returned delta is already 0 on the re-anchoring paint.
  if (anchorRef.current !== anchor) {
    anchorRef.current = anchor
    startRef.current = performance.now()
  }

  const [, forceRepaint] = useState(0)
  useEffect(() => {
    if (!running) return
    const id = setInterval(() => forceRepaint((n) => n + 1), 200)
    return () => clearInterval(id)
  }, [running])

  return Math.max(0, (performance.now() - startRef.current) / 1000)
}
