// Shared display formatters. Kept pure + unit-tested so the timeline axis (T07)
// and the live timers (T06) agree on how seconds read.

/** mm:ss from a non-negative seconds value (floored). 74.3 -> "1:14", 0 -> "0:00". */
export function formatClock(totalSeconds: number): string {
  const s = Math.max(0, Math.floor(totalSeconds))
  const m = Math.floor(s / 60)
  const sec = s % 60
  return `${m}:${String(sec).padStart(2, "0")}`
}

/** whole-second "off for" label. 13.2 -> "13s". Never negative. */
export function formatSeconds(seconds: number): string {
  return `${Math.max(0, Math.floor(seconds))}s`
}
