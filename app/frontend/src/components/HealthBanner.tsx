/**
 * Prominent destructive banner for the two things the operator must notice at a
 * glance: the camera froze (`capture_health == "stalled"`) or the poll loop lost
 * the backend. Renders nothing when both are clear. Never reads `status` itself,
 * so it cannot crash on the null/skeleton state.
 */
export function HealthBanner({
  stalled,
  pollError,
}: {
  stalled: boolean
  pollError: boolean
}) {
  if (!stalled && !pollError) return null

  // Stalled camera is the more specific, more alarming condition — lead with it.
  const message = stalled
    ? "Camera stalled — the live feed is frozen. Check the camera before recording."
    : "Lost connection to the backend — retrying…"

  return (
    <div
      role="alert"
      className="rounded-lg bg-destructive/10 px-4 py-3 text-sm font-medium text-destructive"
    >
      {message}
    </div>
  )
}
