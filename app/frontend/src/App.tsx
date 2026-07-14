import { useEffect, useState } from "react"
import { useStatus } from "@/api/useStatus"
import { useLastSeenCrops } from "@/api/useLastSeenCrops"
import { LiveScreen } from "@/screens/LiveScreen"
import { ReportScreen } from "@/screens/ReportScreen"

/**
 * Phase router (T06 §Routing — App owns it). Routes on `status.phase` plus one
 * local flag `newRecordingRequested`:
 *
 * - recording                       → LiveScreen (recording layout)
 * - setup                           → LiveScreen (setup layout)
 * - finished ∧ ¬flag                → ReportScreen
 * - finished ∧ flag                 → LiveScreen (setup layout, run-2 re-confirm)
 *
 * ReportScreen's "New recording" only SETS the flag (it never calls the API —
 * T07); the gated Start in the setup layout is the real POST /recording/start.
 * The flag clears whenever a poll shows `recording`, and "Back to report" clears
 * it too, so an accidental "New recording" is recoverable while the backend
 * still holds the report (D7). `pollMs` is a test seam; production uses 2 Hz.
 */
function App({ pollMs = 500 }: { pollMs?: number } = {}) {
  const { status, error } = useStatus(pollMs)
  // Per-instrument crops accumulate across setup → recording → report, so the
  // live list and the report can show each instrument's cut-out (D-crops).
  const crops = useLastSeenCrops(status)
  const [newRecordingRequested, setNewRecordingRequested] = useState(false)
  const phase = status?.phase

  useEffect(() => {
    if (phase === "recording") setNewRecordingRequested(false)
  }, [phase])

  if (phase === "finished" && !newRecordingRequested) {
    return (
      <ReportScreen
        crops={crops}
        onNewRecording={() => setNewRecordingRequested(true)}
      />
    )
  }

  return (
    <LiveScreen
      status={status}
      error={error}
      crops={crops}
      showBackToReport={phase === "finished" && newRecordingRequested}
      onBackToReport={() => setNewRecordingRequested(false)}
    />
  )
}

export default App
