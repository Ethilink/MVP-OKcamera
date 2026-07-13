import { useState } from "react"
import { ApiError, api } from "@/api/client"
import type { Status } from "@/api/types"
import { Button } from "@/components/ui/button"
import { HealthBanner } from "@/components/HealthBanner"
import { InstrumentPanel } from "@/components/InstrumentPanel"
import { StartStopControl } from "@/components/StartStopControl"
import { VideoFeed } from "@/components/VideoFeed"
import { formatClock } from "@/lib/format"
import { useSecondsSince } from "@/lib/useSecondsSince"

interface LiveScreenProps {
  status: Status | null
  error: Error | null
  /** finished ∧ newRecordingRequested — show a way back to the still-held report. */
  showBackToReport: boolean
  onBackToReport: () => void
}

/**
 * The operator's live screen for `setup` and `recording` (and `finished` routed
 * here for run 2, which reuses the setup layout — D15). Layout is chosen from
 * `phase`; all state comes from the poll (no optimistic flips). The Start gate
 * lives here per api-contract §/status.
 */
export function LiveScreen({
  status,
  error,
  showBackToReport,
  onBackToReport,
}: LiveScreenProps) {
  const [actionError, setActionError] = useState<string | null>(null)
  const [pending, setPending] = useState(false)

  const isRecording = status?.phase === "recording"
  // Interpolate live counters only while recording; re-anchors on each poll.
  const secondsSincePoll = useSecondsSince(status, isRecording)

  async function run(action: () => Promise<unknown>) {
    setPending(true)
    setActionError(null)
    try {
      await action()
    } catch (err) {
      // 409 = wrong-phase (contract); surface non-fatally, polling continues.
      setActionError(
        err instanceof ApiError ? err.detail : "Something went wrong — try again."
      )
    } finally {
      setPending(false)
    }
  }

  const banner = (
    <HealthBanner
      stalled={status?.capture_health === "stalled"}
      pollError={error !== null}
    />
  )

  return (
    <main className="mx-auto flex min-h-svh max-w-5xl flex-col gap-4 p-6">
      <header className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">ORC</h1>
        {status && (
          <span className="text-sm text-muted-foreground">
            {status.model_version}
          </span>
        )}
      </header>

      {banner}
      {actionError && (
        <p role="alert" className="text-sm text-destructive">
          {actionError}
        </p>
      )}

      {isRecording && status.recording
        ? renderRecording()
        : renderSetup()}
    </main>
  )

  function renderRecording() {
    const rec = status!.recording!
    // Stop + timer pinned at the top; feed and the live panel sit side by side
    // below so both stay above the fold on a laptop (Stop must never be a scroll
    // away). Stacks on narrow screens.
    return (
      <div className="flex flex-1 flex-col gap-4">
        <div className="flex items-center justify-between">
          <div className="flex items-baseline gap-4">
            <span className="text-2xl font-semibold tabular-nums">
              {formatClock(rec.elapsed_s + secondsSincePoll)}
            </span>
            <span className="text-sm text-muted-foreground">
              {rec.on_table_count} on table
            </span>
          </div>
          <StartStopControl
            mode="stop"
            pending={pending}
            onStop={() => run(api.stopRecording)}
          />
        </div>
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
          <VideoFeed />
          <InstrumentPanel recording={rec} secondsSincePoll={secondsSincePoll} />
        </div>
      </div>
    )
  }

  function renderSetup() {
    const setup = status?.setup ?? null
    const healthOk = status?.capture_health === "ok"
    const gateablePhase =
      status?.phase === "setup" || status?.phase === "finished"
    const enabled =
      gateablePhase &&
      healthOk &&
      !!setup &&
      setup.detected_count >= 1 &&
      setup.stable_for_s >= 2

    let reason = "connecting…"
    if (status) {
      if (!healthOk) reason = "camera stalled"
      else if (!setup || setup.detected_count < 1)
        reason = "waiting for detections…"
      else if (setup.stable_for_s < 2)
        reason = "waiting for stable detections…"
    }

    return (
      <div className="flex flex-1 flex-col items-center gap-4">
        {/* Cap the feed width so its 16:9 height stays modest and the gate +
            Start (the whole point of setup) never fall below the fold. */}
        <div className="w-full max-w-3xl">
          <VideoFeed />
        </div>
        <p className="text-sm text-muted-foreground">
          {setup
            ? `${setup.detected_count} instruments detected · stable for ${setup.stable_for_s.toFixed(1)}s`
            : "waiting for the camera…"}
        </p>
        <StartStopControl
          mode="start"
          enabled={enabled}
          reason={reason}
          pending={pending}
          onStart={() => run(api.startRecording)}
        />
        {showBackToReport && (
          <Button variant="outline" onClick={onBackToReport}>
            Back to report
          </Button>
        )}
      </div>
    )
  }
}
