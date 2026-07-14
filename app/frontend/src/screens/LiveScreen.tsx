import { useState } from "react"
import { LayoutGroup } from "motion/react"
import { ApiError, api } from "@/api/client"
import type { Status } from "@/api/types"
import type { CropMap } from "@/api/useLastSeenCrops"
import { DetectionConstellation } from "@/components/DetectionConstellation"
import { HaloBrand } from "@/components/HaloMark"
import { HealthBanner } from "@/components/HealthBanner"
import { InstrumentPanel } from "@/components/InstrumentPanel"
import { StopButton, TrackButton } from "@/components/TrackButton"
import { VideoFeed } from "@/components/VideoFeed"
import { formatClock } from "@/lib/format"
import { useSecondsSince } from "@/lib/useSecondsSince"

/** Shared-layout id: the setup Track pill and the recording Stop pill are the
 *  same element to Motion, so Start slides Track down and recolours it to Stop. */
const PRIMARY_CTA = "primary-cta"

/** Setup and recording MUST share this exact stage so the feed never changes
 *  size or vertical position across Start (the one thing the operator's eye is
 *  locked on). `items-center` centres the whole two-column block in the
 *  available height; `items-stretch` on the grid pins the right column to the
 *  feed's exact height (the feed's 16:9 box is the tallest cell, so it sets the
 *  row height and the right column matches it). */
const STAGE = "flex min-h-0 flex-1 items-center"
const STAGE_GRID =
  "grid w-full items-stretch gap-6 lg:grid-cols-[minmax(0,2.6fr)_minmax(20rem,0.72fr)]"

interface LiveScreenProps {
  status: Status | null
  error: Error | null
  /** Per-instrument crops accumulated across the run (App owns the map). */
  crops?: CropMap
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
  crops,
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
    <main className="mx-auto flex min-h-svh w-full max-w-[112rem] flex-col gap-6 p-6">
      <header className="flex items-center justify-between">
        <HaloBrand />
        <div className="flex items-center gap-3">
          {showBackToReport && (
            <button
              type="button"
              onClick={onBackToReport}
              aria-label="Back to report"
              title="Back to report"
              className="grid size-9 place-items-center rounded-full text-muted-foreground ring-1 ring-border transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <ArrowLeftIcon className="size-[1.15rem]" />
            </button>
          )}
        </div>
      </header>

      {banner}
      {actionError && (
        <p role="alert" className="text-sm text-destructive">
          {actionError}
        </p>
      )}

      <LayoutGroup>
        {isRecording && status.recording ? renderRecording() : renderSetup()}
      </LayoutGroup>
    </main>
  )

  function renderRecording() {
    const rec = status!.recording!
    // Same STAGE/STAGE_GRID as setup so the feed keeps its exact size and
    // position across Start. The right column is absolutely filled inside a
    // zero-intrinsic cell, so it takes the feed's height precisely (never taller,
    // never a growing column): the list fills the space and scrolls internally
    // when the tray is large, and the timer+Stop row sits pinned at the bottom —
    // exactly where the setup Track pill is, so it slides down and reddens into
    // Stop (shared layoutId).
    return (
      <div className={STAGE}>
        <div className={STAGE_GRID}>
          <VideoFeed />

          <div className="relative min-h-0">
            <div className="absolute inset-0 flex flex-col gap-4">
              <InstrumentPanel recording={rec} crops={crops} />
              <div className="flex shrink-0 items-center justify-center gap-5">
                <span className="text-2xl font-semibold tabular-nums text-muted-foreground">
                  {formatClock(rec.elapsed_s + secondsSincePoll)}
                </span>
                <StopButton
                  layoutId={PRIMARY_CTA}
                  pending={pending}
                  onStop={() => run(api.stopRecording)}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    )
  }

  function renderSetup() {
    const setup = status?.setup ?? null
    const connecting = !status
    const stalled = status?.capture_health === "stalled"
    const healthOk = status?.capture_health === "ok"
    const gateablePhase =
      status?.phase === "setup" || status?.phase === "finished"
    const detectedCount = setup?.detected_count ?? 0
    const stableForS = setup?.stable_for_s ?? 0
    const enabled =
      gateablePhase && healthOk && detectedCount >= 1 && stableForS >= 2

    let reason = "Connecting to the camera…"
    if (status) {
      if (!healthOk) reason = "Camera stalled"
      else if (detectedCount < 1) reason = "Waiting for instruments"
      else if (stableForS < 2) reason = "Hold steady…"
    }

    return (
      <div className={STAGE}>
        <div className={STAGE_GRID}>
          {/* Left — the camera the operator is framing the tray in. */}
          <VideoFeed />

          {/* Right — the detection hub and the one action, centred to the feed. */}
          <div className="flex h-full flex-col items-center justify-center gap-7">
            <DetectionConstellation
              detectedCount={detectedCount}
              detections={setup?.detections}
              ready={enabled}
              stalled={stalled}
              connecting={connecting}
            />
            <TrackButton
              layoutId={PRIMARY_CTA}
              enabled={enabled}
              reason={reason}
              pending={pending}
              onTrack={() => run(api.startRecording)}
            />
          </div>
        </div>
      </div>
    )
  }
}

function ArrowLeftIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className={className}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M19 12H5" />
      <path d="m12 19-7-7 7-7" />
    </svg>
  )
}
