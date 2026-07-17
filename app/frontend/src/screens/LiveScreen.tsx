import { useState } from "react"
import { LayoutGroup } from "motion/react"
import { ApiError, api } from "@/api/client"
import type { BlockingReason, Status } from "@/api/types"
import type { CropMap } from "@/api/useLastSeenCrops"
import { AdvancedConfidence } from "@/components/AdvancedConfidence"
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
  /** A confidence change reset enrolment but the pre-reset `/status` is still held:
   *  show "Recognising" and keep Track disabled until a fresh poll (T11/F3/D2). */
  awaitingReset?: boolean
  /** Arm the awaiting-reset hold after a local reset (a confidence change). */
  onReset?: () => void | Promise<unknown>
}

/**
 * The operator's live screen for `setup` and `recording` (and `finished` routed
 * here for run 2, which reuses the setup layout — D15). Layout is chosen from
 * `phase`; all state comes from the poll (no optimistic flips).
 *
 * The Start gate is the BACKEND's verdict (T11/D3/F1): the button renders
 * `setup.ready` directly — the frontend never recomputes eligibility. A disabled
 * Track states the specific blocking reason; a stalled camera's banner wins.
 */
export function LiveScreen({
  status,
  error,
  crops,
  showBackToReport,
  onBackToReport,
  awaitingReset = false,
  onReset,
}: LiveScreenProps) {
  const [actionError, setActionError] = useState<string | null>(null)
  // WHICH action is in flight, so only the relevant button shows its pending
  // label ("Starting…" / "Stopping…") while all of them disable. A confidence
  // PATCH is tracked separately (`settingsPending`) so it disables Track without
  // ever reading as "Starting…" (F4).
  const [pendingAction, setPendingAction] = useState<"start" | "stop" | null>(
    null,
  )
  const [settingsPending, setSettingsPending] = useState(false)
  const busy = pendingAction !== null

  const isRecording = status?.phase === "recording"
  // Interpolate live counters only while recording; re-anchors on each poll.
  const secondsSincePoll = useSecondsSince(status, isRecording)

  async function run(name: "start" | "stop", action: () => Promise<unknown>) {
    setPendingAction(name)
    setActionError(null)
    try {
      await action()
    } catch (err) {
      // 409 = wrong-phase (contract); surface non-fatally, polling continues.
      setActionError(
        err instanceof ApiError ? err.detail : "Something went wrong — try again."
      )
    } finally {
      setPendingAction(null)
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
                  pending={pendingAction === "stop"}
                  onStop={() => run("stop", api.stopRecording)}
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
    const ready = setup?.ready ?? false
    // A reset (confidence change) is under way or its fresh `/status` hasn't
    // landed yet — the tray is being re-recognised, so show "Recognising" and
    // hold Track regardless of the still-stale `ready` (T11/F3/D2).
    const resetting = settingsPending || awaitingReset
    // Track follows the server verdict directly (F1). It is additionally held
    // while a start is in flight or a confidence PATCH is pending, and across the
    // post-reset stale-status gap (F4).
    const enabled = ready && !busy && !settingsPending && !awaitingReset
    const reason = resetting
      ? "Recognising instruments…"
      : blockingReasonCopy(status, connecting)

    return (
      <div className={STAGE}>
        <div className={STAGE_GRID}>
          {/* Left — the camera the operator is framing the tray in. */}
          <VideoFeed />

          {/* Right — the detection hub and the one action, centred to the feed.
              The currently-detected instruments ring the count pill (the dynamic
              hub-and-spoke from T06). Track is held disabled while a reset is
              settling; the ring keeps showing the last detections. */}
          <div className="flex h-full flex-col items-center justify-center gap-7">
            <DetectionConstellation
              detectedCount={setup?.detected_count ?? 0}
              detections={setup?.detections}
              ready={ready && !resetting}
              stalled={stalled}
              connecting={connecting}
            />
            <div className="flex flex-col items-center gap-3">
              <TrackButton
                layoutId={PRIMARY_CTA}
                enabled={enabled}
                reason={reason}
                pending={pendingAction === "start"}
                onTrack={() => run("start", api.startRecording)}
              />
              {status?.detector_control && (
                <AdvancedConfidence
                  control={status.detector_control}
                  disabled={busy}
                  onPendingChange={setSettingsPending}
                  onReset={onReset}
                />
              )}
            </div>
          </div>
        </div>
      </div>
    )
  }
}

/** Map the backend's `blocking_reason` (+ capture health) to the operator copy
 *  under Track (F1). Capture health is folded into `ready` on the backend but NOT
 *  into `blocking_reason`, so a stalled camera is handled here explicitly. */
function blockingReasonCopy(status: Status | null, connecting: boolean): string {
  if (connecting || !status) return "Connecting to the camera…"
  if (status.capture_health === "stalled") return "Camera stalled"
  const setup = status.setup
  if (!setup || setup.ready) return ""
  switch (setup.blocking_reason as BlockingReason | null) {
    case "recognising":
      return "Recognising instruments…"
    case "missing_instruments":
      return `Recognised ${setup.recognised_count} of ${setup.expected_count} instruments`
    case "unknown_objects":
      return `Remove ${setup.unknown_count} unknown object${setup.unknown_count === 1 ? "" : "s"}`
    case "hold_steady":
      return "Hold the tray steady…"
    default:
      // Not ready, but the reason isn't one we render copy for (an unexpected or
      // future backend value). Track stays disabled (fail-safe) with a generic
      // line rather than a blank caption.
      return "Setup isn’t ready yet…"
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
