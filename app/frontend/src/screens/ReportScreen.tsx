import { useEffect, useState } from "react"
import { ApiError, api } from "@/api/client"
import type { Report } from "@/api/types"
import type { CropMap } from "@/api/useLastSeenCrops"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { CompletenessBadge } from "@/components/CompletenessBadge"
import { HaloBrand } from "@/components/HaloMark"
import { NewRecordingButton } from "@/components/TrackButton"
import { ReportSummary } from "@/components/ReportSummary"
import { UsageTimeline } from "@/components/UsageTimeline"
import { instrumentIconFor } from "@/components/instruments/InstrumentIcons"

/**
 * The payoff screen (DESIGN §3): after Stop, per instrument a Usage timeline +
 * Completeness badge. Fetches `api.report()` on mount (the screen can mount
 * early, e.g. before the finished poll) and degrades to a non-crashing error
 * state on 409/failure. "New recording" only calls `onNewRecording` — it does
 * NOT start a recording; the gated Start in T06's setup view owns run 2 (D15).
 */
export function ReportScreen({
  crops,
  onNewRecording,
}: {
  crops?: CropMap
  onNewRecording: () => void
}) {
  const [report, setReport] = useState<Report | null>(null)
  const [errorText, setErrorText] = useState<string | null>(null)
  const [requestVersion, setRequestVersion] = useState(0)

  useEffect(() => {
    let cancelled = false
    setErrorText(null)
    api
      .report()
      .then((r) => {
        if (!cancelled) setReport(r)
      })
      .catch((err) => {
        if (cancelled) return
        // 409 = not in the finished phase yet; anything else = a real failure.
        setErrorText(
          err instanceof ApiError && err.status === 409
            ? "No finished recording yet."
            : "Could not load the report."
        )
      })
    return () => {
      cancelled = true
    }
  }, [requestVersion])

  return (
    <main className="mx-auto flex h-svh w-full max-w-[112rem] flex-col gap-5 p-6">
      {/* The halo lockup sits top-left — the exact place it holds on Setup and
          Recording, so the brand never moves between screens. */}
      <header className="flex shrink-0 items-center">
        <HaloBrand />
      </header>

      {/* The page title, centred and large, with the headline numbers beneath. */}
      <div className="mx-auto flex w-full max-w-5xl shrink-0 flex-col items-center gap-3">
        <h1 className="text-3xl font-semibold tracking-tight">Report</h1>
        {report && <ReportSummary report={report} />}
      </div>

      {/* The instrument list takes the remaining height and scrolls inside its
          card, so the summary above and the action below always stay on screen —
          the whole report fits one viewport however many instruments there are. */}
      <div className="mx-auto flex w-full min-h-0 max-w-5xl flex-1 flex-col">
        {errorText ? (
          <Card>
            <CardContent className="flex flex-col items-center gap-4 py-8 text-center text-muted-foreground">
              <p>{errorText}</p>
              <Button
                type="button"
                variant="outline"
                onClick={() => setRequestVersion((version) => version + 1)}
              >
                Retry
              </Button>
            </CardContent>
          </Card>
        ) : !report ? (
          <Card>
            <CardContent className="py-8 text-center text-muted-foreground">
              Loading report…
            </CardContent>
          </Card>
        ) : (
          <Card className="flex min-h-0 flex-1 flex-col">
            <CardContent className="flex min-h-0 flex-1 flex-col divide-y divide-border overflow-y-auto">
              {report.instruments.map((inst) => (
                <div
                  key={inst.tracker_id}
                  className={`grid grid-cols-[2.5rem_7rem_1fr_auto] items-center gap-4 py-2.5 ${
                    inst.completeness === "missing"
                      ? "rounded-md bg-destructive/5"
                      : ""
                  }`}
                >
                  <ReportCrop
                    trackerId={inst.tracker_id}
                    label={inst.label}
                    crop={crops?.[inst.tracker_id] ?? null}
                  />
                  <span className="truncate font-medium">{inst.label}</span>
                  <UsageTimeline
                    duration_s={report.duration_s}
                    usage={inst.usage}
                  />
                  <CompletenessBadge completeness={inst.completeness} />
                </div>
              ))}
            </CardContent>
          </Card>
        )}
      </div>

      {/* The one action — a liquid-glass pill like Track/Stop — below the card. */}
      <div className="flex shrink-0 justify-center">
        <NewRecordingButton onClick={onNewRecording} />
      </div>
    </main>
  )
}

/** An instrument's cut-out on the report — the crop kept from the run, or a
 *  representative icon when none was captured (e.g. crops disabled in a test). */
function ReportCrop({
  trackerId,
  label,
  crop,
}: {
  trackerId: number
  label: string
  crop: string | null
}) {
  const Icon = instrumentIconFor(trackerId)
  return (
    <div className="grid size-10 place-items-center overflow-hidden rounded-lg bg-card text-foreground/70 ring-1 ring-border">
      {crop ? (
        <img
          src={crop}
          alt={label}
          className="size-full object-cover"
          draggable={false}
        />
      ) : (
        <Icon className="size-5" />
      )}
    </div>
  )
}
