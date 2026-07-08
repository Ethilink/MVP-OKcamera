import { useEffect, useState } from "react"
import { ApiError, api } from "@/api/client"
import type { Report } from "@/api/types"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { CompletenessBadge } from "@/components/CompletenessBadge"
import { ReportSummary } from "@/components/ReportSummary"
import { UsageTimeline } from "@/components/UsageTimeline"

/**
 * The payoff screen (DESIGN §3): after Stop, per instrument a Usage timeline +
 * Completeness badge. Fetches `api.report()` on mount (the screen can mount
 * early, e.g. before the finished poll) and degrades to a non-crashing error
 * state on 409/failure. "New recording" only calls `onNewRecording` — it does
 * NOT start a recording; the gated Start in T06's setup view owns run 2 (D15).
 */
export function ReportScreen({
  onNewRecording,
}: {
  onNewRecording: () => void
}) {
  const [report, setReport] = useState<Report | null>(null)
  const [errorText, setErrorText] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
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
  }, [])

  const newRecordingButton = (
    <Button onClick={onNewRecording}>New recording</Button>
  )

  return (
    <main className="mx-auto flex min-h-svh max-w-5xl flex-col gap-6 p-6">
      <header className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">ORC · Report</h1>
        {newRecordingButton}
      </header>

      {errorText ? (
        <Card>
          <CardContent className="py-8 text-center text-muted-foreground">
            {errorText}
          </CardContent>
        </Card>
      ) : !report ? (
        <Card>
          <CardContent className="py-8 text-center text-muted-foreground">
            Loading report…
          </CardContent>
        </Card>
      ) : (
        <>
          <ReportSummary report={report} />
          <Card>
            <CardContent className="flex flex-col divide-y divide-border">
              {report.instruments.map((inst) => (
                <div
                  key={inst.tracker_id}
                  className={`grid grid-cols-[8rem_1fr_auto] items-center gap-4 py-4 ${
                    inst.completeness === "missing"
                      ? "rounded-md bg-destructive/5"
                      : ""
                  }`}
                >
                  <span className="font-medium">{inst.label}</span>
                  <UsageTimeline
                    duration_s={report.duration_s}
                    usage={inst.usage}
                  />
                  <CompletenessBadge completeness={inst.completeness} />
                </div>
              ))}
            </CardContent>
          </Card>
        </>
      )}
    </main>
  )
}
