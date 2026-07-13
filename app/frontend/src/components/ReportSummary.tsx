import type { Report } from "@/api/types"
import { formatClock } from "@/lib/format"

/** Headline numbers for the run: duration, instrument count, missing count, model. */
export function ReportSummary({ report }: { report: Report }) {
  const missing = report.instruments.filter(
    (i) => i.completeness === "missing"
  ).length

  return (
    <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1">
      <span className="text-3xl font-semibold tabular-nums">
        {formatClock(report.duration_s)}
      </span>
      <span className="text-sm text-muted-foreground">
        {report.instruments.length} instruments
      </span>
      <span className="text-sm text-muted-foreground">{missing} missing</span>
      <span className="ml-auto text-xs text-muted-foreground">
        {report.model_version}
      </span>
    </div>
  )
}
