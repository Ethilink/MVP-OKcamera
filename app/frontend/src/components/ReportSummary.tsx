import type { Report } from "@/api/types"
import { formatClock } from "@/lib/format"
import { cn } from "@/lib/utils"

/**
 * The run's headline numbers as three small stat cards — duration, instrument
 * count, and missing count — centred under the "Report" title. They read as
 * quiet cards (not buttons); the missing card tints red when anything is
 * missing.
 */
export function ReportSummary({ report }: { report: Report }) {
  const missing = report.instruments.filter(
    (i) => i.completeness === "missing"
  ).length

  return (
    <div className="flex flex-wrap justify-center gap-3">
      <Stat value={formatClock(report.duration_s)} label="duration" />
      <Stat value={String(report.instruments.length)} label="instruments" />
      <Stat
        value={String(missing)}
        label="missing"
        tone={missing > 0 ? "alert" : "default"}
      />
    </div>
  )
}

function Stat({
  value,
  label,
  tone = "default",
}: {
  value: string
  label: string
  tone?: "default" | "alert"
}) {
  return (
    <div
      className={cn(
        "flex min-w-[7.5rem] flex-col items-center gap-0.5 rounded-xl bg-card px-6 py-3 ring-1 ring-foreground/10",
        tone === "alert" && "bg-destructive/5 ring-destructive/20"
      )}
    >
      <span
        className={cn(
          "text-2xl font-semibold tabular-nums",
          tone === "alert" && "text-destructive"
        )}
      >
        {value}
      </span>
      <span className="text-xs font-medium text-muted-foreground">{label}</span>
    </div>
  )
}
