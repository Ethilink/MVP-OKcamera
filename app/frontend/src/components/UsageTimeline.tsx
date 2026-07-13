import type { UsageWindow } from "@/api/types"
import { formatClock } from "@/lib/format"
import { axisTicks, usageSegments } from "@/lib/timeline"

/**
 * One instrument's Usage as bars on the shared recording time axis. On-table is
 * the quiet track background; each off-table window is a solid destructive bar;
 * an open window (never returned) runs to the end with a distinct hatched
 * treatment and a "never returned" label. Empty usage reads "never picked up".
 * This is the demo money-shot — it must read at presentation distance.
 */
export function UsageTimeline({
  duration_s,
  usage,
}: {
  duration_s: number
  usage: UsageWindow[]
}) {
  const segments = usageSegments(duration_s, usage)
  const ticks = axisTicks(duration_s)

  return (
    <div className="flex flex-col gap-1">
      <div className="relative h-7 w-full overflow-hidden rounded-md bg-muted">
        {usage.length === 0 && (
          <span className="absolute inset-0 flex items-center justify-center text-xs text-muted-foreground">
            never picked up
          </span>
        )}
        {segments.map((seg, i) => {
          const w = usage[i]
          const label =
            w.on_s === null
              ? `off from ${formatClock(w.off_s)}, never returned`
              : `off ${formatClock(w.off_s)}–${formatClock(w.on_s)}`
          return (
            <div
              key={`${w.off_s}-${w.on_s ?? "open"}`}
              role="img"
              aria-label={label}
              title={label}
              style={{ left: `${seg.leftPct}%`, width: `${seg.widthPct}%` }}
              className={
                seg.open
                  ? "absolute inset-y-0 border-l-2 border-destructive bg-[repeating-linear-gradient(45deg,color-mix(in_oklch,var(--destructive)_35%,transparent)_0,color-mix(in_oklch,var(--destructive)_35%,transparent)_6px,transparent_6px,transparent_12px)]"
                  : "absolute inset-y-0 bg-destructive/70"
              }
            />
          )
        })}
      </div>
      <div className="relative h-4 w-full text-[0.65rem] text-muted-foreground">
        {ticks.map((t) => (
          <span
            key={t.label}
            style={{ left: `${t.pct}%` }}
            className="absolute -translate-x-1/2 tabular-nums"
          >
            {t.label}
          </span>
        ))}
      </div>
    </div>
  )
}
