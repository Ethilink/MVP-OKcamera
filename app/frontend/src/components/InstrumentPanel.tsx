import type { RecordingStatus } from "@/api/types"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { formatSeconds } from "@/lib/format"

/**
 * The live per-instrument list during `recording`: one row per instrument
 * (sorted by tracker_id) — label, ON TABLE / OFF TABLE badge, off_since ticking
 * while off, and pickup count. `secondsSincePoll` interpolates the off_since
 * counter between polls (T06 "Ticking ownership"); the polled value re-anchors it.
 */
export function InstrumentPanel({
  recording,
  secondsSincePoll,
}: {
  recording: RecordingStatus
  secondsSincePoll: number
}) {
  const rows = [...recording.instruments].sort(
    (a, b) => a.tracker_id - b.tracker_id
  )

  return (
    <Card>
      <CardHeader>
        <CardTitle>Instruments</CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="flex flex-col divide-y divide-border">
          {rows.map((inst) => (
            <li
              key={inst.tracker_id}
              className="flex items-center justify-between gap-4 py-2"
            >
              <span className="font-medium">{inst.label}</span>
              <div className="flex items-center gap-3">
                {inst.on_table ? (
                  <Badge
                    variant="outline"
                    className="border-transparent bg-emerald-600/10 text-emerald-700 dark:text-emerald-400"
                  >
                    ON TABLE
                  </Badge>
                ) : (
                  <Badge variant="destructive">OFF TABLE</Badge>
                )}
                {!inst.on_table && inst.off_since_s !== null && (
                  <span className="w-10 text-right tabular-nums text-muted-foreground">
                    {formatSeconds(inst.off_since_s + secondsSincePoll)}
                  </span>
                )}
                <span className="text-sm text-muted-foreground">
                  {inst.pickup_count} pickups
                </span>
              </div>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  )
}
