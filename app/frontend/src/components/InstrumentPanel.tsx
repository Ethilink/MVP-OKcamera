import { motion, useReducedMotion } from "motion/react"
import type { RecordingStatus } from "@/api/types"
import type { CropMap } from "@/api/useLastSeenCrops"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { instrumentIconFor } from "@/components/instruments/InstrumentIcons"
import { cn } from "@/lib/utils"

const ROW_EASE = [0.22, 1, 0.36, 1] as const

/**
 * The live per-instrument list during `recording`: one row per instrument
 * (sorted by tracker_id) — its cut-out crop, its label, and an ON/OFF TABLE
 * state. The crop is live while the instrument is on the table and falls back
 * to its last-seen crop (dimmed) when it is off, so a missing instrument never
 * blanks out. Rows are a fixed height and never wrap, so a state change
 * (ON↔OFF) swaps inline without nudging the layout.
 *
 * The card renders immediately and the rows slide in staggered — the empty card
 * appears first, then the instruments arrive into it. It runs once on mount
 * (stable keys), so live polls never re-trigger it.
 */
export function InstrumentPanel({
  recording,
  crops,
}: {
  recording: RecordingStatus
  crops?: CropMap
}) {
  const reduce = useReducedMotion() ?? false
  const rows = [...recording.instruments].sort(
    (a, b) => a.tracker_id - b.tracker_id
  )

  return (
    <Card className="min-h-0 flex-1">
      <CardHeader>
        <CardTitle>Instruments</CardTitle>
      </CardHeader>
      {/* The list scrolls inside the card when the tray is large, so the card
          and the Stop below it always stay one size (never a growing column). */}
      <CardContent className="min-h-0 flex-1 overflow-y-auto">
        <ul className="flex flex-col divide-y divide-border">
          {rows.map((inst, i) => {
            const crop = crops?.[inst.tracker_id] ?? inst.thumbnail ?? null
            return (
              <motion.li
                key={inst.tracker_id}
                className="grid h-14 grid-cols-[2.5rem_1fr_auto] items-center gap-3"
                initial={reduce ? false : { opacity: 0, x: 12 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.4, delay: i * 0.06, ease: ROW_EASE }}
              >
                <InstrumentCrop
                  trackerId={inst.tracker_id}
                  crop={crop}
                  label={inst.label}
                  onTable={inst.on_table}
                />
                <span className="flex min-w-0 items-center gap-2.5">
                  <InstrumentSwatch colour={inst.colour} />
                  <span className="truncate font-medium">{inst.label}</span>
                </span>
                {inst.on_table ? (
                  <Badge
                    variant="outline"
                    className="w-[5.5rem] justify-center border-transparent bg-emerald-600/10 text-emerald-700 dark:text-emerald-400"
                  >
                    ON TABLE
                  </Badge>
                ) : (
                  <Badge variant="destructive" className="w-[5.5rem] justify-center">
                    OFF TABLE
                  </Badge>
                )}
              </motion.li>
            )
          })}
        </ul>
      </CardContent>
    </Card>
  )
}

/**
 * The instrument's mask colour — the same hex the overlay draws its mask with
 * (the backend derives both from the id and the frozen roster), so a row can be
 * paired with a shape on the video at a glance and the two can never drift.
 *
 * It marks identity, not state: it stays lit while the instrument is off the
 * table, because it is still that instrument. Nothing is communicated by colour
 * alone — the label names the row and the badge carries ON/OFF TABLE — and a
 * hue is meaningless to a screen reader, so the swatch stays out of the a11y
 * tree rather than announcing a duplicate name.
 */
function InstrumentSwatch({ colour }: { colour: string }) {
  return (
    <span
      data-testid="instrument-swatch"
      aria-hidden="true"
      style={{ backgroundColor: colour }}
      className="size-2.5 shrink-0 rounded-full ring-1 ring-border"
    />
  )
}

/** The instrument's cut-out: a real crop when we have one (dimmed while off the
 *  table), else a representative icon. */
function InstrumentCrop({
  trackerId,
  crop,
  label,
  onTable,
}: {
  trackerId: number
  crop: string | null
  label: string
  onTable: boolean
}) {
  const Icon = instrumentIconFor(trackerId)
  return (
    <div
      className={cn(
        "grid size-10 place-items-center overflow-hidden rounded-lg bg-card text-foreground/70 ring-1 ring-border transition-opacity duration-300",
        !onTable && "opacity-40"
      )}
    >
      {crop ? (
        <img
          src={crop}
          alt={label}
          className={cn("size-full object-cover", !onTable && "grayscale")}
          draggable={false}
        />
      ) : (
        <Icon className="size-5" />
      )}
    </div>
  )
}
