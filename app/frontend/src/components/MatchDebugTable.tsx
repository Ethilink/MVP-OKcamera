import type { ReactNode } from "react"
import type { Detection, InstrumentStatus } from "@/api/types"

/**
 * Experimental (feat/matching-tests): a plain, uncapped debug table of every
 * currently-known detection/instrument, showing the detector's raw confidence
 * and the matcher's last score (bind at Start, or a later re-id decision,
 * whichever happened most recently) — including for `Unknown`/rejected rows,
 * so a near-miss reject is visible right next to a confident accept.
 *
 * Deliberately separate from `DetectionConstellation`/`InstrumentPanel`: those
 * are the polished demo UI (animated, capped, filtered to recognised-only).
 * This is a testing aid for tuning the matcher against a real camera — no
 * animation, no cap, no filtering, and it disappears on its own in fake mode
 * (every row's numbers are `null` there, so the table renders em-dashes).
 */

type Row = {
  tracker_id: number
  label: string
  state?: string
  detector_confidence: number | null | undefined
  matcher_score: number | null | undefined
  matcher_tau: number | null | undefined
  matcher_closest_id: number | null | undefined
  matcher_accepted: boolean | null | undefined
}

export function MatchDebugTable({
  title,
  rows,
}: {
  title: string
  rows: Row[]
}) {
  if (rows.length === 0) return null
  const sorted = [...rows].sort((a, b) => a.tracker_id - b.tracker_id)

  return (
    <div className="rounded-lg border border-dashed border-amber-500/40 bg-amber-500/5 p-3 text-xs">
      <div className="mb-2 font-mono font-semibold uppercase tracking-wide text-amber-700 dark:text-amber-400">
        {title} — matching debug
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[36rem] border-collapse font-mono">
          <thead>
            <tr className="border-b border-amber-500/30 text-left text-muted-foreground">
              <Th>id</Th>
              <Th>label</Th>
              {sorted.some((r) => r.state !== undefined) && <Th>state</Th>}
              <Th>detector conf</Th>
              <Th>matcher score</Th>
              <Th>tau</Th>
              <Th>closest</Th>
              <Th>accepted</Th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((row) => (
              <tr key={row.tracker_id} className="border-b border-amber-500/10">
                <Td>{row.tracker_id}</Td>
                <Td>{row.label || "—"}</Td>
                {sorted.some((r) => r.state !== undefined) && <Td>{row.state ?? "—"}</Td>}
                <Td>{fmt(row.detector_confidence)}</Td>
                <Td highlight={nearTau(row)}>{fmt(row.matcher_score)}</Td>
                <Td>{fmt(row.matcher_tau)}</Td>
                <Td>{row.matcher_closest_id ?? "—"}</Td>
                <Td>{fmtBool(row.matcher_accepted)}</Td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

/** Flag a score within 0.05 of its own tau — the borderline cases worth a
 *  second look, whichever side of the gate they landed on. */
function nearTau(row: Row): boolean {
  if (row.matcher_score == null || row.matcher_tau == null) return false
  return Math.abs(row.matcher_score - row.matcher_tau) < 0.05
}

function fmt(value: number | null | undefined): string {
  return value == null ? "—" : value.toFixed(3)
}

function fmtBool(value: boolean | null | undefined): string {
  return value == null ? "—" : value ? "yes" : "no"
}

function Th({ children }: { children: ReactNode }) {
  return <th className="px-2 py-1 font-medium">{children}</th>
}

function Td({
  children,
  highlight,
}: {
  children: ReactNode
  highlight?: boolean
}) {
  return (
    <td className={`px-2 py-1 tabular-nums ${highlight ? "font-bold text-amber-600 dark:text-amber-400" : ""}`}>
      {children}
    </td>
  )
}

/** Setup detections -> shared debug rows. */
export function detectionRows(detections: Detection[] | undefined): Row[] {
  if (!detections) return []
  return detections.map((d) => ({
    tracker_id: d.tracker_id,
    label: d.label,
    state: d.state,
    detector_confidence: d.detector_confidence,
    matcher_score: d.matcher_score,
    matcher_tau: d.matcher_tau,
    matcher_closest_id: d.matcher_closest_id,
    matcher_accepted: d.matcher_accepted,
  }))
}

/** Recording instruments -> shared debug rows. */
export function instrumentRows(instruments: InstrumentStatus[] | undefined): Row[] {
  if (!instruments) return []
  return instruments.map((i) => ({
    tracker_id: i.tracker_id,
    label: i.label,
    detector_confidence: i.detector_confidence,
    matcher_score: i.matcher_score,
    matcher_tau: i.matcher_tau,
    matcher_closest_id: i.matcher_closest_id,
    matcher_accepted: i.matcher_accepted,
  }))
}
