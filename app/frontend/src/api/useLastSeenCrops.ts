import { useEffect, useRef, useState } from "react"
import type { Status } from "./types"

/** A `tracker_id → data-URI crop` map. */
export type CropMap = Record<number, string>

/**
 * Accumulates the most-recent real crop seen for each instrument across the
 * whole run. The backend only sends a crop while an instrument is visible —
 * setup detections every poll, and (T-frontend) a live recording crop while it
 * is on the table. An off-table instrument arrives with `thumbnail: null`, so
 * without this its tile would blank out exactly when it goes missing; here we
 * keep its last-seen crop instead. It also survives the hand-off to the Report
 * screen, which has no live frame of its own.
 *
 * A fresh Start resets the tracker ids, so a new run's crops simply overwrite
 * the old ones under the same ids; stale ids never render (only listed rows are
 * looked up), so we don't bother pruning.
 */
export function useLastSeenCrops(status: Status | null): CropMap {
  const [crops, setCrops] = useState<CropMap>({})
  const ref = useRef(crops)
  ref.current = crops

  useEffect(() => {
    if (!status) return
    // In `finished` the D15 setup block keeps re-detecting for run 2, so a crop
    // shown on the report may be refreshed by a post-stop detection of the same
    // id — same instrument, so harmless (and it can only improve a stale crop).
    const seen: { tracker_id: number; thumbnail: string | null }[] =
      status.setup?.detections ?? status.recording?.instruments ?? []

    let next: CropMap | null = null
    for (const item of seen) {
      if (item.thumbnail && ref.current[item.tracker_id] !== item.thumbnail) {
        next ??= { ...ref.current }
        next[item.tracker_id] = item.thumbnail
      }
    }
    if (next) setCrops(next)
  }, [status])

  return crops
}
