// Hand-written from api-contract.md (§/status, §/report). Field names mirror the
// contract verbatim. D14: these get swapped for openapi-typescript generation at
// integration (T08); until then this file IS the seam.

export type Phase = "setup" | "recording" | "finished"

export type CaptureHealth = "ok" | "stalled"

export interface SetupStatus {
  detected_count: number
  stable_for_s: number
}

export interface InstrumentStatus {
  tracker_id: number
  label: string
  on_table: boolean
  off_since_s: number | null // null when on_table
  pickup_count: number
}

export interface RecordingStatus {
  started_at: string // ISO-8601
  elapsed_s: number
  on_table_count: number
  instruments: InstrumentStatus[]
}

export interface Status {
  phase: Phase
  capture_health: CaptureHealth
  model_version: string
  // present when phase == "setup" | "finished"; null while "recording"
  setup: SetupStatus | null
  // present only when phase == "recording"; null otherwise
  recording: RecordingStatus | null
}

export interface UsageWindow {
  off_s: number
  on_s: number | null // null = never came back -> "missing"
}

export interface InstrumentReport {
  tracker_id: number
  label: string
  completeness: "present" | "missing"
  usage: UsageWindow[] // off-table windows, chronological, non-overlapping
}

export interface Report {
  started_at: string // ISO-8601
  stopped_at: string // ISO-8601
  duration_s: number
  model_version: string
  instruments: InstrumentReport[]
}

export interface StartedResponse {
  started_at: string
}
