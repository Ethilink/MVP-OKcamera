// D14: the payload types are GENERATED from the backend's `openapi.json`, not
// hand-written. Regenerate with `npm run gen:api` (the backend must be running
// first — see RUNBOOK.md §"Generated types"). This file maps the generated
// component-schema names (which mirror the FastAPI Pydantic model names) onto
// the contract-facing names the frontend uses (api-contract.md §/status,
// §/report). Contract drift between the backend and these names now becomes a
// TypeScript compile error here rather than a silent runtime mismatch.

import type { components } from "./schema"

type Schemas = components["schemas"]

export type Status = Schemas["StatusResponse"]
export type SetupStatus = Schemas["SetupStatus"]
export type Detection = Schemas["DetectionModel"]
export type InstrumentStatus = Schemas["InstrumentStatusModel"]
export type RecordingStatus = Schemas["RecordingStatus"]
export type UsageWindow = Schemas["UsageWindowModel"]
export type InstrumentReport = Schemas["InstrumentReportModel"]
export type Report = Schemas["ReportResponse"]
export type StartedResponse = Schemas["StartResponse"]
// T11: the runtime detection-confidence control metadata (always on /status).
export type DetectorControl = Schemas["DetectorControlModel"]

// `phase` and `capture_health` are inline string-literal enums on the generated
// StatusResponse; expose them as the standalone unions the frontend references.
export type Phase = Status["phase"]
export type CaptureHealth = Status["capture_health"]

// `detection.state` is the generated per-item identity union (T11/B6).
export type DetectionState = Detection["state"]

// The backend serialises `blocking_reason` as a plain `string | null`, but it is
// one of a closed set (api-contract §setup). Narrowed here for the operator-copy
// map. NOTE: consumers cast the raw `string | null` to this, so an unknown future
// reason does NOT fail to compile — the copy/badge `switch`es fall through to a
// fail-safe default (Track stays disabled), which is the intended behaviour.
export type BlockingReason =
  | "recognising"
  | "missing_instruments"
  | "unknown_objects"
  | "hold_steady"
