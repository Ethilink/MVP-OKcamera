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
export type InstrumentStatus = Schemas["InstrumentStatusModel"]
export type RecordingStatus = Schemas["RecordingStatus"]
export type UsageWindow = Schemas["UsageWindowModel"]
export type InstrumentReport = Schemas["InstrumentReportModel"]
export type Report = Schemas["ReportResponse"]
export type StartedResponse = Schemas["StartResponse"]

// `phase` and `capture_health` are inline string-literal enums on the generated
// StatusResponse; expose them as the standalone unions the frontend references.
export type Phase = Status["phase"]
export type CaptureHealth = Status["capture_health"]
