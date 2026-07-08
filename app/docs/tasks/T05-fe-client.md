# T05 — Frontend API client, types, polling + MSW fixtures

status: todo
depends-on: T01 (contract only — do NOT wait for backend code)
blocks: T06, T07
spec: [api-contract.md](../api-contract.md) (normative); [DESIGN.md](../DESIGN.md) D4, D12, D14

## Goal

Everything the screen tasks share: TypeScript types hand-written from the
contract, a typed fetch client, the 2 Hz polling hook, and the MSW handlers +
scripted fixtures that T06/T07 component tests (and pre-backend dev) run on.

## In scope

`app/frontend/src/api/` (`types.ts`, `client.ts`, `useStatus.ts`),
`app/frontend/src/test/` (MSW `handlers.ts`, `fixtures.ts`), colocated Vitest
tests. Single agent, test-first.

## Out of scope

Screens/components (T06, T07). Generated types (T08 swaps these in).

## Public interface (frozen)

```ts
// src/api/types.ts — mirror api-contract.md exactly (field names verbatim)
export type Phase = "setup" | "recording" | "finished";
export interface SetupStatus { detected_count: number; stable_for_s: number }
export interface InstrumentStatus { tracker_id: number; label: string;
  on_table: boolean; off_since_s: number | null; pickup_count: number }
export interface RecordingStatus { started_at: string; elapsed_s: number;
  on_table_count: number; instruments: InstrumentStatus[] }
export interface Status { phase: Phase; capture_health: "ok" | "stalled";
  model_version: string; setup: SetupStatus | null; recording: RecordingStatus | null }
export interface UsageWindow { off_s: number; on_s: number | null }
export interface InstrumentReport { tracker_id: number; label: string;
  completeness: "present" | "lost"; usage: UsageWindow[] }
export interface Report { started_at: string; stopped_at: string;
  duration_s: number; model_version: string; instruments: InstrumentReport[] }

// src/api/client.ts
// Base URL: const BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000"
// (backend serves on :8000, Vite dev on :5173 — cross-origin, matches T04 CORS).
// All requests + streamUrl are absolute against BASE. Tests/MSW intercept these
// absolute URLs.
export class ApiError extends Error { status: number; detail: string }
export const api: {
  status(): Promise<Status>;
  startRecording(): Promise<{ started_at: string }>;
  stopRecording(): Promise<Report>;
  report(): Promise<Report>;
  streamUrl: string;               // `${BASE}/stream` — for <img src={api.streamUrl}>
};

// src/api/useStatus.ts — poll every `intervalMs` (default 500)
export function useStatus(intervalMs?: number): {
  status: Status | null;          // last good payload (kept during errors)
  error: ApiError | Error | null; // non-null after a failed poll
}

// src/test/fixtures.ts — scripted Status sequences + a canned Report:
//   setupUnstable, setupStable, recordingAllOn,
//   recordingOneOff (instrument off_since_s ticking),
//   finishedStatus (phase "finished": setup block PRESENT & stable,
//     recording null — drives the run-2 gate; T06 AC4b needs it),
//   captureStalled (capture_health "stalled" — T06 AC1 banner + disabled Start),
//   demoReport (2 instruments with windows incl. one lost), plus
//   scriptedHandlers(sequence) -> MSW handlers stepping through it.
```

## Acceptance criteria

- **AC1** `tsc --noEmit` passes; types match the contract field-for-field
  (checked by fixtures compiling against them).
- **AC2** `api.*` methods hit the right method+path (asserted via MSW) and
  return parsed bodies; a 409 rejects with `ApiError` carrying `detail`.
- **AC3** `useStatus` (fake timers) polls at the interval, exposes the latest
  payload, and does not stack overlapping requests when a response is slow.
- **AC4** A failed poll sets `error` while keeping the last good `status`; a
  subsequent success clears `error`.
- **AC5** `scriptedHandlers` steps through its sequence one response per poll
  and holds the last entry thereafter; `demoReport` exercises both
  `completeness` values and an open (`on_s: null`) window.
- **AC6** `npm run dev` with MSW enabled (dev-only worker) renders a page
  proving the loop: raw `useStatus` output updating on screen — so T06/T07 can
  develop before the backend exists.

## Log

- (append dated one-liners here)
