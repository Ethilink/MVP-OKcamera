# T07 — Report screen + Usage timeline

status: todo
depends-on: T05
blocks: T08
spec: [DESIGN.md](../DESIGN.md) §What the demo does 3, D12; [api-contract.md](../api-contract.md) §/report

## Goal

The payoff screen: after Stop, per instrument a **Usage timeline** (off-table
windows as bars on a shared time axis) and a **Completeness** badge. This is
the one component with real visual design in the app — it's what UZ Leuven
photographs.

## In scope

`src/screens/ReportScreen.tsx` + `src/components/{UsageTimeline,
CompletenessBadge,ReportSummary}.tsx` + `src/lib/timeline.ts` (pure geometry),
colocated tests. Single agent, test-first, then Claude-in-Chrome verification
(BOARD rule 4).

## Out of scope

LiveScreen/App routing (T06 owns `App.tsx` — export `ReportScreen` for it).
Fetching machinery (use T05's `api.report()`).

## Public interface (frozen)

```ts
// src/lib/timeline.ts — pure, unit-tested geometry (no React)
export interface Segment { leftPct: number; widthPct: number; open: boolean }
export function usageSegments(duration_s: number, usage: UsageWindow[]): Segment[]
// off_s/on_s -> percentage segments; on_s null -> open segment to 100%.
// Input assumptions (backend guarantees per api-contract §/report invariants):
// windows are sorted and non-overlapping — assert in a dev guard, don't
// re-sort/merge. Degenerate input (duration 0, window past duration) clamps
// into [0,100], never NaN.

export function axisTicks(duration_s: number): { pct: number; label: string }[]
// "nice" step = smallest of {1,2,5}×10ⁿ seconds that yields ≤ 6 ticks across
// duration_s (guarantees a solution for every duration; typically 3-6 ticks,
// minimum 3 for any non-trivial duration). labels mm:ss. First tick at 0%
// (0:00), last tick ≤ 100% (the final nice tick that fits; need not hit 100).

// src/screens/ReportScreen.tsx — props: { onNewRecording: () => void }
//   fetches api.report() on mount, renders:
//   ReportSummary (duration mm:ss, instrument count, N lost, model_version)
//   one row per instrument: label · UsageTimeline · CompletenessBadge
//   "New recording" button -> calls onNewRecording() (App then routes to the
//   live setup view; the gated Start there is the real POST /recording/start).
//   It does NOT call api.startRecording() itself — D15/T06 routing owns run 2.
```

## UI behaviour (normative)

- **UsageTimeline**: horizontal track = full recording; on-table = quiet
  background; off-table window = solid destructive-tinted bar; open window
  (never returned) = bar to the end with a visually distinct treatment +
  "never returned" in its tooltip/label. Shared mm:ss axis; empty usage reads
  "never picked up".
- **CompletenessBadge**: `present` → green "PRESENT"; `lost` → destructive
  "LOST". A lost instrument's whole row is visually flagged.
- Handles `api.report()` 409/failure with a non-crashing error state ("no
  finished recording") — the screen can mount early.

## Acceptance criteria

- **AC1** (unit) `usageSegments(100, [{off_s:20,on_s:35}])` →
  `[{leftPct:20,widthPct:15,open:false}]`; open window → widthPct to 100 and
  `open:true`; `[]` → `[]`; clamping cases produce values in [0,100], no NaN.
- **AC2** (unit) `axisTicks` returns 3–6 strictly-monotonic ticks at a nice
  step ({1,2,5}×10ⁿ s) ≤ 6 ticks, first at 0% ("0:00"), last at ≤100%, mm:ss
  labels ("1:20"); assert the step choice for a few durations incl. an awkward
  one (e.g. 13 s, 100 s, 336 s) — every duration must yield a valid result.
- **AC3** (RTL, `demoReport` fixture) every instrument renders exactly one
  row: label, timeline, badge; lost instrument shows LOST + flagged row +
  open-ended bar; present-with-window shows a closed bar; never-picked-up
  shows "never picked up".
- **AC4** (RTL) summary shows duration mm:ss, counts, model_version;
  clicking "New recording" invokes the `onNewRecording` prop exactly once and
  does NOT call `api.startRecording()` (the gated Start in T06's setup view does).
- **AC5** (RTL) report fetch failing → non-crashing error state, both for a
  409 ("no finished recording") and a generic network failure (thrown
  fetch/500).
- **AC6** *(Chrome)* Render with `demoReport` (MSW dev or `--fake` backend
  full run): screenshot; verify bars visually match the window numbers and
  the screen reads at presentation distance (this is the demo money-shot);
  log the pass.

## Log

- (append dated one-liners here)
