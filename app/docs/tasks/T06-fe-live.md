# T06 — Setup + Recording screens

status: todo
depends-on: T05
blocks: T08
spec: [DESIGN.md](../DESIGN.md) §What the demo does 1–2, D2–D4, D12, D13; [api-contract.md](../api-contract.md) §/status

## Goal

The operator's live screen for phases `setup` and `recording`: video feed,
Start/Stop control with the human-judgment gate, and the live per-instrument
panel.

## In scope

`src/screens/LiveScreen.tsx` + `src/components/{VideoFeed,StartStopControl,
InstrumentPanel,HealthBanner}.tsx`, colocated RTL tests on T05's
`scriptedHandlers`. `App.tsx` routing (see **Routing** below). Single agent,
test-first, then Claude-in-Chrome verification (BOARD rule 4).

## Routing (normative — App.tsx owns it)

App routes on `status.phase` plus one local flag `newRecordingRequested`
(boolean, default false):

- `phase == "recording"` → **LiveScreen** in its recording layout.
- `phase == "setup"` → **LiveScreen** in its setup layout.
- `phase == "finished"` and NOT `newRecordingRequested` → **ReportScreen** (T07).
- `phase == "finished"` and `newRecordingRequested` → **LiveScreen** in its
  **setup layout** (D15: `finished` still publishes the `setup` block, so the
  live feed + gated Start work identically). This is the "run 2" re-confirmation.

Flag transitions: ReportScreen's "New recording" button sets
`newRecordingRequested = true` (it does NOT call the API directly — T07). The
Start button in the setup layout calls `api.startRecording()`; App clears
`newRecordingRequested` whenever `phase == "recording"` is observed. LiveScreen
picks setup vs recording layout from `phase` (`finished` routed here ⇒ setup
layout).

**Back to report (no dead-end).** Because "New recording" only sets a local flag
and the backend still holds the report (D7: in memory until the next Start), the
setup layout — *when it is showing because of the flag*, i.e. `phase=="finished"`
∧ `newRecordingRequested` — MUST render a **"Back to report"** control that
clears `newRecordingRequested` (→ App routes back to ReportScreen). It is absent
in a genuine `phase=="setup"` (run 1, no report exists) and once `recording`
starts. This makes an accidental "New recording" click recoverable without any
persistence or contract change. (Recording *history* across sessions is an
explicit non-goal for the MVP — DESIGN D7.)

**T07 seam (keeps T06/T07 parallel-safe).** App.tsx renders T07's `ReportScreen`
in the `finished`/no-flag case. T06 does NOT implement it — it creates a minimal
**placeholder** `src/screens/ReportScreen.tsx` with the frozen signature
`export function ReportScreen(props: { onNewRecording: () => void })` (renders
nothing but a "New recording" button wired to `props.onNewRecording`). T07
**replaces** this file. This is the one shared file; both briefs name it and the
prop contract is identical (T07 §Public interface), so the two agents don't
collide.

## Out of scope

Report visuals (T07). API/polling internals (T05 — consume, don't modify).

## UI behaviour (normative)

- **VideoFeed**: `<img src={api.streamUrl}>`, fixed 16:9 box, shadcn Card. Per
  **D16**, an `onError` handler replaces the `<img>` with a plain styled
  "no stream (dev mode)" panel (no image asset). MSW can't intercept the MJPEG
  `<img>` load, so under RTL/MSW and pre-backend `npm run dev` the panel is what
  renders; real video appears only against `--fake`/real backend (AC7). RTL:
  assert the fallback panel shows when the `<img>` errors — don't assert a live
  frame.
- **Setup**: feed + "N instruments detected, stable for Xs" + **Start**.
  Start enabled ⟺ `phase ∈ {setup, finished}` ∧ `capture_health=="ok"` ∧
  `detected_count ≥ 1` ∧ `stable_for_s ≥ 2` (gate lives HERE, per contract).
  Disabled Start shows why (e.g. "waiting for stable detections…").
- **Recording**: feed + elapsed timer + `on_table_count` + **Stop** (always
  enabled) + **InstrumentPanel**: one row per instrument (sorted by
  tracker_id): label, ON TABLE / OFF TABLE badge (shadcn Badge,
  green/destructive), `off_since_s` ticking when off, pickup count.
  **Ticking ownership:** `off_since_s` and the elapsed timer come from the poll
  (2 Hz), but the UI interpolates smoothly between polls **client-side** (last
  polled value + a local clock delta since that poll arrived); each fresh poll
  re-anchors. The backend value is always the source of truth on arrival — the
  client only fills the ½-second gaps so the numbers don't visibly stutter.
- **HealthBanner**: `capture_health=="stalled"` or poll `error` → prominent
  destructive banner; never crash on `status:null` (skeleton state).
- Start/Stop call the API; UI state comes ONLY from the next `/status` poll
  (no optimistic phase flips); a 409 on Start surfaces non-fatally.

## Acceptance criteria (RTL + MSW unless marked Chrome)

- **AC1** `setupUnstable` fixture → Start disabled with reason;
  `setupStable` → enabled; `capture_health:"stalled"` → disabled + banner.
- **AC2** Clicking Start POSTs `/recording/start`; UI switches to the
  recording layout only after a poll returns `phase:"recording"`.
- **AC3** `recordingOneOff` → the off instrument shows OFF TABLE +
  `off_since_s` (rendered `12s`-style) + pickup count; others show ON TABLE;
  header shows `on_table_count`; rows sorted by tracker_id.
- **AC4** Clicking Stop POSTs `/recording/stop`; when the poll returns
  `phase:"finished"` (and `newRecordingRequested` is false), App routes to
  ReportScreen (LiveScreen no longer rendered).
- **AC4b** Run-2 restart (uses `finishedStatus` fixture): with
  `phase:"finished"` and `newRecordingRequested` set, App renders LiveScreen's
  setup layout driven by the `finished` payload's `setup` block; the gate
  behaves exactly as in `setup` (disabled until `stable_for_s ≥ 2`); clicking
  Start POSTs `/recording/start` and, when the poll returns `phase:"recording"`,
  App shows the recording layout with `newRecordingRequested` cleared.
- **AC4c** Back-to-report (uses `finishedStatus` fixture): with
  `phase:"finished"` and `newRecordingRequested` set, the setup layout shows a
  "Back to report" control; clicking it clears `newRecordingRequested` so App
  routes to ReportScreen. The control is NOT rendered for a genuine
  `phase:"setup"` payload (run 1, no report).
- **AC5** Poll failure mid-recording → banner + last panel stays; recovery
  clears the banner.
- **AC6** 409 on Start → toast/inline error, no crash, polling continues.
- **AC7** *(Chrome)* Against MSW dev fixtures (or `--fake` backend if T04 is
  done): drive setup → start → live panel → stop in the real browser;
  screenshot each state; log the pass in the Log.

## Log

- (append dated one-liners here)
