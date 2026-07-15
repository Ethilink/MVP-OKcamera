---
id: T05
title: Implementation spec — handoff to the build
type: wayfinder:task
status: open
assignee:
blocked-by: [T01, T03, T04, T06]
---

## Question

Turn the pinned design (T03), the acceptance harness (T04), and Constantijn's
constraints (T01) into the build-ready spec: scoped issues/briefs for the
Claude-agent build in `model/` (Bram's process of choice — blind-TDD or
`/implement`), each sized to one session, acceptance = T04's scenario readout
plus the tracker-interface contract (≤ 1.0 s resolve, no leaks, no −1 ids,
mask/data preserved through ByteTrack).

This is the **edge of the map** — when this closes, the way is clear and
implementation starts. Wayfinder plans; it does not build.

Resolution = the briefs/issues created and linked here; update
`app/docs/DESIGN.md` D8 and `model/docs/tracker-interface.md` open items to
their final wording.

## Progress (2026-07-15) — brief delivered, build started

The build-ready spec exists: [assets/T05/ISSUES.md](../assets/T05/ISSUES.md)
(checker-verified work items W1–W9 + corrections C1–C9). §6.5 grilled with
Bram and pinned in `linker-design.md`. W1 (doc rewrite) + W2 (this trail) done
by the orchestrator; W3–W5/W7–W8 run via Sonnet coder subagents with
orchestrator review, W6 (`SessionLinker`) via blind-TDD. Still owed for
closure: final wording for `app/docs/DESIGN.md` D8 + `tracker-interface.md`
open items, and the T04-scenario acceptance readout.
