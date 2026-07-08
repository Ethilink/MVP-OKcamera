# ORC demo app build — task board

Source spec: [`DESIGN.md`](../DESIGN.md) + the frozen
[`api-contract.md`](../api-contract.md). Each `T##-*.md` file is a
**self-contained brief**: an agent gets ONE task file (plus the spec) and has
everything needed — goal, frozen interface, scope, numbered acceptance
criteria. No GitHub issues; this folder is the tracker.

> This board builds the **MVP demo app** (`app/`), not the data-collection
> dashboard. Do not treat `data-collection/` as a spec — only T03 deliberately
> adapts its capture-thread pattern.

## Rules for agents

1. **Claim before you touch code.** Set `status: in-progress (<agent-name>)` in
   your task file's header AND in the status table below. One agent per task.
2. **Stay inside "In scope".** Every task declares a **Public interface** —
   frozen contracts other tasks build against. If your task seems to require
   changing another task's interface, STOP, write the problem in your Log, set
   `status: blocked`, and surface it to Bram.
3. **Backend tasks (T02–T04): blind-TDD.** Run the `/blind-tdd` skill and pass
   the T## task file **as** its SPEC.md — do not author a duplicate spec. Test
   agent writes failing tests from the ACs + Public interface only (touches
   `tests/` only, names tests `test_ac05_...`); implementer makes them green
   (never edits tests); Codex + Opus review before tests run; fix loop uses
   `pytest --tb=line`.
4. **Frontend tasks (T05–T07): single agent, test-first.** Vitest + React
   Testing Library + MSW component tests are REQUIRED (D12), written before the
   component. Then a **visual verification pass with Claude-in-Chrome**: drive
   the UI in the browser, screenshot each AC'd state, note the result in the Log.
   The normal verification substrate is **MSW-served dev fixtures** (T05's
   `scriptedHandlers`) — frontend tasks do NOT wait for the backend. When the
   `--fake` backend already exists (`uv run orc-demo --fake` + `npm run dev`),
   verify against it too; T08 re-verifies the whole thing against `--fake`.
5. **Review before done (all tasks).** When every AC passes and tests are green,
   set `status: review`, not `done`. A **second agent that did not write the
   code** reviews it — Codex via `codex:rescue` and/or a fresh Claude (Bram's
   rule: Claude + Codex review everything). Only after sign-off (logged) →
   `done`. Backend blind-TDD tasks already bake in the Codex+Opus pre-test
   review (rule 3); this rule additionally covers the frontend tasks.
6. **Done means**: every AC checked off, tests green (`uv run pytest` in
   `app/backend/`, `npm test` in `app/frontend/`), a reviewer signed off (rule
   5), status `done`, dated Log entry describing what shipped.
7. **Log everything at handoff.** Append a dated one-liner on claim, block,
   handoff, and completion.

Status values: `todo` · `in-progress (<agent>)` · `blocked` · `review` · `done`.
The task file header is the source of truth; the table below is a mirror.

## Status

| id  | task                                             | depends on               | status | owner |
|-----|--------------------------------------------------|--------------------------|--------|-------|
| T01 | [Scaffold backend + frontend](T01-scaffold.md)   | —                        | done   | claude |
| T02 | [Session state machine + report](T02-session.md) | T01                      | review | claude |
| T03 | [Capture-infer loop + overlay](T03-capture.md)   | T01                      | review | claude |
| T04 | [FastAPI layer](T04-api.md)                      | T02, T03                 | review | claude |
| T05 | [Frontend API client + polling](T05-fe-client.md)| T01 (contract only)      | review | claude |
| T06 | [Setup + Recording screens](T06-fe-live.md)      | T05                      | review | claude |
| T07 | [Report screen + Usage timeline](T07-fe-report.md)| T05                     | review | claude |
| T08 | [Integration + demo runbook](T08-integration.md) | T04, T06, T07            | review | claude |

> Backend T01–T04 built via the `orc-backend-nightly` workflow (blind-TDD +
> dual-Opus review + adversarial verify on T02/T03) and committed
> (`2c8eb11`→`19cf144`); T08 integration committed (`8920359`). Backend suite
> 91 passed; frontend 42 passed; `orc-demo --fake` e2e smoke green. `review`
> (not `done`) because the Codex sign-off (Bram's dual-review rule) and T08's
> visual Chrome pass are still pending.

## Phases / parallelism

- **Phase 0** — T01 (one agent; creates both project skeletons + fakes).
- **Phase 1** — T02, T03, T05 **in parallel** (disjoint files: `session.py` /
  `capture.py`+`render.py` / `frontend/src/api`). T05 builds against
  `api-contract.md`, not against backend code.
- **Phase 2** — T04 (needs T02+T03), and T06, T07 **in parallel** (near-disjoint:
  live screens vs report components; both consume T05's client + MSW fixtures).
  One shared file, `src/screens/ReportScreen.tsx`: T06 lands a placeholder with
  the frozen `{ onNewRecording }` prop (so App routing compiles), T07 replaces
  it — see T06 "T07 seam". T07 owns the final file.
- **Phase 3** — T08 (end-to-end against `--fake`, then real-camera smoke).

## Escalation

Anything ambiguous → Bram decides. The re-identification risk (DESIGN D8) is
Constantijn's; if a task discovers a NEW cross-seam problem, log it here, don't
solve it locally.

### Open findings

- **[2026-07-09] Scenario→wall-clock time dilation (~1.37×) — T01/T03 seam,
  demo-timing only.** `FakeCaptureSource.read()` (T01) sleeps `1/fps` *before*
  yielding a frame, but T03's capture loop adds ~40 ms/frame (1080p mask-render
  + JPEG-encode), so effective throughput is ~7.3 fps, not 10. Scenario time
  (tracker `frame_count/fps`) runs slower than the wall-clock `t` the `Session`
  stamps windows with, so scripted events at t=20/35/50 s surface at wall
  ≈28/48/68 s. **Not a correctness bug:** events fire in the right order and
  recorded report values are correct (Session stamps from the wall `t` passed
  in); only *when* they appear drifts. Surfaced by T08's e2e smoke; RUNBOOK
  rehearsal timings were corrected to wall-clock. Fix option (for the T01/T03
  owner): make `read()` a rate-limiter that subtracts elapsed loop time instead
  of a fixed pre-sleep, so scenario-seconds ≈ wall-seconds as the docstring
  claims. Bram to decide whether the demo needs true-time or the caveat suffices.

- **[2026-07-09] Codex post-build review — 4 edge-case bugs (none on the demo
  happy-path; full suite + adversarial verify + e2e `--fake` smoke all pass).**
  Codex session `019f43fa-40b6-7ca0-9006-b40691b9c518`. Bram to triage/assign:
  - **CONFIRMED ❌ `capture.py:94` `stop()` doesn't guarantee release/dead on a
    hung read.** Only sets `_stop` + `join(timeout=2.0)`; `cap.release()` is only
    in the thread's `finally` (:150). A real camera blocked in `read()` >2 s →
    `stop()` returns with thread alive, capture unreleased, `health != "dead"`
    (T03 AC9 gap). Fake returns promptly, so demo unaffected; matters for
    real-camera + the crash playbook. Fix: release + mark dead even on join
    timeout. Small, but wants a test — do supervised, not blind at night.
  - **CONFIRMED ❌ `capture.py:79` `start()` has no already-running guard.**
    Double-`start()` spawns two daemon threads on one tracker (violates "exactly
    one thread"). Low real-world risk (app starts once); add a guard + test.
  - **NEEDS ADJUDICATION ❌ `main.py:89` `/stream` reads `generation` then
    `snapshot()` as two ops** → a publish landing between them can re-yield a
    frame while recording the stale generation (T04 AC5 "no duplicate frames").
    Real TOCTOU on the stream path; cosmetic-ish (video feed). Fix = snapshot +
    generation under one lock read.
  - **NEEDS ADJUDICATION ❌ `session.py:183` projected debounce transition not
    committed if the next observe contradicts it.** Codex scenario: `start(0)`,
    `observe(0.1,{9})`, then `observe(2.0,∅)`, `stop(3.0)` → instrument 9 absent
    from report though `recording_status(1.2)` had projected it confirmed. MAY
    be a misread of the intentionally NON-mutating projection (AC10) rather than
    a bug, and the scenario (one sparse observe, no ~10 fps stream between) is
    unrealistic. Bram/spec call — this is the correctness heart, do NOT patch
    blindly.
  - ✅ Codex confirmed: publish path lock-correct, exception isolation/rate-limit
    (AC4/6/7/8), `/status` shape + 409/503 + tz-aware timestamps + stop==report,
    scenario boundary convention, and that the time-dilation finding above does
    NOT corrupt recorded report values (they're wall-clock, not scenario-frame).
