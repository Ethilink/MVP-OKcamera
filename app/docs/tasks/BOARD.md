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
| T01 | [Scaffold backend + frontend](T01-scaffold.md)   | —                        | todo   | —     |
| T02 | [Session state machine + report](T02-session.md) | T01                      | todo   | —     |
| T03 | [Capture-infer loop + overlay](T03-capture.md)   | T01                      | todo   | —     |
| T04 | [FastAPI layer](T04-api.md)                      | T02, T03                 | todo   | —     |
| T05 | [Frontend API client + polling](T05-fe-client.md)| T01 (contract only)      | todo   | —     |
| T06 | [Setup + Recording screens](T06-fe-live.md)      | T05                      | review | claude |
| T07 | [Report screen + Usage timeline](T07-fe-report.md)| T05                     | review | claude |
| T08 | [Integration + demo runbook](T08-integration.md) | T04, T06, T07            | todo   | —     |

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
