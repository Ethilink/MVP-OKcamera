# T08 — Integration, generated types, demo runbook

status: todo
depends-on: T04, T06, T07  (T05 is an indirect dependency, pulled in via T06/T07)
blocks: — (last task)
spec: [DESIGN.md](../DESIGN.md) D14, §Run modes; [BOARD.md](BOARD.md) rule 4

## Goal

Prove the whole thing end-to-end against `--fake`, replace hand-written TS
types with generated ones, and write the runbook that gets the demo running on
demo day — including the real-camera/real-weights swap.

## In scope

Type generation wiring, small integration fixes anywhere (log which task's
area you touched), `app/docs/RUNBOOK.md`, final Chrome verification. Single
agent + Claude-in-Chrome.

## Out of scope

New features. Contract changes (a discovered mismatch is a bug in T04 or T05 —
fix the code, not `api-contract.md`; if truly impossible, `blocked` + Bram).

## Work items

1. **Generated types (D14):** `npm run gen:api` fetches `openapi.json` from
   the running backend and runs `openapi-typescript`; replace the hand-written
   payload interfaces in `src/api/types.ts` with (re-exports of) generated
   ones. Frontend tests must still pass unchanged — differences are bugs.
2. **End-to-end fake run:** backend `uv run orc-demo --fake` + `npm run dev`
   (MSW OFF against a real backend), then with Claude-in-Chrome drive the full
   demo: setup (gate waits for stability) → Start → watch scripted instrument 1
   go OFF TABLE (~t=20s) and return (~t=35s), instrument 3 go off (~t=50s) →
   Stop → report shows instrument 1 present with one closed window, instrument
   3 LOST with an open bar. Screenshot every phase.
3. **RUNBOOK.md:** demo-day sheet — hardware checklist (tripod, lighting,
   Camo/camera index), exact commands for fake and real mode, the weights
   handoff from Constantijn (`--weights` path), pre-demo rehearsal script
   (10 steps, 5 minutes), failure playbook (stalled camera → banner → restart
   command; backend crash mid-demo → restart, re-Start recording).
4. **Real-mode smoke (only if weights + camera available):** `--camera N
   --weights PATH`, verify overlay + one pickup round-trips to a report; else
   record in the Log that this remains open for Bram.

## Acceptance criteria

- **AC1** With the backend already running (`uv run orc-demo --fake` started
  FIRST — `gen:api` fetches `openapi.json` over HTTP), `npm run gen:api` works
  from a clean checkout; generated types compile and all frontend tests pass
  against them. Document the "backend up, THEN gen:api" ordering in RUNBOOK.md.
- **AC2** Full-suite green: `uv run pytest` (backend) and `npm test` +
  `npm run build` (frontend) from clean.
- **AC3** The scripted end-to-end run above passes in a real browser with
  screenshots in the Log — every phase, gate behaviour, live OFF TABLE flip,
  and the final report exactly matching `DEFAULT_SCENARIO`.
- **AC4** `RUNBOOK.md` exists and a cold reader (Bram) can start fake mode
  from it in < 5 minutes.
- **AC5** Any cross-task fix is logged in BOTH this Log and the owning task's
  Log (one line each).
- **AC6** Real-mode smoke (work item 4) is **conditional**: if weights + camera
  are available, `--camera N --weights PATH` shows the overlay and one pickup
  round-trips to a report (screenshot in Log). If not available, it **may be
  skipped but MUST be logged** as "real-mode unverified — open for Bram" so the
  gap is explicit, not silent.

## Log

- (append dated one-liners here)
