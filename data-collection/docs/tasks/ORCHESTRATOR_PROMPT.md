# Orchestrator prompt — recording-mode build (TR1–TR7)

Paste everything below the line into a fresh Claude Code session started at the
repo root (`MVP-OKcamera`, branch `data-collection`).

---

You are the **orchestrator** for building recording mode on the ORC
data-collection dashboard. You do not write feature code yourself — you spawn
subagents per task, verify their output, keep the board honest, and stop the
line when something is off. Work phase by phase; never start a phase before the
previous one is fully green.

## Ground truth — read these first, in this order

1. `data-collection/docs/tasks/BOARD.md` — rules, status table, the recording
   phases (R1–R4), file-ownership map. The board's rules bind you and every
   subagent.
2. `data-collection/docs/RECORDING.md` — the spec. All design decisions in it
   are SETTLED. Do not re-litigate them; do not let a subagent re-litigate them.
3. `data-collection/docs/IMPORT_FORMAT_VIDEO.md` — the output contract.
4. The seven task files `data-collection/docs/tasks/TR1-*.md` … `TR7-*.md` —
   each is a self-contained brief with a FROZEN public interface and numbered
   ACs. Frozen means frozen: a subagent that needs an interface change must
   stop, write the problem in its task Log, and you surface it to Bram instead
   of approving a workaround.

Working directory for all test runs: `data-collection/dashboard/` — the gate is
`uv run pytest` from there, full suite, after every task and every phase.

## Phase plan

**Phase R1 — TR1, TR2, TR3, TR6 in parallel (four subagents).**
They share no files (see the board's ownership map; TR1's single-line
`backend/app.py` carve-out is pre-coordinated — TR5 hasn't started).

- TR1, TR2, TR3: build with **blind TDD** (see below).
- TR6: single implementer subagent, no blind TDD (plain-JS frontend against
  TR5's frozen endpoint table; manual checklist ACs + keep T06's existing
  jsdom checks green).

**Phase R2 — TR4 (one subagent, blind TDD).** Needs TR2 + TR3 merged and TR1's
`tests/recording_fakes.py` present.

**Phase R3 — TR5 (one subagent, blind TDD).** Wires TR1–TR4 into the app.
Extra scrutiny here: it edits `backend/app.py`, and T05's `tests/test_api.py`
passing UNCHANGED is the regression gate.

**Phase R4 — TR7, two halves.**
- Automated half (a subagent): `tests/test_recording_e2e.py` (fake numbered
  camera, TR7 ACs 1–6) + `scripts/spike_fps.py`. Blind TDD is optional here —
  e2e tests ARE the deliverable, so a single careful agent is fine.
- Hardware half (TR7 ACs 7–9: Camo, real weights, the 1080p60 spike): **do not
  attempt it.** Mark TR7 `blocked (hardware — Bram)` with a Log entry listing
  the exact commands to run, and end your run there.

## Blind TDD — how to run it per task

Follow BOARD.md rule 3 exactly: invoke the `/blind-tdd` skill and pass the TR
task file **as** its SPEC.md (do not author a second spec). The skill runs a
test-writer agent (touches `tests/` only, names tests `test_ac03_...` after the
AC they cover) and a separate implementer (touches `backend/` only, never sees
test source), with review before tests run and a blind fix loop after.
Non-negotiables you enforce on top:

- The regression suites named in each task (T02's `tests/test_dataset_writer.py`,
  T03's `tests/test_capture.py`, T05's `tests/test_api.py`) are **never edited**
  and must pass unchanged — they are the proof that image mode survived.
- No real camera, encoder, ffmpeg, or model weights in unit tests — every task
  file names its injectable fakes; hold the test agent to them.
- A task is done only when: every AC is checked off in the task file, the FULL
  suite is green, the header says `done`, and a dated Log entry says what
  shipped.

## Your loop, per phase

1. Set each task header + board table to `in-progress (<agent>)` when you spawn.
2. Spawn the phase's subagents in parallel (one per task, each gets its ONE
   task file + the spec paths above).
3. When a subagent reports done: verify yourself — run the full suite, spot-check
   its ACs against the actual diff (don't take its word), check it stayed inside
   its In-scope file list (`git status`/`git diff --stat`).
4. Commit per completed task on `data-collection` (message: `TRx: <summary>`).
   Do not push unless Bram says to.
5. Only when ALL of a phase's tasks are done + full suite green → next phase.
6. If anything needs an interface change, an out-of-scope file edit, or
   contradicts RECORDING.md: stop that task (`blocked`, Log entry), continue
   unaffected tasks, and report it clearly in your final message.

## Final report to Bram

What shipped per task, the full-suite result, any Log-flagged deviations, and
the TR7 hardware checklist (spike command, real-run steps) ready to execute
with Camo.
