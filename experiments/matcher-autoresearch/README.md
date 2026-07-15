# matcher-autoresearch — contained overnight run

A **self-contained** sandbox for the overnight matcher auto-research. It is
**separate from the chat working on T2**: it reads shared data **read-only** and
writes **only inside this folder**, so the two never collide.

- **Reads (read-only):** `model/data/instruments`, `model/data/other_objects`,
  `matching/data/testing/09-07-26-00{1..4}`, `model/weights/…onnx`.
- **Writes (only here):** `runs/` (per-experiment), `method/`, `champion/`,
  `LEADERBOARD.md`, `TRIED.md`, `FINDINGS.md`.
- **Never touches** `docs/wayfinder/session-linker/assets/T02-*` (the other
  chat's turf).

## Contents

- `program.md` — the research brief (objective, guarded metric, frozen/mutable
  rule, technique menu). **The substance.** A copy of the T02-stage2 program.
- `workflow.mjs` — the budget-paced champion/challenger orchestration
  (Codex leak-check + subagent challengers + `loop-until-budget`). **The engine.**
- `harness/` — the stage-1 code, to be split into a **frozen** eval/loader/split
  and a **mutable** `method/`.
- `setup.sh` — re-runnable scaffold (already applied once).

## How to start it (in a FRESH chat, not the T2 chat)

1. (Optional, 2 min each) accept the DINOv3 HF licence for the ablation, and drop
   a few easy foreign crops (phone/pen/tape) into `model/data/other_objects/`.
2. Open a new chat in this repo and paste:

   > Run the overnight matcher auto-research. Read
   > `experiments/matcher-autoresearch/program.md` and
   > `experiments/matcher-autoresearch/README.md`, then launch
   > `experiments/matcher-autoresearch/workflow.mjs` as a Workflow. Work ONLY
   > inside `experiments/matcher-autoresearch/`; the data is read-only. ultracode

   The `ultracode` keyword + the explicit ask are the Workflow opt-in. **No budget
   target needed** — the loop is **loop-until-dry**: it keeps exploring while it
   finds improvements and stops once it plateaus (3 dry rounds), with a hard cap
   of 30 rounds so it can't run away. If you *do* want a ceiling anyway, add a
   `+Nk` directive and it becomes an extra stop-condition.

3. It runs in the background and writes results continuously. Check
   `LEADERBOARD.md` / `FINDINGS.md` in the morning.

## Resume (if it stops on a session/limit or you kill it)

The run is designed to lose nothing (continuous writes). To continue where it
left off, in the same session:

> Resume the matcher-autoresearch workflow:
> `Workflow({ scriptPath: "experiments/matcher-autoresearch/workflow.mjs",
> resumeFromRunId: "<runId from the first launch>" })`

Unchanged `agent()` calls return cached; only new/edited ones re-run.

## Notes / gotchas (from the last nightly run)

- Budget is **optional** now (loop-until-dry drives it). If you add one, pass a
  real `+Nk` directive, not text — it becomes `budget.total` and acts as an extra
  ceiling on top of the dry-convergence + 30-round cap.
- `codex:codex-rescue` is used for one challenger + the leak-check; if Codex isn't
  set up (`/codex:setup`), those slots degrade gracefully (a dead agent → null,
  and a big unreviewed jump is *not* promoted) — the loop still runs on the other
  challengers.
- All numbers are **directional** (8×15 crops overfit); the deliverable is
  *ranked techniques + recommended params*, confirmed at stage-2 (T07 photos +
  real room). `FINDINGS.md` feeds `model/docs/linker-design.md`.
