# Tracker autoresearch

You are running an autonomous research loop to improve multi-object tracking of surgical
instruments on a tray, inspired by [Karpathy's `autoresearch`](https://github.com/karpathy/autoresearch):
edit, run, score, keep-or-discard, repeat.

## The point is research, not busywork

Don't just try random hyperparameter combinations. For each iteration:

1. Form a **hypothesis** about why the current score is what it is (e.g. "SORT is losing
   short tracks right after occlusion because `lost_track_buffer` is too short relative to
   how long instruments get hidden behind hands" or "GIoU penalizes small, nearby boxes too
   harshly for this camera angle — try CIoU").
2. Make **one change** in `experiment.py` that tests that hypothesis.
3. Run it, look at the result, and write down what you learned in `notes` — win or lose. A
   run that makes the score worse but teaches you something concrete about *why* is not a
   wasted iteration. A run where the score improves but you don't know why is worth a
   follow-up to find out, since it won't generalize reliably otherwise.

Before editing code, it's often worth grounding the hypothesis first:

- Search the `company-brain` Obsidian vault, under `workspaces/surgical-sets` (see this
  repo's root `CLAUDE.md` for how to reach it) for prior domain context — how the tray/camera
  setup works, what's already been learned about occlusion patterns, instrument shapes, etc.
- Search the web/docs for how a specific tracker algorithm or association strategy actually
  behaves (e.g. what OC-SORT's velocity-consistency term is for, or how ByteTrack's two-stage
  matching differs from SORT's single-stage IoU matching) before assuming it'll help here.
- Read the codebase — `orc_model`, the `trackers` package source (it's a git dependency, so
  its source is available in the venv) — rather than guessing at behavior from the outside.

## What you can and can't touch

- **Only edit `experiment.py`** (and any new helper modules you create alongside it in this
  directory, e.g. `custom_tracker.py`). That's the one file this loop is about.
- **Never edit** `prepare.py`, `eval.py`, or `runs.jsonl` directly. They're the fixed
  scoring/logging infrastructure — if you could change how scoring or logging works, a bad
  run could quietly rewrite its own grade. If you think the metric itself is flawed, say so
  in `notes` and flag it to the human rather than changing `eval.py`.
- **Never edit** anything under `model/src/orc_model` — that's shared production code, out
  of scope for this playground loop.

## Freedom within `experiment.py`

Full freedom, not just hyperparameter tuning:

- Swap `SORTTracker` for another tracker in the `trackers` package (ByteTrack, DeepSORT,
  whatever's available), or write entirely custom tracking/association logic.
- Change the detector confidence threshold, the target FPS, the IoU variant, post-processing,
  ensembling multiple trackers, whatever you think the hypothesis calls for.
- The only contract you must keep: consume `prepare.load_cached_detections(clip_name)` for
  detections (don't re-run the detector — it's cached for a reason), and report results
  through `eval.score_clip(...)` and `eval.log_run(...)` so every run is scored and logged
  consistently.

## The loop

1. Read `runs.jsonl` to see the current best `aggregate_score` and skim recent `notes` so you
   don't repeat an already-tried idea.
2. Form a hypothesis (see above).
3. Edit `experiment.py`.
4. Run it: `uv run experiment.py` (from this directory).
5. Compare the new `aggregate_score` (over `TRAIN_CLIPS`) to the current best.
   - **Better or equal**: copy `experiment.py` to `best_experiment.py` as the new checkpoint
     to build on next.
   - **Worse**: revert `experiment.py` from `best_experiment.py` before starting the next
     idea, so you're always iterating from the best known state.
   - Either way, `experiment.py`'s `main()` already calls `eval.log_run(...)` — make sure the
     `notes` argument captures the hypothesis and what you learned, not just "tried X".
6. Also look at the `holdout_score` (the `HOLDOUT_CLIP`, excluded from `aggregate_score`).
   If train score keeps improving while holdout score gets worse, you're overfitting
   hyperparameters to the 5 training clips specifically — that's worth a note and a course
   correction (e.g. prefer changes that are more likely to generalize, like fixing an
   association strategy over hand-tuning thresholds).
7. Repeat.

There's no fixed time or iteration budget here (unlike the original `autoresearch`'s 5-minute
GPU-training budget — tracking over cached detections is fast on its own). Keep going for as
long as you're finding real hypotheses worth testing; if you run out of ideas, say so rather
than continuing to poke randomly at the same three knobs.
