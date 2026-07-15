# orc-e2e-autoresearch — overnight brief

Push the ORC **session linker** toward the best *validated* accuracy on the two
demo-script takes, and lock a recommended config for the UZ Leuven demo
(2026-07-20). Runs autonomously overnight, **sandbox-only**, writes continuously.

## The one-paragraph situation

The shipped pipeline (RF-DETR → Deep OC-SORT → `SessionLinker` with the SRC
matcher + T08 gallery binding) has **never been scored end-to-end** against ground
truth — only human trace-reading. This run closes that gap and then searches for
improvements. Ground truth for the two July-15 demo takes is the scripted
checklist in `model/docs/demo-validation.md` (Bram filmed against it). The COCO
`annotations.json` are detection/mask GT only — NOT re-ID GT (per-annotation ids).

## Two evaluation surfaces (both used)

1. **Crop-level frozen eval — the trustworthy graded optimizer.**
   `experiments/matcher-autoresearch/frozen/` — 5-seed leave-frame-group-out CV on
   8×15 instrument crops + 60 foreign crops. Metric: re-ID top-1 (±seedBand),
   foreign-reject (hard floor `F ≥ 0.9733`), twin errors. Champion = **SRC**
   (`matching/champion.py`, CV re-ID 0.9333). This is where matcher/embedder
   changes are *graded* (it re-sweeps hyperparameters — essential for a backbone
   swap, whose `alpha/tau/cos_tau` need recalibration).

2. **End-to-end regression gate — `experiments/e2e-autoresearch/score_e2e.py`.**
   Replays the REAL pipeline from cache (`caches/take-{a,b}.dets.npz`, ~80–125 s
   each via `model/scripts/replay_session.py --from-cache`) and scores the trace
   against `gt_events.json` (built from `demo-validation.md`). Reports per take:
   `roster_ok`, `foreign_reject_rate` (must stay 1.0), `link_retention` (correct
   re-links kept), `wrong_links`/`regressions`, `hardcase_changes`, `demo_safe`,
   and `binding_matches_baseline`. **Validated:** shipped config → `demo_safe:True`
   on both takes; binding-disabled → correctly flagged `rebound/demo_safe:False`.

   Because replay is from fixed detections and OC-SORT uses its own mobilenet
   embedder, **raw track ids are identical across all matcher/embedder/gallery
   challengers** — only linker outcomes change. With binding on, `session id ==
   specimen == fixed physical tool`, so `linked:5` = "re-identified as instrument5"
   and is directly comparable across configs. A challenger that re-binds differently
   is flagged `rebound` (review), never silently scored.

## The two documented hard cases (do NOT fake them)

Take B has two genuine fail-safe rejects (currently `unknown`, correct behaviour):
`raw 72` @302s (twin ring-forceps swap) and `raw 88` @368s (flipped instrument).
demo-validation.md, MAP.md, and T08-SPEC.md forbid lowering `tau/margin/cos_tau`
to pass them (a wrong link on camera is worse than a safe "Unknown"). A challenger
may only fix them via a legitimate method/backbone/augmentation win that keeps
`foreign_reject_rate == 1.0`, adds **zero** `wrong_links`, and does not lower any
threshold. Any hardcase change is FLAGGED for human review, never auto-counted.

## Priority levers (highest expected value first)

1. **Flip/rotation gallery augmentation (§8)** — demo-validation.md explicitly names
   this as the next experiment for the 368s flip. Embed horizontally-flipped and
   180°-rotated views into each gallery. Test end-to-end: does raw 88 become a
   correct link while foreign-reject and all other links hold? Also grade on crop eval.
2. **DINOv3-B backbone swap** — cached (`~/.cache/huggingface/.../dinov3-vitb16-pretrain-lvd1689m`),
   never run. `ChampionMethod(model_id="facebook/dinov3-vitb16-pretrain-lvd1689m")`
   (768-dim CLS, same as DINOv2-B). **Must re-sweep `alpha/tau/size_alpha/cos_tau`
   on the crop frozen eval** — a naive swap without recalibration is misleading.
   Also try dinov3-vits16. Winner → end-to-end confirmation.
3. **Novel matcher families** — the matcher-autoresearch loop; rotate families NOT in
   `experiments/matcher-autoresearch/TRIED.md` (SRC survived ~25 families / rounds 0–9;
   the round-9 hyperbolic candidate `0.9833` CV is unvalidated — reconstruct + leak-check it).

## Hard rules

- **Sandbox-only.** Edit only under `experiments/e2e-autoresearch/` and
  `experiments/matcher-autoresearch/runs|method`. NEVER edit `model/src/**`,
  `matching/**`, or any shipping code. End-to-end tests inject variants by
  monkeypatching `orc_model.pipelines.matching.ChampionMethod` BEFORE `load_tracker()`
  (local import at `tracking.py:266` picks it up) — no shipping edit.
- **Offline.** `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`; weights are cached.
- **Promotion:** a challenger beats the champion only if it wins the crop frozen
  eval by > seedBand AND holds the foreign-reject floor AND passes a Codex leak-check
  AND is `demo_safe` end-to-end on both takes (no wrong_links, no lost links, roster {1..8}).
- **Deliver a recommendation, not a commit.** Output the recommended config as a
  described diff + numbers; do not modify shipping code. Bram applies it in the morning.
- Treat all crop-eval numbers as directional (tiny data). The end-to-end demo-safety
  result on the two takes is the decision-relevant signal for tomorrow.

## Deliverables (write here, continuously)

- `LEADERBOARD.md` — every experiment: lever, crop-eval metrics, end-to-end score, verdict.
- `FINDINGS.md` — what helped / what didn't; the recommended locked config with numbers;
  the flip-case and DINOv3 verdicts; honest caveats.
- `RECOMMENDATION.md` — the single config to run tomorrow + why + the exact change (as a diff).
- `runs/<exp>/` — per-experiment artifacts (traces, scores, sweeps).
