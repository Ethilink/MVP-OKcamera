# Champion — round 3 (sparse-representation-based classification / SRC)

**Family:** `sparse-representation-based classification (SRC): joint dictionary
reconstruction over concatenated candidate galleries via L1-penalized
multi-task (window-joint) least squares, open-set rejection via the Sparsity
Concentration Index (SCI)` (`TRIED.md` "Round 3, challenger 2").

Superseded the round-0 baseline (DINOv2-B CLS + moderate size-fusion +
top-3-mean + multi-frame voting) on 2026-07-15 after clearing a Codex
leak-check (`clean=true` — see `../TRIED.md` "Round 3 leak-check verdict" for
the full file:line audit). Round 0's record is kept in `../LEADERBOARD.md`,
marked superseded, not deleted.

Implements `build_gallery / score / accept` exactly as named in
[`linker-design.md` §6](../../../../model/docs/linker-design.md#6--match-rule--open-set-multi-candidate).
Standalone module — no dependency on the `frozen/`/`method/` sandbox packages,
drops straight into the linker build. Ported and cleaned up from
`../runs/r3-c2/method.py` (the research variant, which depended on `frozen/`
for its eval driver and duplicated the DINOv2 embedder/size-feature code
already living in this package): the ablation-switch parameters (`use_sci`,
`joint_window`) and the CV-sweep-only content-hash embedding cache were
dropped since the winning configuration is fixed (SCI on, joint window on)
and this module is not re-run inside a grid sweep; the DINOv2 embedder and
size-feature extraction now reuse `champion/embedder.py` /
`champion/size_features.py` instead of a second copy. Verified to reproduce
the research variant's guarded numbers bit-for-bit through the frozen eval
driver (CV re-ID `0.9333±0.0425`, reject `0.9733±0.0533`, 0 twin errors;
locked-holdout re-ID `1.00`, reject `0.8333`) before promotion.

## Hyperparameters (winners of the CV sweep)

| param | value | meaning |
|---|---|---|
| `alpha` | **0.0003** | `MultiTaskLasso` L1 penalty — controls how sparse the joint reconstruction code is (too low = dense/non-sparse dictionary, too high ≥0.002 collapses the code to all-zero, a degenerate operating point) |
| `size_alpha` | **0.5** | weight of z-scored log(area,length) fused onto the L2-normalized DINO embedding before it becomes a dictionary atom/query target (same cue, same constants as the round-0 baseline) |
| `tau` | **0.30** | accept threshold on `sim × SCI` (bounded in (0, 1] by construction) |
| `margin` | **0.02** | required best-vs-second-best gap (vacuous with 1 candidate — the one-missing prior) |
| `mask_dilate_px` | **2** | mask dilation before neutral-fill masking, same convention as the round-0 baseline |
| `max_iter` | **2000** | `MultiTaskLasso` solver iteration cap |
| window | **3 frames** (caller-side) | how many query crops make up one return-window; not an internal `ChampionMethod` param — the caller (linker / eval) decides how many crops to hand to `score()` |

Selected by a 630-point grid (`alpha × size_alpha × tau × margin` =
7×3×10×3) over the CV/selection pool, 5-seed leave-frame-group-out CV,
maximizing re-ID top-1 subject to the round-0 champion's reject floor
(`F=0.9467`, program.md §4 — "may only go up"). Full grid:
`../runs/r3-c2/cv_sweep.json`.

## Guarded held-out metrics

**CV (5-seed leave-frame-group-out, selection pool = 8 instruments × 12 frames, order 0..11):**

| metric | value |
|---|---|
| re-ID top-1 (mean ± seed std) | **0.9333 ± 0.0425** |
| foreign-reject rate (mean ± seed std) | **0.9733 ± 0.0533** |
| true-accept rate | 0.9333 |
| instrument1↔2 twin errors (summed, 5 seeds) | **0** |

vs. round-0 champion bar: re-ID `0.850 ± 0.043`, reject `0.9467` — beats re-ID
by `+0.083`, more than the round-0 seed-variance band (`0.043`); reject also
rises.

**Locked holdout (ONE-TIME report, never optimized against — last 3
frames/instrument as query, first 5 as gallery, last foreign source image as
negatives; tiny n, directional only):**

| metric | value | n |
|---|---|---|
| re-ID top-1 | **1.00** | 8 windows (8 instruments × 1 window each) |
| foreign-reject | **0.83** | 6 windows (5/6) |
| instrument1↔2 twin errors | 0 | — |
| false-reject rate (genuine) | 0.0 | — |
| wrong-link rate (genuine) | 0.0 | — |

Up from round 0's locked-holdout reject of `0.67` (4/6) on the same tiny
locked set. Full numbers: `../runs/r3-c2/holdout_report.json`.

## Ablations (CV pool, alpha/size_alpha/tau/margin held at the winning point)

| variant | re-ID top-1 | foreign-reject | twin errors |
|---|---|---|---|
| no size fusion (α=0) | 0.9083 ± 0.0312 | 0.7466 ± 0.0267 | **2** |
| no SCI reject rule (score = reconstruction quality alone) | **0.9583 ± 0.0373** | **0.6133 ± 0.1424** | 0 |
| independent per-frame sparse code (no joint window) | 0.9500 ± 0.0312 | 0.9200 ± 0.0266 | 0 |
| window=1 (no temporal window) | 0.9143 ± 0.0134 | 0.7500 ± 0.0689 | 0 |
| **full champion** (α_lasso=0.0003, size_alpha=0.5, joint window=3, SCI on) | 0.9333 ± 0.0425 | **0.9733 ± 0.0533** | 0 |

Reading:
- **Size fusion still resolves the twin pair, even under a completely
  different matching paradigm.** Without it, CV twin errors reappear (2) and
  reject drops ~23pp — the same instrument1↔2 near-twin failure mode the
  round-0 champion found, now confirmed independent of the matching
  mechanism (cosine top-K vs. sparse reconstruction).
- **SCI is this family's open-set mechanism, and it costs some re-ID.**
  Dropping SCI (scoring on reconstruction quality alone) actually raises
  re-ID slightly (0.9583 vs 0.9333) but collapses reject to 0.6133 — the
  Lasso solver happily "explains" a foreign object's signal well enough to
  fool a residual-only score. SCI (how concentrated the coefficient mass is
  on one candidate) is what catches this, at a ~2.5pp re-ID cost. Mirrors
  round 0's own "voting is the real open-set fix" finding, but here the
  fix is SCI.
- **The joint (window-shared) sparse code beats independent per-frame
  solves, specifically on rejection.** Solving each frame's sparse code
  independently and averaging the scores gets re-ID 0.95 (comparable) but
  reject only 0.92 vs the joint code's 0.9733 — confirms the family's
  central hypothesis (a genuine window should be *jointly* reconstructible
  from one identity's atoms) is doing real work, not just averaging
  independent per-frame decisions.
- **The multi-frame window itself still matters inside this paradigm.**
  window=1 gets reject down to 0.75 — consistent with round 0's finding that
  a single confident frame is easier to fool than a window.

## Leak-check

Cleared a Codex leak-check before promotion (2026-07-15, `clean=true`) — see
`../TRIED.md` "Round 3 leak-check verdict" for the full file:line audit:
harness/eval/split/loader untouched (SHA-256 of the frozen evaluator matches
prior round-3 runs); zero gallery/query item or content-hash overlap across
all 5 CV seeds and the locked split; holdout scored exactly once, after every
hyperparameter was fixed from CV; `score()`/`accept()` never read a query
label, only the offered `galleries` dict and the solved reconstruction
coefficients.

## Known limitations / scope of this round

- Multi-frame voting/joint-window solving is evaluated on **windowed chunks
  of the static per-instrument crops**, not on real video return-events —
  same caveat round 0 recorded (`frozen/loader.py` exposes
  `testing_video_ids()` read-only for a future harness/T04 use).
- The `other_objects` negatives are still stage-1's adversarially tool-like
  set (60 crops) — the reject numbers above are a lower bound against easy
  negatives, same caveat as round 0.
- All numbers are **directional** (8×15 crops, 60 foreign crops) — confirmed
  later at stage-2 (T07 photos + real room), per program.md §8.
- `MultiTaskLasso`'s solve cost scales with the number of Missing candidates
  (dictionary width = sum of all candidate gallery sizes) — not yet profiled
  against the app's real-time budget at N=8 roster size; flagged for the T04
  harness/fps work, not benchmarked in this round.
