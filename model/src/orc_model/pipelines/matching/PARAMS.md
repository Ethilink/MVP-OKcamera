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

## Round 3b — re-sweep after adding `aspect_ratio` (2026-07-17)

`app/mvp.toml` enabled a 3rd fused feature, `aspect_ratio` (see
`size_features.py`, calibrated by `calibrate_shape_features.py`). Adding a
feature changes the fused vector's geometry, so `alpha`/`size_alpha`/`tau`/
`margin` — tuned above against the 2-feature vector only — needed
re-selection, same "may only go up [from the prior champion's own CV floor]"
discipline as every other round (program.md §4).

**Method:** `matcher-autoresearch/sweep_hyperparams.py`. Same frozen harness
(CV/selection pool, 5-seed leave-frame-group-out split, `window=3`) as round
3's own sweep, decoupled for tractability: `alpha`/`size_alpha` require a
real Lasso re-solve (7×4=28 points, DINO embeddings memoized across the
whole sweep since embedding doesn't depend on any of these 4 hyperparams);
`tau`/`margin` (10×5=50 points) are pure post-hoc thresholding of the
already-computed scores, reimplementing `accept()`'s decision rule exactly —
1400 threshold-grid points total from 28 actual re-solves. Selection rule
unchanged: maximize CV re-ID top-1 subject to `foreign_reject >= 0.9733`
(round 3's own guarded floor).

**Sanity check:** re-running the identical harness at the 2-feature
composition (`log_area`, `log_length` only) over this round's own (wider)
grid reproduced round 3's guarded CV numbers exactly — `reid_top1
0.9333±0.0425`, `foreign_reject 0.9733±0.0533` — confirming the
re-implementation is faithful before trusting its verdict on the 3-feature
composition.

| param | round 3 (2 features) | round 3b (3 features) |
|---|---|---|
| `alpha` | 0.0003 | **0.0003** (unchanged) |
| `size_alpha` | 0.5 | **0.25** (halved — a 3rd shape feature means the same total shape-cue weight needs splitting further, or it starts to overweight the shape block vs. DINO appearance) |
| `tau` | 0.30 | **0.30** (unchanged) |
| `margin` | 0.02 | **0.02** — genuinely untied at any grid value 0.01–0.08 tested (never the binding constraint in this CV data); kept at round 3's value rather than the arbitrary lowest tie-break |

**CV (5-seed leave-frame-group-out, same pool as round 3):**

| metric | round 3 (2 features) | round 3b (3 features) |
|---|---|---|
| re-ID top-1 | 0.9333 ± 0.0425 | **0.9500 ± 0.0312** |
| foreign-reject | 0.9733 ± 0.0533 | 0.9733 ± 0.0533 (unchanged) |
| twin errors (1↔2) | 0 | 0 |

+1.67pp re-ID at an identical reject floor and a tighter seed-std, from
`aspect_ratio` alone.

**Locked holdout (ONE-TIME report, same tiny n=8/6 windows as round 3):**

| metric | round 3 (2 features) | round 3b (3 features) |
|---|---|---|
| re-ID top-1 | 1.00 (8/8) | **0.875 (7/8)** — missed `instrument1` (false-reject, not a wrong-link) |
| foreign-reject | 0.83 (5/6) | 0.83 (5/6) (unchanged) |

**Reading:** the CV improvement doesn't carry over to the locked holdout —
at n=8, one window is 12.5pp, so this single miss is noise-scale, not a
contradiction of the CV result, but it's not confirmation either. Same
"directional, tiny-n" caveat as every number in this file (program.md §8) —
worth a real-photo check (T07/stage-2) before fully trusting the CV gain in
a live recording, same as round 3 itself was never fully confirmed beyond
this dataset.

**Deployed:** `app/mvp.toml` updated to `size_alpha = 0.25` (the only
changed value). `alpha`, `acceptance_threshold`, `winner_margin` already
matched the new optimum, so no other `mvp.toml` change was needed.

## Round 3c — solidity/extent/width_profile tried, swept, NOT PROMOTED (2026-07-17)

`app/mvp.toml` briefly swapped `aspect_ratio` for `solidity`, `extent`, and
`width_profile` (all pre-calibrated, `size_features.py`). Per this file's own
discipline, the swap was re-swept rather than deployed on the old
hyperparameters — same tooling as round 3b, `matcher-autoresearch/
sweep_hyperparams.py` (`ACTIVE_FEATURES` updated to the new 5-feature, 11-dim
composition; grid: `alpha`×`size_alpha` = 7×4 real Lasso re-solves, `tau`×
`margin` = 10×5 free thresholding, same CV/selection pool and 5-seed
leave-frame-group-out split as every prior round).

**Result — CV, full new composition:** re-ID top-1 **0.8083 ± 0.0500** at the
best feasible point (`alpha=0.0003, size_alpha=0.25, tau=0.30, margin=0.01`),
foreign-reject **0.9733 ± 0.0533** (exact floor tie, the same "fragile tie"
pattern flagged throughout the original research), twin errors 0. **A ~14pp
re-ID regression from round 3b's 0.9500** — does not clear "may only go up"
(program.md §4); **NOT PROMOTED**.

**Per-feature ablation (isolating the cause, `alpha=0.0003` fixed, best
`size_alpha`/`tau`/`margin` per composition, same harness):**

| composition | best `size_alpha` | re-ID | reject | twins |
|---|---|---|---|---|
| champion (`log_area,log_length,aspect_ratio`) | 0.25 | **0.9500** | 0.9733 | 0 |
| baseline (`log_area,log_length` only) | 0.50 | 0.9333 | 0.9733 | 0 |
| `+solidity` only | 0.50 | 0.9333 | 0.9733 | 0 |
| `+extent` only | 0.50 | 0.9167 | 0.9733 | 0 |
| `+width_profile` only | 0.10 | **0.8750** | 0.9733 | 0 |
| `+solidity+extent` (no width_profile) | 0.50 | 0.9167 | 0.9733 | 0 |
| full new (`solidity,extent,width_profile`) | 0.25 | 0.8083 | 0.9733 | 0 |

**Reading:** `width_profile` (7 raw width-at-major-axis-bin dimensions) is the
dominant cause — it alone drags re-ID down to 0.875, worse than dropping
`aspect_ratio` entirely with no replacement (0.9333). Plausible mechanism, not
independently confirmed further given the time-box: `width_profile` samples
mask width at 7 fixed points along the PCA major axis, so small
segmentation-boundary jitter shifts which pixels land in which bin —
the same noise-sensitivity failure mode round 4's continuous
hole-persistence metric hit (`TRIED.md` "Round 4, challenger 2" —
`w_holes=0.0` in every top Stage-1 config, its wall-thickness value backwards
on inspection). `solidity` alone is roughly neutral vs. the no-features
baseline (0.9333 either way); `extent` alone costs ~1.7pp. Neither is
disqualifying on its own, but neither beats `aspect_ratio` either, and none of
the combinations tested recovers champion's 0.9500 — this specific swap adds
no value over what's already deployed.

**Deployed:** reverted `app/mvp.toml` to the round-3b composition
(`aspect_ratio`, `size_alpha=0.25`) — the only composition of the four tested
here that clears its own reject floor without a re-ID cost vs. round 3b.
`solidity`/`extent`/`width_profile` remain in `size_features.py`'s
`SHAPE_FEATURES` registry (selectable, calibrated) for any future attempt,
but are not part of the live fusion. A future round could still try `solidity`
alone alongside `aspect_ratio` (additive, not a swap) or `eccentricity`/
`hu_moments` (untested by this round) — not built here.

## Round 3d — eccentricity/hu_moments tried, swept, NOT PROMOTED (2026-07-17)

Added `eccentricity` and `hu_moments` (both pre-calibrated) on top of round
3b's composition (`log_area, log_length, aspect_ratio`), additively rather
than as a swap this time. Same tooling (`sweep_hyperparams.py`, updated to
the new 5-feature, 11-dim composition — `hu_moments` alone is 7 dims), same
harness, same widened `size_alpha` grid (`0.1, 0.15, 0.25, 0.5, 0.75, 1.0`,
extended down from round 3c after its optimum landed on the old grid's own
boundary).

**Result — CV, full new composition:** re-ID top-1 **0.8667 ± 0.0312** at the
best feasible point (`alpha=0.0002, size_alpha=0.25, tau=0.25, margin=0.01`),
foreign-reject **0.9733 ± 0.0533** (floor tie), twin errors **1**. A ~8pp
re-ID regression from round 3b's 0.9500, AND a twin error reappears — does
not clear "may only go up" (program.md §4); **NOT PROMOTED**.

**Per-feature ablation (same method as round 3c — isolating alpha too this
time, since the full sweep's optimum moved to `alpha=0.0002`; best
`alpha`/`size_alpha`/`tau`/`margin` per composition, same harness):**

| composition | best alpha / size_alpha | re-ID | reject | twins |
|---|---|---|---|---|
| champion (`log_area,log_length,aspect_ratio`) | 0.0003 / 0.25 | **0.9500** | 0.9733 | 0 |
| baseline (`log_area,log_length` only) | 0.0002 / 0.75 | 0.9333 | 0.9733 | 0 |
| `+eccentricity` (no `aspect_ratio`) | 0.0003 / 0.50 | 0.8750 | 0.9733 | **1** |
| `+hu_moments` (no `aspect_ratio`) | 0.0002 / 0.25 | 0.8583 | 0.9867 | 0 |
| champion `+eccentricity` only | 0.0003 / 0.25 | 0.9167 | 0.9733 | **1** |
| champion `+hu_moments` only | 0.0002 / 0.25 | 0.8833 | 0.9733 | 0 |
| full new (champion `+eccentricity+hu_moments`) | 0.0002 / 0.25 | 0.8667 | 0.9733 | **1** |

**Reading:** both features regress re-ID individually, and `eccentricity`
specifically reintroduces the `instrument1↔2` twin error even when added
*alongside* `aspect_ratio` (which alone gets 0 twin errors) — a real
interaction, not just dilution. The likely mechanism: `eccentricity` and
`hu_moments` are both explicitly rotation/scale/translation-invariant
descriptors by construction (`size_features.py` docstrings) — they carry
*no* absolute-scale signal, which is the one thing that resolves the
2×-scaled twin pair. Adding scale-invariant noise into the fused vector's
z-scored block dilutes `aspect_ratio`'s own (also scale-invariant, but
apparently more twin-relevant) contribution and the Lasso solve's atom
selection, at a real re-ID cost, without buying anything back — the
data confirms this rather than assumes it: `hu_moments` alone (no
`aspect_ratio`) does raise foreign-reject (`0.9867` vs `0.9733`), so it isn't
useless, just not a net win here at any tested operating point.

**Deployed:** reverted `app/mvp.toml` to the round-3b composition
(`aspect_ratio` only, `size_alpha=0.25`) — unchanged from before this round.
Of all seven `SHAPE_FEATURES` registry entries, only `log_area`, `log_length`,
and `aspect_ratio` have now shown a real CV benefit under SRC; `solidity`,
`extent`, `width_profile`, `eccentricity`, and `hu_moments` all regress it,
individually or combined, across rounds 3c-3d. Any future attempt should
treat this as a real, ablation-confirmed pattern for this method — not
assume the next untested feature will fare differently.
