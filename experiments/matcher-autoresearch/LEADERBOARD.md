# LEADERBOARD — matcher-autoresearch

Ranked techniques with guarded held-out metrics. **Selection score** = re-ID
top-1 (CV, 5-seed leave-frame-group-out) subject to a hard floor on
foreign-reject (program.md §4). A challenger only becomes champion if it beats
the current champion's re-ID top-1 by **more than the seed-variance band**
(`seedBand` below) while the reject floor `F` holds or rises. All numbers are
**directional** (8 instruments × 15 crops, 60 foreign crops) — confirmed later
at stage-2.

Current reject floor **F = 0.973** (round-3 champion's CV foreign-reject mean —
future challengers must clear this or better).

| round | technique | family | re-ID top-1 (CV) | seedBand (±std) | foreign-reject (CV) | true-accept | twin errors (1↔2) | locked holdout re-ID | locked holdout reject | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | **DINOv2-B + masked-crop + moderate size-fusion + top3-mean + multi-frame voting** (champion baseline, program.md §5) | baseline: multi-cue fusion + temporal voting | 0.850 | 0.043 | 0.947 | 0.850 | **0** | 1.00 (n=8) | 0.667 (n=6) | superseded (round 3, 2026-07-15) |
| 3 | **Sparse-representation-based classification (SRC): joint dictionary reconstruction + Sparsity Concentration Index** (challenger 2, `runs/r3-c2/`) | sparse coding: dictionary-reconstruction classification (SCI open-set rejection) | **0.9333** | 0.0425 | **0.9733** | 0.9333 | **0** | 1.00 (n=8) | 0.83 (n=6) | **CHAMPION** (2026-07-15) |

## Round 0 detail

- **Hyperparameters:** `size_alpha=0.5, top_k=3, tau=0.55, margin=0.03, mask_dilate_px=2, window=3`. Selected via a 24-point CV grid sweep (`size_alpha × tau × margin`) — full grid in `runs/baseline/cv_sweep.json`.
- **Ablations** (see `champion/PARAMS.md` for the full table + reading):
  - no size fusion (α=0): re-ID 0.742±0.049, reject 0.960, **twin errors 1** — size fusion is what resolves the instrument1↔2 pair.
  - no multi-frame voting (window=1): re-ID **0.893±0.030** (higher!), reject **0.596** (much lower) — voting trades a little raw top-1 for the open-set win; confirms stage-1's "voting is the real open-set fix."
  - full champion (window=3): re-ID 0.850±0.043, reject 0.947 — the selected operating point.
  - A stricter alternative exists at τ=0.65: reject **1.00**, re-ID 0.80 — not selected (favoured re-ID this round), flagged as a safety-first fallback.
- **Locked holdout** (one-time, never optimized against — last 3 frames/instrument as query, first 5 as gallery, held-out foreign source image): re-ID **1.00** (8/8, tiny n), foreign-reject **0.667** (4/6, tiny n) — 2 false-accepts both went to `instrument7` (score 0.60–0.68), not stage-1's `instrument8` magnet. See `champion/PARAMS.md` "instrument8 / instrument1↔2 verdicts."
- **Artifact:** `runs/baseline/` (sweep, ablations, holdout report, logs). No longer the contents of `champion/` — see "Round 3 detail" below.
- **cheatRisk:** false — no edits to `frozen/`, no query-label reads (structurally prevented by `frozen/eval.py` keeping ground truth eval-side only), holdout touched exactly once after hyperparameters were fixed from CV.
- **Status:** superseded 2026-07-15 by round 3's SRC challenger (below) — beat this baseline's re-ID by +0.083 (more than this round's own 0.043 seed band) while also raising foreign-reject. Kept here as history per program.md §5 (`TRIED.md`'s idea log is never pruned, and neither is this row).

## Round 3 detail — CHAMPION (promoted 2026-07-15)

Sparse-representation-based classification (SRC): a query return-window's
embeddings are jointly reconstructed as a sparse linear combination of every
Missing candidate's gallery atoms (one L1-penalized `MultiTaskLasso` solve
per window, atoms from all candidates concatenated into a single
dictionary), scored by reconstruction quality × the Sparsity Concentration
Index (SCI, Wright et al. 2009) — how much of the solved code's mass sits on
one candidate vs. spread across all of them. Full family boundary argument
(why this isn't a repeat of any earlier round's family) is in `TRIED.md`
"Round 3, challenger 2."

- **Hyperparameters:** `alpha=0.0003 (MultiTaskLasso L1 penalty), size_alpha=0.5, tau=0.30, margin=0.02, mask_dilate_px=2, max_iter=2000, window=3`. Selected via a 630-point CV grid (`alpha × size_alpha × tau × margin`) over the CV/selection pool — full grid in `runs/r3-c2/cv_sweep.json`.
- **Ablations** (see `champion/PARAMS.md` for the full table + reading):
  - no size fusion (α=0): re-ID 0.9083±0.0312, reject 0.7466, **twin errors 2** — size fusion still resolves the instrument1↔2 pair under this matching paradigm too.
  - no SCI reject rule (score = reconstruction quality alone): re-ID **0.9583±0.0373** (higher!), reject **0.6133** (much lower) — SCI is this family's open-set mechanism, at a ~2.5pp re-ID cost. Mirrors round 0's "voting is the real open-set fix," but the fix here is SCI.
  - independent per-frame sparse code (no joint window): re-ID 0.9500±0.0312, reject 0.9200 — close on re-ID but the **joint** window-shared code is markedly better on reject (0.9733 vs 0.9200), confirming the family's central "joint sparse representation" hypothesis is doing real work.
  - window=1 (no temporal window): re-ID 0.9143±0.0134, reject 0.7500 — the multi-frame window still matters inside this paradigm.
  - full champion (window=3, joint+SCI): re-ID 0.9333±0.0425, reject **0.9733** — the selected operating point.
- **Locked holdout** (one-time, never optimized against — last 3 frames/instrument as query, first 5 as gallery, held-out foreign source image): re-ID **1.00** (8/8, tiny n), foreign-reject **0.83** (5/6, tiny n) — up from round 0's 0.67 (4/6) on the same locked set; 0 twin errors, 0 false-reject, 0 wrong-link.
- **Leak-check:** cleared 2026-07-15, `clean=true` — full file:line audit in `TRIED.md` "Round 3 leak-check verdict" (harness/loader untouched, zero gallery/query overlap across all 5 seeds + the locked split, holdout scored exactly once after hyperparameters were fixed, no query-label reads).
- **Artifact:** `champion/` (clean standalone module, ported from `runs/r3-c2/method.py` — no dependency on `frozen/`/`method/`; verified to reproduce the research variant's numbers bit-for-bit through the frozen eval driver before promotion) + `runs/r3-c2/` (sweep, ablations, holdout report, logs).
- **cheatRisk:** false — see leak-check above.

## Next round's floor (for challengers)

- `foreign_reject_mean (CV) >= 0.9733` to be eligible for promotion at all.
- Must beat `reid_top1_mean (CV) > 0.9333 + seedBand` (i.e. > ~0.9758) to promote.
- See `TRIED.md` for every family already covered (don't repeat: multi-cue
  fusion (embedding+size), top-K gallery aggregation, temporal/window voting,
  dense local patch-correspondence, optimal-transport correspondence,
  native-scale retinotopic matching, combinatorial set-to-set assignment,
  UBM-style cohort log-likelihood-ratio scoring, and round 3's own sparse-
  coding/dictionary-reconstruction family) — rotate to a genuinely different
  family per program.md §6.
