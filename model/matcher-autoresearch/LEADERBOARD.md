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
| 3 (e2e-ar) | **Error-correcting output-code (ECOC) class-decomposition: exhaustive Dietterich-Bakiri code + shrinkage-Fisher dichotomizers + loss-based decoding** (challenger 1, `runs/r3-c1/`) | output coding: class-decomposition + error-correcting decoding | 0.9167 | 0.0263 | 0.9733 | 0.9167 | **0** | not called (non-winner) | not called (non-winner) | not promoted — honest loss (nested honest est. 0.8833) |
| 4 (e2e-ar) | **Social-choice rank aggregation: per-frame×block×agg VOTERS cast ordinal ballots over candidates + an explicit none-of-the-above alternative, decided by the Schulze beat-path Condorcet method** (challenger 1, `runs/r4-c1/`) | social choice: Condorcet/Schulze ordinal rank aggregation | 0.9417 | 0.0333 | 0.9867 | 0.9417 | **0** | not called (non-winner) | not called (non-winner) | not promoted — inside champion seed band (bar 0.9758; nested honest 0.9417) |
| 6 (e2e-ar) | **Tangent-distance one-shot invariant matching (Simard 1993): ridge-regularised point/plane-to-plane projection onto per-atom finite-difference flip/rotate/scale tangent planes** (challenger 1, `runs/r6-c1/`) | tangent distance: transformation-manifold linear-invariance projection | 0.9750 (nested honest; flat argmax 0.9833) | 0.0204 | 0.9867 (safe pt) | 0.9750 | **0** | not called (non-winner) | not called (non-winner) | not promoted — nested honest 0.9750 is 0.0008 UNDER bar 0.9758 (+0.042 over champion point); crop win is two-sided-only & non-injectable; e2e query-only variant not demo-safe (3 links lost, 0 wrong) |

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

## Round 3 (2026-07-16 e2e-autoresearch) detail — challenger 1, NOT PROMOTED

Error-correcting output codes: the K-way problem is decomposed into 127 binary
dichotomies **of the candidate set** (exhaustive Dietterich-Bakiri code, minimum
Hamming distance 64), each solved by its own shrinkage-Fisher discriminant fit on
gallery atoms, decoded by exponential loss against each identity's codeword. Full
family-boundary argument (why this is not r7-c0's hyperplane-*representation*
code) is in `TRIED.md` "Round 3 (2026-07-16 e2e-autoresearch), challenger 1".

- **Frozen CV:** re-ID **0.9167 ± 0.0263**, reject **0.9733** (exact floor tie), **0** twin errors. Champion 0.9333; bar 0.9758. **Nested leave-one-seed-out honest estimate 0.8833 ± 0.0486** — the flat argmax over 75,980 feasible rows is ~3.3pp optimistic.
- **Collapse theorem (transferable):** ECOC with any *linear-in-targets* dichotomizer (ridge/kernel-ridge/least-squares/nearest-centroid) is **provably identical to one-vs-rest ridge** — the codeword is `Mᵀ C`, and for the exhaustive code the decoding partition function factorizes and cancels. Verified numerically (`runs/r3-c1/smoke.py`). That whole sub-branch is closed analytically; don't spend a round on it.
- **Ablations** (each at its own best feasible point, via the real frozen evaluator): `loss=linear` (≡ OvR ridge) 0.7250; `lam=10` (→ nearest-centroid dichotomizer) 0.6667; `code=onevsrest` (8 columns) 0.8083; `loss=hamming` 0.8917; `agg=code` (window-joint codeword) 0.8167; `size_alpha=0` 0.7750 with **1 twin error**.
- **The code is real (+0.11):** one-vs-rest 0.8083 → long code 0.9167 with identical dichotomizers/decoder. The family's hypothesis is directionally confirmed; it just lands below SRC.
- **A "sweet spot" was found and then killed by its own check:** `random32 @ code_seed=0` hit 0.9333 at reject 0.9867. Over 10 code draws random32 is 0.8792 ± 0.0388 — 0.9333 was the **max**, and the draw spread (±0.039) ≈ the champion's whole seed band. **Any future round reporting a random-code number must average over code draws.**
- **Window-joint decoding HURTS here — opposite of SRC.** SRC won *because* of its joint window code; ECOC prefers per-frame loss averaging (0.9167) to a window-joint mean codeword (0.8167). Pool evidence in loss space, not code space.
- **Why it loses, structurally:** 5 atoms per identity means every dichotomizer is fit on the same 40 points, so column errors are strongly *correlated* — the code cannot correct what every column gets wrong together. SRC sidesteps this by fitting no discriminant at all. The binding constraint is the enrolment budget, not the decision rule.
- **End-to-end gate:** not run (reserved for a crop-eval winner).
- **Locked holdout:** deliberately not called — reserved for a champion report; spending the one-time split on a clear non-winner would erode it for no information.
- **Artifact:** `runs/r3-c1/` (method, sweep, nested selection, code-length + code-seed ablations, frozen evidence package, leak check, `RESULTS.md`).
- **cheatRisk:** false — `runs/r3-c1/leak_check.json` `clean: true` (10 mechanical checks); frozen `*.py` sha256 identical before/after; no shipping/`frozen/` edits; no threshold lowered.


## Round 4 (2026-07-16 e2e-autoresearch) detail — challenger 1, NOT PROMOTED

Social-choice-theoretic rank aggregation: each (query frame × feature block ×
gallery-aggregation rule) is an independent **voter** that casts only an
**ordinal ballot** over the offered candidates plus an explicit
**none-of-the-above** alternative (its ballot position fixed by that voter's
gallery-internal leave-one-view-out self-similarity). The winner is the strict
**Schulze beat-path** Condorcet winner of the pairwise majority-margin graph;
open-set rejection is the Condorcet outcome "nothing beats none-of-the-above,"
not a threshold on any similarity. Full family-boundary argument (why this is
not round 0's continuous window-voting, r1-c0's dissimilarity profiles, or any
cardinal-score family) is in `TRIED.md` "Round 4, challenger 1".

- **Frozen CV (locked config `size_alpha=0.25, top_k=2, reject_q=0.0, reject_mode=global, dom_req=0.35`):** re-ID **0.9417 ± 0.0333**, reject **0.9867**, **0** twin errors, **0** wrong-links every seed. Champion 0.9333; bar 0.9758. **Nested leave-one-seed-out honest estimate 0.9417** (re-ID) / **0.96** (reject) — the honest reject falls *below* the 0.9733 floor, i.e. the floor-clearing operating point does not transfer to an untuned seed.
- **Verdict:** clears the champion's point estimate (+0.0084) but sits well inside its 0.0425 seed band and far below the seed-band promotion bar (0.9758). NOT a win. Ordinal ranking ceiling (reject gate off) is **0.9917** (119/120) — the representation orders identities almost perfectly; the entire loss is the open-set gate, which sits on a cliff (dom_req 0.2 → reject 0.80; dom_req 0.5 → reject 1.0).
- **Ablations (each at its own best feasible point, real frozen evaluator):**
  - `rule=borda` (same ballots, Borda count) 0.9083 / reject 1.0; `rule=plurality` 0.9250 / 0.9733 — **the Schulze/Condorcet machinery is load-bearing**: +0.033 over Borda, +0.017 over plurality at the same ballots. This is the one finding that distinguishes the family from a plain rank-average.
  - `reject_mode=per_candidate` (the intransitive-voter generalization) **0.8583** — identity-adaptive ballot placement *hurts* badly; `global` (pooled roster self-similarity) is decisively better. The exotic intransitive-preference angle is a dead end.
  - no none-of-the-above alternative (rivals-only, reject via dominance margin only) 0.9333 / 0.9733 — the explicit REJECT alternative adds a little reject headroom (0.9867 vs 0.9733) but is not essential.
  - `blocks=cls` (no size cue) **0.75**, reject 0.9733, **1 twin error** — size fusion still resolves instrument1↔2, consistent with every prior family.
  - `blocks=fused only` 0.9417 / 0.9733 and `aggs=topk only` 0.9417 / 0.9867 — voter *diversity* across blocks/aggs is not needed for re-ID; it only buys a little reject headroom. The electorate can be small.
  - `window=1` (single-frame electorate) **0.6857** — the multi-frame electorate is essential; a Condorcet election needs enough voters.
- **End-to-end gate:** not run (reserved for a crop-eval winner — this is not one).
- **Locked holdout:** deliberately not called — reserved for a champion report; spending the one-time split on a clear non-winner would erode it for no information (same discipline as r2-c1 / r3-c1).
- **Artifact:** `runs/r4-c1/` (`social_choice.py`, `method.py`, sweeps, nested selection, structural ablations, frozen evidence package, leak check, `RESULTS.md`).
- **cheatRisk:** false — `runs/r4-c1/leak_check.json` `clean: true`; frozen `*.py` sha256 identical before/after and matching the prior round; identity-blind scores (candidate relabelling leaves decisions bit-identical); locked holdout never touched in selection; no threshold lowered; no shipping/`frozen/` edits.

## Round 5 (2026-07-16 e2e-autoresearch) detail — challenger 1, NOT PROMOTED

k-reciprocal re-ranking (Zhong et al. 2017, "Re-ranking Person Re-identification
with k-reciprocal Encoding") used as the WHOLE matcher: the probe→atom distance
is the **Jaccard distance between the k-reciprocal nearest-neighbour encodings**
of the query probe and each candidate gallery atom over a pooled context,
blended with the original cosine distance; per-candidate score = top-k mean of
`(1 − d)`; open-set rejection is the collapse of reciprocal-neighbour overlap for
outliers, gated by the champion's own tau+margin `accept`. Representation is
byte-identical to the champion (masked DINOv2-B CLS + z-scored size cue), so the
delta is the matching family alone. Genuinely new (grep: `Jaccard`→0; TRIED.md's
own menu lists k-reciprocal re-ranking as *not yet tried*); distinct from r4-c0
graph diffusion (no random walk / no absorbing sink / no propagated mass — a
closed-form set overlap). Full boundary argument in `TRIED.md` "Round 5,
challenger 1".

- **Frozen CV (locked config `k1=4, k2=2, lam=0.3, size_alpha=0.5, top_k=3, win_mode=pool, tau=0.675, margin=0.0`):** re-ID **0.8833 ± 0.0612**, reject **0.9733 ± 0.0327**, **0** twin errors. Champion 0.9333; bar 0.9758.
- **Verdict:** clears the reject floor but does NOT beat the champion — 0.8833 is *below* the champion point estimate and far below the seed-band bar. `PROMOTED=False`.
- **The diagnostic trade-off:** raw ranking power is excellent — reject gate OFF gives top-1 **0.9833** (k1=14, k2=2, lam=0.5, win=pool), tying the best raw ranking any family has produced. But the knobs pull opposite ways: raw re-ID peaks at LARGE k1 (14) while open-set reject needs SMALL k1 (4) — only tight, mutually-agreed neighbourhoods deny a foreign outlier shared reciprocal mass. Every feasible operating point (reject ≥ floor) sits at k1=4 and tops out at 0.8833, a ~10pp collapse from raw. The reciprocity signal that rejects foreigners also rejects the harder genuine returns (flipped/low-overlap views). Same shape (strong ranking, weak reject-at-floor) that sank most prior families; SRC's algebraic SCI reject separates foreign from hard-genuine better at a fixed operating point.
- **End-to-end gate:** not run (reserved for a crop-eval winner — this is not one).
- **Locked holdout:** deliberately not called (reserved for a champion report; same discipline as r2-c1/r3-c1/r4-c1).
- **Artifact:** `runs/r5-c1/` (`method.py`, `sweep.py`/`cv_sweep.json`, `run_eval.py`/`cv_report.json`, leak check, `RESULTS.md`).
- **cheatRisk:** false — `runs/r5-c1/leak_check.json` `clean: true`; frozen `*.py` sha256 identical before/after; identity-blind scores (candidate relabelling leaves decisions bit-identical); locked holdout never touched in selection; no threshold lowered; no shipping/`frozen/` edits.

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
