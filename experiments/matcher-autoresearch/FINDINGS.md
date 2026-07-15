# FINDINGS — matcher-autoresearch

**Status:** rewritten from `LEADERBOARD.md`, the **full** `TRIED.md` (every
round-0-through-9 entry plus the post-round-9 follow-up dispatches, and all
three leak-check verdicts), and `champion/PARAMS.md` + `champion/champion.py`,
per `program.md` §9. **This supersedes and corrects an earlier partial version
of this file** whose own header said "rounds 0–3" but which had in fact gone
stale even inside that window: it still described the **round-0 baseline** as
champion and gave the baseline's hyperparameters as "the champion's," even
though round 3's SRC challenger had already been promoted (2026-07-15) and is
what `LEADERBOARD.md` and `champion/` actually ship. That error is fixed
throughout this version, disclosed here rather than silently patched, per this
project's own "report the mistake, not just the fix" convention. **Every
number below is DIRECTIONAL** — tiny set (8 instruments × 15 crops, 60 foreign
crops), confirmed later at stage-2 (T07 photos + real room). This is the
T02-stage-2 (partial) output that reconciles into `model/docs/linker-design.md`.

---

## 0 · Bottom line

The champion is **round 3's sparse-representation classification (SRC)**
challenger — joint dictionary reconstruction over concatenated candidate
galleries via an L1-penalized multi-task (window-joint) least-squares solve,
open-set rejection via the Sparsity Concentration Index (SCI) — **not** the
round-0 multi-cue-fusion baseline. SRC beat the baseline on both guarded axes
(CV re-ID `0.9333` vs `0.850`, reject `0.9733` vs `0.9467`), cleared a Codex
leak-check (`clean=true`), and was promoted 2026-07-15. It is a **genuinely
different method family**, not a re-tuning of the baseline's knobs — per
`program.md` §9(b) this means SRC **replaces** `linker-design.md` §6's match
rule (nearest-view-max cosine + Hungarian assignment) rather than merely
filling in its open parameters. `linker-design.md` §6 as currently written
(checked 2026-07-15) still describes the old crop-to-crop cosine rule; it
needs to be **rewritten**, not parameterized, to land SRC's design — see §2.

**Six more rounds (4–9), plus a handful of post-round-9 follow-up dispatches
into earlier slots — roughly 30 additional challenger experiments across
~25 distinct method families** (energy-based associative memory, leaky
absorbing graph diffusion, distributionally-robust spherical-cap
certification, counterfactual intervention-response fingerprints, random-
partition tree ensembles, reservoir computing, classical RANSAC keypoint
matching, geometric part-decomposition, SO(2) group-orbit correlation,
Gaussian-process kernel regression, Dirichlet-process von Mises-Fisher
clustering, Beta-Bernoulli hyperplane codes, OpenMax/Weibull extreme-value
theory, split-conformal calibration, rough-path signature kernels,
Grassmannian quantum-fidelity subspace matching, kernel MMD two-sample
testing, Dempster-Shafer evidence theory, medial-axis heat-kernel
diffusion-wavelet scattering, hyperbolic Poincaré-ball embedding, fuzzy
Choquet-integral fusion, Kalman-filter trajectory matching, topological
persistent homology, second-order SPD-manifold geometry, and more) — **failed
to unseat SRC.** That six more rounds of genuinely diverse challengers, many
built from scratch with zero shared code with SRC or the baseline, could not
beat it strengthens confidence that SRC's specific recipe (a convex sparse
solve + an algebraic concentration statistic) sits in a real, favourable spot
for this 8×15 dataset — not that round 3 got lucky.

**Two challengers, on paper, beat SRC's own numbers and were never cleanly
promoted for procedural (not correctness) reasons:**

- **Round 6's SO(2) rotation-orbit group-correlation** challenger beat both of
  SRC's raw point estimates (CV re-ID `0.9666` vs `0.9333`, reject `1.0000` vs
  `0.9733`) but **failed its Codex leak-check** for a sequencing problem: the
  locked holdout had already been opened and logged for one configuration
  before a materially different, stronger configuration was fixed — a
  post-holdout-disclosure leak, not a data leak (§6).
- **Round 9's hyperbolic (Poincaré-ball) embedding** challenger is the
  **single strongest raw result of the entire run**: CV re-ID `0.9833` (which
  clears not just SRC's raw `0.9333` but the strict promotion bar
  `> 0.9333 + 0.0425 ≈ 0.9758`) and reject `0.9867` (clears the floor
  `0.9733` with real margin) — the *only* challenger across all 9 rounds to
  clear `program.md` §4's strictest promotion test on both axes at once. Its
  own writeup flags it for the standard Codex leak-check "before any actual
  promotion," the same deferral every other "BEATS" verdict in this log uses
  — **but no round-9 leak-check verdict exists anywhere in `TRIED.md`, and no
  later entry revisits it.** The run's post-round-9 activity went back and
  backfilled earlier under-filled challenger slots (round 4/5 follow-ups)
  instead. This is **the single most important fact rounds 4–9 add**: on
  paper the strongest result in the whole log was never checked for leakage
  before the workflow stopped. It must be treated as **unvalidated, not
  disproven** — a high-priority item for any continuation of this research,
  not evidence the method doesn't work.

**One open risk flagged in `champion/PARAMS.md` but not previously surfaced
here:** SRC's `MultiTaskLasso` solve cost scales with the dictionary width
(the sum of every Missing candidate's gallery size) — this was **never
profiled against the app's real-time budget at the real roster size (N=8, or
whatever N the live demo uses)**. This is a genuine open risk for
production, not just a research footnote; flagged for the T04 harness/fps
work (`docs/wayfinder/session-linker/tickets/T04-harness-and-fps.md`), not
resolved in this round.

---

## 1 · `champion/` confirmation

Confirmed clean, standalone, and interface-correct — matches
`linker-design.md` §6's **interface**, though not (see §0, §2) its **match
rule**, which SRC replaces.

- **Standalone.** `champion/{__init__,champion,embedder,interface,size_features}.py`
  + `PARAMS.md`. No dependency on the research sandbox's `frozen/`/`method/`
  packages — drops straight into the linker build as `program.md` §9 requires.
  Ported and cleaned from `runs/r3-c2/method.py` (the research variant), which
  depended on `frozen/` for its eval driver; the ablation-switch parameters
  (`use_sci`, `joint_window`) were dropped since the winning configuration is
  fixed, and the DINOv2 embedder/size-feature code now reuses
  `champion/embedder.py` / `champion/size_features.py` instead of a duplicate
  copy.
- **Interface match.** `interface.py` implements exactly
  `build_gallery(crops, masks, meta) -> Gallery`,
  `score(query_crops, query_masks, meta, galleries) -> {candidate: score}`,
  `accept(scores) -> decision | REJECT`.
- **No drift.** `champion.py`'s defaults (`alpha=0.0003, size_alpha=0.5,
  tau=0.30, margin=0.02, mask_dilate_px=2, max_iter=2000`) match `PARAMS.md`'s
  table, `runs/r3-c2/cv_sweep.json`'s `"selected"` entry, and
  `LEADERBOARD.md`'s "Round 3 detail" section exactly.
- **Verified before promotion.** `champion/`'s standalone module was checked
  to reproduce the research variant's guarded numbers bit-for-bit through the
  frozen eval driver: CV re-ID `0.9333±0.0425`, reject `0.9733±0.0533`, 0 twin
  errors; locked-holdout re-ID `1.00`, reject `0.8333` (`champion/PARAMS.md`).
- **Leak-check.** Cleared 2026-07-15, `clean=true` — full file:line audit in
  `TRIED.md` "Round 3 leak-check verdict" (§6 below).

---

## 2 · Recommended parameter values — SRC replaces the baseline's match rule (program.md §9b)

Because a **different method family won** (SRC, not the baseline), this
section follows `program.md` §9(b), not §9(a): describe the winning family
and say explicitly that it replaces `linker-design.md` §6's match rule, not
just parameterizes it.

**What SRC replaces.** `linker-design.md` §6 (checked 2026-07-15, still
unmodified) specifies: embed with a frozen backbone, score each candidate as
**nearest-view max cosine** aggregated across the window, accept iff best ≥ τ
and beats second-best by margin δ, resolve simultaneous returns via a
**Hungarian one-to-one assignment**. SRC's matching primitive is structurally
different: a query window's embeddings are **never compared pairwise to any
gallery view at all**. Instead every offered candidate's enrolled views are
concatenated into one dictionary, and the window is **jointly reconstructed**
as a sparse (L1-penalized `MultiTaskLasso`) linear combination of that
dictionary; the accept score is reconstruction quality × the Sparsity
Concentration Index (how much of the solved code's mass sits on one
candidate vs. spread across all of them) — an algebraic property of a convex
solve, not a similarity, a likelihood, or a discrete assignment. **§6 needs to
be rewritten around this mechanism**, not filled in with SRC's numbers under
the old rule's shape; the "simultaneous returns → Hungarian" clause (§6.5)
also needs a decision — SRC as evaluated here scores each candidate
independently against a shared dictionary and does not itself resolve
mutual-exclusivity across multiple simultaneous new tracks, so §6.5 either
needs its own multi-candidate SRC formulation or an explicit note that the
Hungarian step still runs as a post-hoc layer on top of SRC's per-candidate
scores (not evaluated by this research either way — flagged as a gap, §7).

**SRC's own recommended parameter values** (all from the 630-point CV grid +
ablations in `runs/r3-c2/cv_sweep.json` / `champion/PARAMS.md`):

| open param | recommended value | held-out number that backs it |
|---|---|---|
| **`MultiTaskLasso` L1 penalty (`alpha`)** | `0.0003` | Winner of a 630-point grid (`alpha×size_alpha×tau×margin`). `alpha≥0.002` collapses the solved code to all-zero (a degenerate operating point); too low is dense/non-sparse. |
| **mask-size fusion weight (`size_alpha`)** | `0.5` | No-size-fusion ablation: **0.9083±0.0312 re-ID / 0.7466 reject / 2 twin errors** vs. full: **0.9333±0.0425 / 0.9733±0.0533 / 0**. Size fusion still resolves the instrument1↔2 pair under SRC's matching paradigm, exactly as it did for the baseline. |
| **SCI reject rule** | ON | No-SCI ablation (score = reconstruction quality alone): re-ID **0.9583±0.0373** (higher!) but reject **0.6133±0.1424** (a 36-point collapse) — SCI, not reconstruction quality, is doing essentially all of the open-set work. |
| **joint (window-shared) sparse code** | ON | Independent-per-frame ablation: re-ID **0.9500±0.0312** (close) but reject **0.9200±0.0266** (vs. joint's **0.9733**) — the joint, window-shared formulation is markedly better on rejection, confirming the family's central "joint sparse representation" hypothesis. |
| **accept threshold (`tau`)** | `0.30` | Top of a small plateau tied with `tau=0.28` at `reid=0.9333`; `0.30` picked for more reject margin (`0.9733` vs `0.9467`). |
| **margin** | `0.02` | Vacuous with 1 candidate (the one-missing prior, same as the baseline). |
| **window** | `3` frames (caller-side) | `window=1` ablation: re-ID **0.9143±0.0134**, reject **0.7500±0.0689** — the multi-frame window still matters inside this paradigm. |
| **`mask_dilate_px`** | `2` | Reused unchanged from the baseline's convention; **not independently re-swept** for SRC (same caveat the prior version of this doc flagged for the baseline's own top-K aggregation — now moot for that parameter specifically, see below, but live for `mask_dilate_px`). |
| **`max_iter`** | `2000` | Solver iteration cap, not a matching knob — no ablation needed. |
| **canonicalization** | OFF (unchanged) | Still untested — no challenger family across all 9 rounds tried mask principal-axis canonicalization (§7). |

**The baseline family's old open question is now moot, not answered.** The
prior version of this document flagged "K=1 vs. K=3 top-K gallery
aggregation was never independently isolated" as the one baseline-family knob
this research never swept. Since SRC won and SRC's architecture has **no
top-K gallery-view aggregation step at all** (it uses every gallery atom as a
dictionary column, not a top-K-mean), that specific open question no longer
applies to the current champion. It would still apply if the baseline family
were ever revisited from scratch.

**Voting/window-consistency is now the single most robustly-confirmed
finding in the whole log, by a wide margin.** Every one of the ~35
additional challengers across rounds 1–9 that tested a `window=1` ablation
found the same shape: raw re-ID stays the same or rises slightly, reject
collapses substantially. A representative sample spanning totally unrelated
mechanisms: SRC's own `window1` (reject `0.973→0.750`), SO(2) group
correlation (`1.00→0.741`), Gaussian Process kernel regression
(`0.973→0.941`), DP-vMF clustering (`0.987→0.632`), reservoir computing
(`0.987→0.896`), Grassmannian quantum-fidelity (`0.973→0.818`), hyperbolic
embedding (`0.987→0.614`), Kalman-filter trajectory matching, tree ensembles,
kernel MMD, split-conformal calibration, evidence theory, and every other
family that tested it. `TRIED.md` itself stops counting exact repetitions
past round 7 ("a dozen-plus separate confirmations") — this is no longer a
finding that needs re-confirming, it is the load-bearing design fact of this
whole research program.

---

## 3 · instrument1↔2 verdict — **fully resolved, now confirmed by ~15 independent families**

**Champion (SRC):** 0 twin errors across all 5 CV seeds and the locked
holdout at `size_alpha=0.5`, vs. **2** twin errors at `size_alpha=0`
(`champion/PARAMS.md` — note: the *baseline's* own no-size ablation had found
only 1 twin error; SRC's matching mechanism depends on the size cue slightly
more than the baseline's did for this specific pair, but the qualitative
finding — size fusion resolves it, the mechanism alone does not — is
unchanged).

**Cross-family confirmation only grew stronger in rounds 4–9.** Beyond the
four round-0–3 families the earlier version of this document already listed
(dense Chamfer correspondence, optimal-transport correspondence, classical
descriptors + cohort LLR, vector-symbolic holographic binding — all of which
resolved the pair *without* an explicit size cue), rounds 4–9 add:

- **Genuinely resolves it "for free" (no explicit size cue needed), a new
  mechanism each time:**
  - **r4-c2, topological persistent homology** — 0 twin errors at every
    grid point and ablation, **including** `shape_channel_only` (holes/
    surface channels zeroed): native-scale shape topology (the mask's own
    distance-transform skeleton, never rescaled) separates the 2×-scaled
    pair purely from absolute geometric structure.
  - **r5-c1, tree-ensemble (Extra-Trees)** — 0 twin errors even at
    `size_alpha=0`: random axis-aligned splits on raw CLS coordinates find
    twin-discriminating features without an explicit size prior.
  - **r6-c1, Gaussian Process kernel regression** — 0 twin errors at
    *every* ablation including `no_size_fusion`: the GP's smooth kernel
    geometry separates this pair robustly regardless of the size cue,
    even while every other quality metric collapses without it.
- **Inherits the size-cue dependency (mechanism alone does NOT resolve it),
  the majority pattern:** Hopfield associative memory (+3 twin errors
  without size), SO(2) group correlation (+2), DP-vMF clustering (+7, the
  worst of any family), OpenMax/EVT (+4), split-conformal (+9, also severe),
  kernel MMD (+2), Grassmannian quantum-fidelity (+1), hyperbolic embedding
  (+3), classical keypoint+RANSAC (1 error even with the RANSAC scale-gate
  active, matching the baseline's own no-size count) — every one of these
  reuses champion's DINOv2 CLS+size representation tap-point and inherits its
  size dependence, the same finding the round-0–3 challengers that reused the
  representation (r2-c1, SRC itself) already showed.
- **A genuinely different, partial finding — r9-c2's medial-axis heat-kernel
  scattering.** Its hand-crafted hole-count feature (a first-Betti-number
  proxy computed straight from the mask) cleanly flags "this crop belongs to
  the ring-handled family" — mean hole-count `2.53` for *both* instrument1
  and instrument2, vs. `0.0` for every other instrument and `~0.03` for
  foreign objects — but **cannot tell the two ring instruments apart from
  each other**, since they share the same hole count. Only the explicit
  absolute-size cue does that final split (`no_size_fusion` reintroduces
  **11** twin errors, the most of any family in the whole log). This is a
  useful nuance the round 0–3 evidence didn't have: shape/topology can narrow
  "which family" without narrowing "which specific twin," which is a
  different, weaker claim than the "resolves for free" cases above.

**Recommendation unchanged:** no further work needed on this specific pair.
Keep SRC's (and the linker's) `size_alpha=0.5` size fusion as the resolution
mechanism — it is now independently corroborated by roughly fifteen
unrelated method families across the whole 9-round run, one of the two or
three most robustly-established findings in this document (alongside
window-voting, §2).

---

## 4 · instrument8 verdict — **still unresolved; rounds 4–9 add no new evidence either way, but sharpen a related, distinct finding**

**Champion's (SRC's) own locked-holdout diagnostic does not break down which
foreign source caused its one miss** (`champion/PARAMS.md` reports
`foreign-reject 0.83 (5/6)` with no per-instrument attribution); the
round-0 baseline's own locked holdout, by contrast, found its 2 false-accepts
both went to `instrument7` (score 0.60–0.68), **not** stage-1's `instrument8`
magnet.

**The one piece of direct corroboration for the original stage-1
"instrument8 magnet" hypothesis remains round 1's r1-c2** (optimal-transport
patch correspondence): one specific foreign source image scored consistently
high against `instrument8` across all 3 frames of a window
(`[0.694, 0.593, 0.760]`) — a real, representation-independent cross-family
confirmation (unchanged from the prior version of this document).

**Rounds 4–9 add no further sighting of this specific mechanism** — no
challenger among the ~30 additional experiments reports a foreign object
being consistently absorbed *by instrument8 specifically*. What they add
instead is a **related but distinct** and now very well-established pattern:
**instruments 5, 6, and 7 (not 8) are consistently this dataset's hardest
*genuine* identities**, failing via conservative false-rejection, not via
being a magnet for foreign objects:

- r4-c2 (persistent homology): `instrument5`'s own per-instrument top-1 is
  exactly 0.0 in every CV seed — a margin-gate false-reject, diagnosed as
  the largest instrument by mask area sitting close to two other
  identities' shape-skeleton persistence.
- r6-c2 (DP-vMF): its `kappa_shrink` extension's benefit concentrates on
  "instruments 6 and 7, this project's recurring hard cases."
- r7-c0 (Beta-Bernoulli channels): `instrument7` is the weakest genuine
  identity (`0.80`), while instrument3/8 are perfect.
- r8-c2 (Grassmannian fidelity): false-rejects concentrate on
  `instrument6`/`instrument7` (occasionally `instrument4`).
- r9-c2 (medial-axis scattering): `instrument6` is the weakest
  (`0.33–0.67` across seeds).

**Verdict, updated but not reversed:** the original instrument8-as-magnet
question is **still genuinely open, still not settled either direction** —
one cross-family confirmation (r1-c2) is not contradicted by anything in
rounds 4–9, but nothing in rounds 4–9 adds a second one either. Do **not**
remove `instrument8` from the demo set on the strength of this evidence
alone. Separately, and now well-established across ~5 independent families
spanning representation types from spectral shape descriptors to Bayesian
clustering to subspace fidelity: **instruments 5/6/7 are this dataset's
hardest *genuine* returns**, a distinct, real finding worth carrying into
stage-2 alongside the instrument8 question, not instead of it. The setup
checklist's "add easy phone/pen/tape foreign crops" item was **never
completed anywhere across all 9 rounds** (§7) — every reject number in this
whole log, including every number above, is a lower bound against easy
negatives; the instrument8 question needs more foreign-negative coverage to
settle, not more matcher engineering, and that remains true through round 9.

---

## 5 · What worked / what didn't, across all families (rounds 0–9 + follow-ups)

Selection floor changed after round 3's promotion: rounds 0–3 challengers
were judged against the **round-0 baseline's** floor (`foreign-reject ≥
0.9467`); rounds 4–9 (and the post-round-9 follow-ups) were judged against
**SRC's** floor (`foreign-reject ≥ 0.9733`) and the strict promotion bar
`re-ID > 0.9333 + 0.0425 ≈ 0.9758`. "Tie" below means the reported reject
mean lands on the then-current floor to 4 decimal places.

| round | challenger | family | CV re-ID | CV reject | twin | verdict |
|---|---|---|---|---|---|---|
| 0 | baseline | multi-cue fusion (CLS+size) + temporal voting | 0.850±0.043 | 0.947±0.078 | 0 | **CHAMPION at the time — superseded 2026-07-15 by round 3's SRC** |
| 1-c0 | (Codex-proposed idea) | — | 0 (stub) | 0 (stub) | — | Background run never resolved in time; zero-valued placeholder, no code artifact. Honest non-fatal loss. |
| 1-c1 | dense Chamfer patch correspondence | local patch matching | 0.958±0.046 | 0.840±0.053 | 0 | NOT PROMOTED — breaks the hard reject floor despite the largest re-ID jump in rounds 0–1 |
| 1-c2 | optimal-transport patch correspondence + dustbin | local patch matching | 0.792±0.037 | 0.973±0.033 | 0 | NOT PROMOTED — re-ID short |
| 2-c0 | native-scale retinotopic prototype | representation | 0.9167±0.000 | 1.000±0.000 | 5 (summed) | Cleared both bars on paper but scored below r2-c2 — **never independently leak-checked** (loop reviews only the round's top scorer) |
| 2-c1 | combinatorial set-to-set (Hungarian) assignment | matching | 0.9167±0.0263 | 0.9467±0.0499 (tie) | 0 | Cleared both bars on paper (fragile tie on reject) — **never independently leak-checked** |
| 2-c2 | classical shape/colour/texture + cohort log-likelihood (UBM) | representation | 0.925±0.0312 | 0.9867±0.0267 | 0 | **Beat champion on both axes → selected as round's top scorer → Codex leak-check FAILED (transductive leakage, §6) → NOT PROMOTED** |
| 3-c0 | second-order SPD-manifold activation geometry | representation | 0.575±0.133 | 0.9867±0.0267 | 0 | NOT PROMOTED — honest loss, re-ID crashes |
| 3-c1 (attempt 1) | vector-symbolic holographic (HRR/VSA) binding | representation | 0.683±0.068 | 0.9467±0.0499 (tie) | 0 | NOT PROMOTED — re-ID short |
| 3-c1 (attempt 2) | algorithmic-information / compression-distance (NCD) | representation | 0.358±0.043 | 0.9467±0.0499 (tie) | 3 | NOT PROMOTED — weakest re-ID of any floor-clearing entry in the file |
| 3-c2 | **sparse-representation classification (SRC)** | sparse coding | **0.9333±0.0425** | **0.9733±0.0533** | **0** | **Beat champion on both axes → Codex leak-check CLEAN (clean=true) → PROMOTED, CURRENT CHAMPION** |
| 4-c0 | leaky absorbing graph diffusion | graph / label propagation | 0.9083±0.0553 | 1.000±0.000 | 0 | NOT PROMOTED — re-ID short despite perfect reject |
| 4-c0 (isolated follow-up) | distributionally robust spherical-cap dominance certification | robust optimization | 0.9000±0.0204 | 0.9867±0.0267 | 0 | NOT PROMOTED — re-ID short |
| 4-c1 (attempt 1) | energy-based associative memory (modern Hopfield network) | associative memory | 0.9167±0.0263 | 0.9867±0.0267 | 0 | NOT PROMOTED — re-ID short |
| 4-c1 (attempt 2) | fuzzy Choquet-integral cue fusion + possibility/necessity gating | fuzzy set theory | 0.9167±0.0263 | 0.9733±0.0327 (tie) | 0 | NOT PROMOTED — re-ID short |
| 4-c2 | topological persistent homology (diagram Wasserstein) | algebraic topology | 0.683±0.086 | 0.9733±0.0327 (tie) | 0 | NOT PROMOTED — re-ID crashes |
| 4-c2 (attempt 2) | Kalman-filter (linear-Gaussian) trajectory matching | state-space filtering | 0.458±0.115 (PCA) / 0.408±0.067 (full-dim rescue) | 1.000 / 0.9867 | 0 / 2 | NOT PROMOTED — honest loss, both variants; re-ID craters (raw ranking ceiling 0.99, near-perfect) |
| 5-c0 | counterfactual intervention-response fingerprints | active challenge-response | 0.9583±0.0373 | 0.9733±0.0327 (tie) | 0 | Beats champion's re-ID point, ties the floor, but gain < own seed band — promising, **not a clean promotion** |
| 5-c1 (attempt 1) | random-partition tree ensembles (Extra-Trees + Isolation Forest) | ensemble / anomaly detection | 0.7833±0.0808 | 0.9867±0.0267 | 0 | NOT PROMOTED — re-ID crashes at the floor (raw ranking ceiling 0.9833, among the best in the file) |
| 5-c1 (attempt 2) | reservoir computing (Echo State Network) | dynamical systems | 0.9167±0.0263 | 0.9867±0.0267 | 0 | NOT PROMOTED — re-ID short, but the **shallowest** cliff of any floor-clearing entry (within champion's own seed band) |
| 5-c2 (attempt 1) | classical keypoint (Harris+SIFT) + RANSAC geometric verification | classical local-feature | 0.450±0.055 | 1.000±0.000 | 1 | NOT PROMOTED — re-ID craters despite a perfect, non-fragile reject margin |
| 5-c2 (attempt 2) | geometric silhouette part-decomposition (dustbin Hungarian) | part/segment matching | 0.875±0.037 | 0.9733±0.0327 (tie) | 0 | NOT PROMOTED — re-ID short |
| 6-c0 | SO(2) rotation-orbit cyclic group-correlation | group representation | 0.9666±0.0167 | 1.000±0.000 | 0 | **Beat both of champion's raw point estimates → Codex leak-check FAILED (post-holdout-disclosure sequencing leak, §6) → NOT PROMOTED** |
| 6-c1 | Gaussian Process kernel regression (posterior-variance gating) | Bayesian nonparametric (kernel) | 0.9417±0.0425 | 0.9733±0.0533 (tie) | 0 | Edges past champion's raw re-ID but inside the seed band — **not a clean promotion** |
| 6-c2 | Dirichlet-Process / CRP von Mises-Fisher clustering | Bayesian nonparametric (clustering) | 0.9000±0.0623 | 0.9867±0.0267 | 0 | NOT PROMOTED — re-ID short |
| 7-c0 | episodic Beta-Bernoulli hyperplane-code channel matching | noisy binary channel | 0.9000±0.0333 | 0.9733±0.0327 (tie) | 0 | NOT PROMOTED — re-ID short |
| 7-c1 | OpenMax-style Weibull extreme-value meta-recognition | extreme value theory | 0.8833±0.0409 | 0.9733±0.0533 (tie) | 0 | NOT PROMOTED — re-ID short |
| 7-c2 | split-conformal calibration + Fisher's-method window fusion | distribution-free calibration | 0.7834±0.1034 | 0.9867±0.0267 | 0 | NOT PROMOTED — re-ID well short |
| 8-c0 | transformer-depth rough-path signature kernel matching | rough-path theory | 0.9083±0.0312 | 0.9867±0.0267 | 2 | NOT PROMOTED — re-ID short |
| 8-c1 | kernel two-sample testing (Maximum Mean Discrepancy) | kernel / RKHS set-to-set | **0.9333±0.0204 (exact tie)** | 0.9867±0.0267 | 0 | Ties champion's re-ID exactly, clears reject with margin, **half the seed variance** — inside the seed band, **not a clean promotion**, but the closest genuine near-tie besides r9-c1 |
| 8-c2 | Grassmannian quantum-fidelity subspace matching | quantum information / subspace | 0.8583±0.0500 | 0.9733±0.0533 (tie) | 0 | NOT PROMOTED — re-ID short |
| 9-c0 | Dempster-Shafer evidence theory + Yager conflict-to-unknown fusion | evidence theory | 0.8500±0.0204 | 0.9733±0.0533 (tie) | 0 | NOT PROMOTED — re-ID short |
| 9-c1 | **hyperbolic (Poincaré-ball) embedding matching** | hyperbolic geometry | **0.9833±0.0204** | **0.9867±0.0267** | 0 | **Beats champion on BOTH axes, INCLUDING the strict seed-band promotion bar (`0.9833 > 0.9758`) — the single strongest raw result in the entire run. Flagged for a Codex leak-check that never happened before the run ended. UNRESOLVED / NOT PROMOTED — treat as unvalidated, not disproven (§0, §6).** |
| 9-c2 | medial-axis metric-graph heat-kernel diffusion-wavelet scattering | spectral shape / graph | 0.8083±0.0204 | 0.9733±0.0327 (tie) | 0 | NOT PROMOTED — re-ID short |

**Cross-family patterns worth flagging (extended for rounds 4–9):**

- **Multi-frame window voting** is, by a wide margin, the single most
  robustly load-bearing mechanism found across the entire run — reconfirmed
  independently across every family that tested it, now dozens of times
  (see §2).
- **A recurring reject-vs-re-ID cliff**, present in round 0–3 already, is
  **confirmed again and again through round 9**: SPD-manifold, both VSA/NCD
  attempts, Hopfield, persistent homology, both Kalman variants, both
  tree-ensemble/reservoir attempts, both keypoint/part-decomposition
  attempts, DP-vMF, Beta-Bernoulli, OpenMax, split-conformal, rough-path
  signatures, Grassmannian fidelity, Dempster-Shafer, and medial-axis
  scattering all hit some version of "pushing an operating point hard enough
  to clear champion's reject floor costs substantially more re-ID than
  champion's own recipe costs." Only four entries in the *whole* 9-round run
  got genuinely close without falling off this cliff: round 5's counterfactual
  intervention-response fingerprints, round 5's reservoir computing (the
  shallowest cliff of any floor-clearing entry), round 6's Gaussian Process,
  and round 8's kernel MMD (an exact re-ID tie) — and exactly **one** entry
  cleared the cliff outright: **round 9's hyperbolic embedding**. Read
  together, this is now very strong evidence that champion's specific
  combination (a convex sparse solve + an algebraic concentration statistic,
  or historically the baseline's pooled cosine + orthogonal size cue + window
  voting) sits in a genuinely favourable, and — except for the two open
  procedural questions in §0 — leakage-free spot for this 8×15 dataset.
- **A second-order pattern, newly visible once rounds 4–9 are read together:**
  almost every family's own open-set mechanism follows the same shape —
  `score = a raw similarity/likelihood/reconstruction term × a separate
  concentration-or-consistency term`, and it is consistently the **second**
  term, not the first, doing almost all of the open-set work, at a real but
  usually modest re-ID cost. SCI (SRC), attractor consensus (Hopfield), the
  DP's own new-table posterior (DP-vMF — the single most dramatic instance
  in the whole log: disabling it drops reject from `0.987` to `0.000`),
  alignment-peak sharpness (SO(2)), cross-grade coherence (rough-path
  signatures), the concentration gate (Grassmannian fidelity), and the
  spread gate (hyperbolic embedding) are all structurally the same idea
  independently reinvented across a dozen unrelated mathematical paradigms.
  This is a strong argument that *some* second, separately-computed
  "how concentrated/consistent is this evidence" signal — not just a better
  first-order similarity — is what any future matcher design needs, echoing
  and generalizing what round 0 originally found for window voting alone.
- **No family tried across all 9 rounds and the follow-up dispatches needed
  backbone fine-tuning or a real-video temporal benchmark** — every entry,
  champion included, evaluates multi-frame voting on windowed chunks of
  static per-instrument crops, not labelled return-window video
  (`matching/data/testing` still has no per-instrument identity ground
  truth as of round 9). This limitation is inherited by every number in this
  document (§7).

---

## 6 · The leak-checks — three verdicts on record, plus one open item

Three Codex leak-check verdicts exist in `TRIED.md` (rounds 2, 3, and 6).
They found **two structurally different failure mechanisms**, plus one clean
promotion — do not conflate them.

### Round 2 — transductive leakage (NOT CLEAN)

Round 2 produced three challengers that all beat the round-0 baseline's raw
bars; the loop leak-checks only the round's single highest-scoring candidate,
which was r2-c2 (classical descriptors + cohort log-likelihood). Codex's
review returned `clean: false`: `runs/r2-c2/derive_feature_stats.py` fit
standardization/PCA/whitening and the background ("none-of-the-above")
cohort model on **all** CV instrument crops and **all** CV foreign crops at
once, before the outer-fold split — every evaluated foreign negative directly
influenced the background model used to reject it, and every test-fold query
influenced the fitted representation. This is **transductive test-fold
leakage**: a global statistic fit across the whole CV pool and then reused,
unchanged, across every fold that pool feeds. Codex separately confirmed no
query→gallery leak, no modified eval/split/loader, no holdout peeking, and no
hardcoded labels — the issue was specifically the transductive fit. Round 0
correctly remained champion at the time. (r2-c0 and r2-c1, which also beat
the round-0 bars that round, were never independently leak-checked — the
loop reviews only the top scorer.)

### Round 3 — clean promotion (CLEAN)

SRC's promotion-relevant improvement (+8.3pp re-ID over the baseline) was
flagged per `program.md` §3's "any suspiciously large jump is presumed a leak"
rule and reviewed. Codex's verdict: `clean=true`. A read-only reproduction
through the frozen evaluator matched the reported numbers exactly. Every
category Codex checks came back clean: no harness/eval/split/loader edit or
bypass; no query-crop-into-own-gallery leak (zero item/group-id overlap
across all 5 seeds); no holdout peeking (the locked holdout was called
exactly once, after every hyperparameter was fixed from CV alone); no
hardcoded labels or answer-key access. Crucially, SRC's dictionary is rebuilt
**fresh, per window, from only that seed's own gallery split** — there is no
persistent cross-fold fit anywhere, which is precisely the mechanism that
sank round 2's challenger. This is the one clean promotion on record.

### Round 6 — post-holdout-disclosure sequencing leak (NOT CLEAN FOR PROMOTION)

This is a **different leak mechanism from round 2's**, and the two should not
be conflated: round 2's problem was a *data* leak (test-fold information
statistically contaminating a fitted model); round 6's problem is a
*procedural/sequencing* leak with no data contamination found at all. Codex's
review of the SO(2) rotation-orbit challenger found: no query-label read, no
identity hardcode, no same-sample gallery injection, no alternate
evaluator/loader — every one of the "does the code cheat" checks came back
clean, just as they had for round 2's harness/loader/label categories. The
disqualifying issue instead is **evaluation sequence and provenance**: the
locked holdout was opened and logged for one focused configuration
(`size_alpha=0.5, top_k=3, frame_consensus_power=0.5`, holdout re-ID `1.0`,
reject `0.8333`), and only *afterward* was a materially different, stronger
configuration (`size_alpha=0.35, top_k=1, frame_consensus_power=1.2`) fixed
and reported as the promotion candidate. That later configuration exists in
the run's artifacts only as a hardcoded row in a CV-only verification script
— the "concurrent broader search" claimed to have produced it has no
surviving grid file, log, or selection record. The artifacts therefore cannot
establish that the already-visible holdout numbers played no role in the
later feature/threshold choices — not a claim that they *did*, but that the
provenance needed to rule it out is missing. Codex's own summary: "the method
may be genuinely better, but this submission is not clean enough to certify
or promote it." The method's own CV ablations (cyclic alignment, peak
evidence, frame consensus, size fusion, window=1) are all internally
consistent and support a real matching effect — the finding is about
sequencing discipline, not about the underlying technique being fake.

### Round 9 — an open item, not a fourth verdict

Round 9's hyperbolic-embedding challenger (§0, §5) beat champion on both axes
by more than the strict seed-band margin — a stronger raw result than round
6's, on a cleaner-looking implementation (its own self-audit found no
cross-fold fit, no query-label read, holdout called exactly once, after
selection). Its writeup explicitly flags it for the standard Codex
leak-check before promotion, the same deferral rounds 2, 5, 6, and 8's own
"BEATS" verdicts used. **No such review appears anywhere in `TRIED.md`,** and
the run's remaining activity (several post-round-9 follow-up dispatches into
earlier, under-filled challenger slots) never returns to it. This is not a
fourth leak-check verdict — it is the **absence** of one on the run's single
strongest candidate. Any continuation of this research should run that
review before treating SRC's position as settled; until then, per
`program.md` §3's own rule, an unreviewed result cannot be promoted, but it
also cannot be dismissed.

---

## 7 · Scope this document does NOT cover

All four gaps flagged in the earlier (rounds 0–3) version of this document
were checked against every round-4-through-9 and post-round-9 entry — **none
were closed.**

- **Real return-window video.** Every number above (champion's and every
  challenger's, through round 9 and the follow-up dispatches) evaluates
  temporal voting on windowed chunks of static per-instrument crops —
  `matching/data/testing` still has no per-instrument identity ground truth,
  so a labelled real-video temporal metric still isn't buildable from
  current data.
- **Easy foreign negatives.** `other_objects` is still stage-1's
  adversarially tool-like 60-crop set; the setup checklist's "add easy
  phone/pen/tape crops" item was never completed in any of the 9 rounds or
  the follow-up dispatches (data was read-only throughout). Every reject
  number in this document, across every round, is a lower bound against easy
  negatives.
- **Canonicalization (§8 Line 3).** Still not tested by any round 0–9 or
  follow-up challenger.
- **DINOv3 backbone swap.** Still not attempted anywhere in the log (gated
  on licence acceptance per `program.md`'s setup checklist; not confirmed
  done at any point in the run).

**Two new gaps surfaced by rounds 4–9 specifically, not present in the
earlier version of this document:**

- **SRC's real-time cost at the live roster size was never profiled.**
  `MultiTaskLasso`'s solve cost scales with the dictionary width (every
  offered candidate's gallery size summed together); `champion/PARAMS.md`
  flags this as a known limitation, "not yet profiled against the app's
  real-time budget at N=8 roster size" — a genuine open risk for stage-2/T04
  (§0), not a research footnote.
- **§6.5's "simultaneous returns → Hungarian one-to-one" clause was never
  evaluated under SRC.** Every challenger in this log, including SRC, scores
  one candidate at a time against a shared dictionary or memory; none was
  tested on genuinely simultaneous multi-track returns needing mutual
  exclusivity. `linker-design.md` §6 will need an explicit decision on
  whether the existing Hungarian layer still runs on top of SRC's
  per-candidate scores, or whether SRC's own joint-dictionary formulation
  should be extended to handle multiple simultaneous queries directly (§2).

All figures in this document are **directional** (tiny 8×15 / 60-crop set) —
final confirmation is stage-2, per `program.md` §8.
