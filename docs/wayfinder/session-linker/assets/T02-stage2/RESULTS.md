# T02 stage 2 — matcher autoresearch results

**Substance lives in `experiments/matcher-autoresearch/`** (a separate,
self-contained sandbox — see its `README.md` for why it's split out: it reads
`model/data`/`matching/data` read-only and writes only inside itself, so it
never collides with a chat working the wayfinder ticket tree at the same
time). This file is the wayfinder-side pointer + headline summary, mirroring
[`T02-stage1/RESULTS.md`](../T02-stage1/RESULTS.md) and
[`T04/RESULTS.md`](../T04/RESULTS.md)'s pattern.

Ran the `program.md` brief (copied verbatim into
`experiments/matcher-autoresearch/program.md`) as a Karpathy-style autoresearch
champion/challenger Workflow overnight: frozen harness + one guarded held-out
metric (re-ID top-1 subject to a foreign-reject floor) + a mutable method
module, budget-paced, every promotion gated on a Codex leak-check.

## Headline result

- **Champion changed once, cleanly.** The round-0 baseline (DINOv2-B CLS +
  masked crop + moderate mask-size fusion + top-3-mean + multi-frame voting —
  stage-1's own recipe) held for 3 rounds, then was beaten and superseded by
  round 3's **sparse-representation-based classification (SRC)**: a query
  return-window's embeddings are jointly reconstructed as a sparse linear
  combination of every *Missing* candidate's gallery atoms (`MultiTaskLasso`),
  scored by reconstruction quality × Sparsity Concentration Index (open-set
  rejection signal). Passed a Codex leak-check (`clean=true`; full file:line
  audit in `TRIED.md` "Round 3 leak-check verdict").
  - CV (5-seed): re-ID **0.9333±0.0425**, foreign-reject **0.9733±0.0533**, 0
    twin errors.
  - Locked holdout (one-time, never optimized against): re-ID **1.00** (8/8),
    foreign-reject **0.83** (5/6).
- **6 more rounds (4–9), ~18 more challengers, ~20 distinct method families in
  total** — patch correspondence, optimal transport, retinotopic prototypes,
  Hungarian set assignment, cohort log-likelihood, SPD-manifold geometry,
  holographic binding, compression distance, energy-based associative memory,
  persistent homology, graph diffusion, counterfactual fingerprints, RANSAC
  keypoints, extra-trees/isolation-forest, SO(2) group correlation, Gaussian
  processes, Dirichlet-process clustering, Beta-Bernoulli codes,
  OpenMax/Weibull, split-conformal calibration, rough-path signatures,
  Grassmannian quantum-fidelity, MMD two-sample testing, Dempster-Shafer,
  wavelet scattering, hyperbolic embedding, robust spherical-cap certification
  — **none displaced SRC.** One (round 6, SO(2) rotation-orbit matching) beat
  the champion's raw numbers but **failed its leak-check** on a real
  procedural issue (hyperparameters were fixed *after* the locked holdout had
  already been seen once — a sequencing leak, a different failure mode than
  round 2's transductive-leakage failure). Diminishing returns are clear by
  round 9; **no further open-ended method search is recommended.** Full
  per-round table + both leak-check writeups: `FINDINGS.md` §5–6.
- **Twin (`instrument1`↔`instrument2`) verdict: fully resolved**, corroborated
  independently across 4+ unrelated method families, not just the champion's
  own size-fusion cue (`FINDINGS.md` §3).
- **`instrument8` (thin rongeur) verdict: still open.** The champion's own
  locked holdout didn't reproduce stage-1's "magnet" risk, but one other
  family (round 1's optimal-transport correspondence) reproduced it
  non-trivially across a full 3-frame window. Needs more foreign-negative
  coverage to settle, not more matcher engineering (`FINDINGS.md` §4).

## Recommended parameter values → feed into `model/docs/linker-design.md`

Full table + the held-out number behind each value: `FINDINGS.md` §2.
Headline: **multi-frame window voting (window=3) is the single most
robustly-confirmed mechanism in the whole log** — independently reconfirmed
across 5+ unrelated representations/matching paradigms, more than any other
design choice tested.

## What this does NOT cover (open before T05)

- **Real return-window video.** Every number above evaluates windowed *static*
  per-instrument crops, not real return events. [`T04/RESULTS.md`](../T04/RESULTS.md)
  runs a real-video re-ID + open-set benchmark instead, but with the
  round-0-style champion, not SRC — **the two results are not yet reconciled
  against each other on the same data.** Validating (or re-running) SRC
  against T04's cached real-video tracklets is the natural next step before
  committing to it as the linker's match rule.
- **SRC's runtime cost at real roster size** (`MultiTaskLasso` dictionary width
  scales with the number of currently-Missing candidates) was never profiled
  against the ≤1 s / event-driven budget in `linker-design.md` §9 — flagged in
  `champion/PARAMS.md`'s "known limitations," not yet resolved. This matters
  more than usual because SRC adds a new runtime dependency (`scikit-learn`'s
  Lasso solver) that the vendored ~100-line cosine embedder in
  `linker-design.md` didn't need.
- Easy foreign negatives (phone/pen/tape), mask canonicalization, a DINOv3
  backbone swap — none tested this round either (`FINDINGS.md` §7).

## Where things live

- Full detail: `experiments/matcher-autoresearch/{LEADERBOARD.md, FINDINGS.md,
  TRIED.md, champion/}`.
- This ticket's stage 1: [`T02-stage1/RESULTS.md`](../T02-stage1/RESULTS.md).
- The brief that was actually run:
  `experiments/matcher-autoresearch/program.md` (a copy of this folder's
  `program.md`, per its own README — kept separate so the sandbox never
  touches the wayfinder ticket tree mid-run).

Ticket stays **open** — the real-video reconciliation against T04 and the SRC
runtime profiling are the two concrete gaps before T02 can close.
