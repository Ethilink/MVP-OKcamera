# orc-e2e-autoresearch — LEADERBOARD

Every experiment: lever, crop-eval metrics, end-to-end demo-safety score, verdict.
The end-to-end demo-safety result on the two July-15 takes (`score_e2e.py` vs
`gt_events.json`) is the decision-relevant signal for the UZ Leuven demo
(2026-07-20). Crop-eval numbers are directional (tiny data).

**Demo-safe** = on BOTH takes: roster `{1..8}`, `foreign_reject_rate==1.0`,
`link_retention==1.0`, zero `wrong_links`, zero `regressions`,
`binding_matches_baseline==true`.

## Legend

- **lever** — the change under test (matcher / embedder / gallery / augmentation).
- **crop re-ID / F** — crop-level frozen-eval CV re-ID top-1 and foreign-reject
  (hard floor F ≥ 0.9733). `—` = not graded on crop eval.
- **A safe / B safe** — end-to-end `demo_safe` per take.
- **hardcase** — change on the two documented fail-safe rejects (raw 72 @302s twin
  swap, raw 88 @368s flip); `none` = both stay Unknown (correct). Any change is
  FLAGGED for human review, never auto-counted as a win.
- **verdict** — CHAMPION / candidate / rejected / regression.

## Results

| # | lever | matcher / embedder / gallery | crop re-ID | crop F | A safe | B safe | link ret A/B | foreign B | hardcase | verdict |
|---|---|---|---:|---:|:---:|:---:|:---:|:---:|:---:|---|
| 0 | **baseline** (shipped champion) | SRC / DINOv2-B / T08 binding | 0.9333 | 0.9733+ | ✅ | ✅ | 15/15, 19/19 | 6/6 reject | none | **CHAMPION** |
| 1 | round-9 reconstruction | Poincare-ball / DINOv2-B / T08 binding | **0.9833±0.0204** | **0.9867** | ❌ | ❌ | 14/15, 19/19 | **5/6 reject** | none | rejected: lost A link + wrong B foreign link |
| r5-c1 | new matcher family | k-reciprocal re-ranking (Jaccard on contextual NN sets) / DINOv2-B | 0.8833±0.0612 | 0.9733 | n/r | n/r | — | — | none | rejected: strong raw ranking (0.9833) but reject-at-floor collapses re-ID to 0.8833 (< champion 0.9333); e2e not triggered |
| r6-c1 | new matcher family | tangent-distance one-shot invariant matching (Simard 1993) / DINOv2-B | 0.9750 honest (flat 0.9833) two-sided; **0.9583 query-only (the injectable form)** | 0.9867 | ❌ | ❌ | 13/15, 18/19 | **6/6 reject** | none (raw 88 flip stays unknown) | rejected: crop win is two-sided-only & NOT injectable (linker gallery is points-only); injectable query-only variant loses 3 correct links to unknown (0 wrong links, binding intact) → not demo-safe |
| flip-aug | **flip/rotation gallery augmentation (§8, priority 1)** | SRC / DINOv2-B / **+horizontal-flip atoms** (20→40 atoms/gallery) | 0.9333±0.0425 | 0.9733 | ✅ | ✅ | 15/15, 19/19 | 6/6 reject | **none (raw 88 stays unknown — NOT fixed)** | rejected: safe but inert — does not fix raw 88 (flipped instrument3 scores 0.0 on its own gallery, 0.46 on similar instrument4); no threshold lowered |
| flip-aug | flip/rotation gallery augmentation (§8, priority 1) | SRC / DINOv2-B / **+flip +rot180 atoms** (20→60 atoms/gallery) | 0.9333±0.0425 | 0.9733 | ❌ | ✅ | **13/15**, 19/19 | 6/6 reject | none (raw 88 stays unknown — NOT fixed) | rejected: raw 88 unfixed AND breaks Take A — rot180 atoms drop instrument3's razor-thin bind 0.3115→0.2793 (<τ=0.30) → rebind, roster {1,2,4,5,6,7,8,9}, raw70 linked:3→linked:9, raw77 →unknown |
| dinov3-vitb16 | **DINOv3-B backbone swap (priority 2)** | SRC / **DINOv3-vitb16 CLS** / T08 binding (re-swept α=0.0002, size_α=0.25, τ=0.24, margin=0.05, cos_τ=0.57) | **1.0000±0.0000** | **0.9867** | ❌ | ❌ | **13/15**, **18/19** | **5/6 reject** | raw72 unknown→linked:2 (flag); raw88 n/a (rebound) | rejected: wins crops big (re-ID 1.0) but FAILS e2e both takes — instrument3 bind score drops 0.30/0.47→0.295/0.294 (<fixed bind_tau=0.30) → roster {1,2,4,5,6,7,8,9} rebound on BOTH takes; Take B wrong foreign link raw28→6 (0.2726 accepted at τ=0.24). Rebind unfixable by matcher (binding uses linker's fixed bind_tau, not matcher τ). No threshold lowered. |

_r5-c1 / r6-c1 crop-eval detail in `../matcher-autoresearch/runs/r{5-c1,6-c1}/RESULTS.md`; both leak-check clean=true. `n/r` = e2e not run._
_flip-aug detail + frame verification (raw 88 = **instrument3**, correcting demo-validation.md's "identity 6") in `runs/flip-aug/NOTES.md`. raw 88 @368s is a K=8 SCI decision, NOT a K=1 cos_tau decision as prior FINDINGS assumed._
_dinov3-vitb16 detail below; crop-eval artifacts in `../matcher-autoresearch/runs/dinov3-vitb16/` (sanity_check, cv_sweep, confirm, cos_tau_calib), e2e in `runs/dinov3-vitb16/` (driver, traces, score JSONs, NOTES.md)._

## dinov3-vitb16 detail (DINOv3-B backbone swap, priority lever 2)

**Embedder sanity check PASSED first** (`../matcher-autoresearch/runs/dinov3-vitb16/sanity_check.json`):
DINOv3 `last_hidden_state` = `[CLS, 4 register, 196 patch]` (seq 201); index 0 is the
CLS (HF `pooler_output` == index 0, cos 1.0000); norms non-degenerate; CLS gives
same-vs-diff instrument separation 0.347 vs 0.164 for patch-mean → shipped
`last_hidden_state[:,0]` pooling is correct for DINOv3, no preprocessing change.

**Crop eval (re-swept, authoritative `frozen.eval.run_cv`):** re-ID **1.0000±0.0000**,
F **0.9867±0.0267**, 0 twins — beats SRC/DINOv2 (0.9333±0.0425) by +0.0667 (> seedBand).
Best point `α=0.0002, size_α=0.25, τ=0.24, margin=0.05`; crop metrics are FLAT for
τ∈[0.20,0.24]×margin∈[0,0.05] (underdetermined at n≈120). cos_tau recalibrated to
0.57 (inactive on both takes).

**End-to-end: FAILS both takes — the crop win did not transfer.** Two independent
root causes:

1. **Roster rebind (both takes), unfixable by the matcher.** Binding uses the
   linker's fixed `bind_tau=0.30`/`bind_margin=0.02` (`session_linker.py:107`), NOT
   the matcher's τ. DINOv3 drops instrument-3's bind score 0.3007/0.4683 (DINOv2) →
   **0.2951/0.294**, just under 0.30 on both takes → instrument 3 never binds →
   enrols as session 9 → roster `{1,2,4,5,6,7,8,9}`, `binding_matches_baseline=false`.
   The other 7/8 binds are identical to baseline. No matcher operating point can move
   a linker-side gate, so this is not tunable without lowering `bind_tau` (forbidden).
2. **Wrong foreign link (Take B).** raw 28 (foreign) → `linked:6` at score 0.2726;
   accepted because the re-swept τ=0.24 < 0.2726. Would reject at τ=0.30 — but raising
   τ re-breaks instrument-3 binding (needs ≤0.294). Foreign (0.2726) and genuine-hard
   (0.294) are only 0.017 apart under DINOv3; DINOv2 keeps them cleanly separated.
   There is no clean operating point.

Also lost correct links (raw70/77 on A, raw71 on B) and swapped the twin hardcase
(raw72 unknown→linked:2, flagged for review). **Verdict: REJECTED for the demo. SRC
/ DINOv2-B remains champion.** No threshold was lowered anywhere.

## r6-c1 detail (tangent distance)

The strongest crop-eval challenger since SRC (two-sided nested-honest re-ID
0.9750, a 0.0008 near-miss on the 0.9758 bar, +0.042 over champion point) but a
clean e2e rejection with two transferable findings:

1. **The win needs two tangent planes; the shipping linker can only supply one.**
   Two-sided (gallery + query planes) = 0.9833 flat / 0.9750 honest; query-only =
   0.9583; gallery-only = 0.9250 (below champion). The linker stores
   `gallery.views` and rebuilds a points-only `ChampionGallery` for scoring
   (`session_linker.py:337,486,546`) — gallery-side crops/planes are gone. So
   only the query-side plane is injectable, and it is sub-bar. **Promoting tangent
   distance would require the gallery to carry per-atom crops or precomputed
   tangent bases (a gallery-schema change), not a matcher swap.**
2. **The flip hardcase (raw 88) is a K=1 decision.** The driver faithfully
   replicates the shipped `cos_tau=0.60` gate there (never lowered), so the query
   tangent never acts on it — the family's flip hypothesis is untestable through
   the shipped K=1 path without lowering cos_tau (forbidden). raw 88 stays
   unknown (correct fail-safe held).

E2E query-only variant is conservative-safe (0 wrong links, 6/6 foreign reject,
binding_matches_baseline true — cleaner than the Poincare experiment's wrong
foreign link) but drops 3 correct links to unknown (raw 63, 70 on A; raw 37 on
B) because tau=0.52 is calibrated on the tangent score scale, not the shipping
bind/decision contract. Artifacts: `runs/r6-c1-tangent/driver.py`, `take-{a,b}.json`.

## Baseline detail (reproduced 2026-07-16, `--from-cache`, shipped config, no extra flags)

Replayed `runs/baseline/take-{a,b}.json` and scored against `gt_events.json`.

| metric | Take A (`001`) | Take B (`002`) |
|---|---:|---:|
| roster | `{1..8}` ✅ | `{1..8}` ✅ |
| decision batches | 24 | 46 |
| binding_matches_baseline | true | true |
| foreign_reject_rate | 1.0 (0 foreign) | 1.0 (6/6 reject) |
| link_retention | 1.0 (15/15) | 1.0 (19/19) |
| wrong_links | 0 | 0 |
| regressions | 0 | 0 |
| hardcase_changes | none | none |
| **demo_safe** | **true** | **true** |

Cross-check vs `model/docs/demo-validation.md` "Latest results" — all reproduce exactly:

- Take B **46** decision batches, Take A **24**. ✅
- **6** Take B foreign tracks all settle Unknown: raw 12 @29.2s, 25 @122.2s,
  26 @122.4s, 30 @123.4s, 28 @123.8s, 36 @165.0s. ✅
- Hard case @302.0s → `{71: 'linked:1', 72: 'unknown'}`. ✅ (twin ring-forceps swap
  fail-safe holds)
- Hard case @368.0s → `{88: 'unknown'}`. ✅ (flipped-instrument fail-safe holds)

This is the reference the whole search diffs against. A challenger promotes only if
it wins the crop frozen eval by > seedBand, holds the foreign-reject floor, passes a
Codex leak-check, and is `demo_safe` on BOTH takes with no lowered threshold.
