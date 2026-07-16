# orc-e2e-autoresearch — FINDINGS

What the overnight run learned about improving the ORC session linker for the UZ Leuven demo, synthesised from `LEADERBOARD.md`, `runs/`, and `../matcher-autoresearch/TRIED.md`.

**Bottom line:** the shipped pipeline (SRC matcher / DINOv2-B / T08 gallery binding) is still the only configuration that is `demo_safe` on BOTH July-15 takes. Nothing beat it end-to-end. Every challenger that beat it on the directional crop eval failed the end-to-end gate — most by turning correct links into safe Unknowns, one by making a wrong link. See `RECOMMENDATION.md`.

Both priority levers named in the brief were eventually run (2026-07-16, after session-limit retries): **flip/rotation gallery augmentation was falsified** (does not fix the flipped return, and confirms it is a genuine appearance limit), and the **DINOv3-B backbone swap crushed the crop eval (re-ID 1.000) but failed the end-to-end gate on both takes** (roster rebind + a wrong foreign link). Neither is demo-safe; both are written up below.

---

## The decision-relevant result (both takes, real pipeline from cache)

`score_e2e.py` replays the real pipeline on `caches/take-{a,b}.dets.npz` and scores the trace against `gt_events.json` (built from `model/docs/demo-validation.md`). Raw track ids are identical across all matcher/gallery challengers (OC-SORT uses its own embedder), so only linker outcomes change and configs are directly comparable.

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
| **demo_safe (brief definition)** | **true** | **true** |

This reproduces `demo-validation.md`'s "Latest results" exactly: Take B 46 / Take A 24 decision batches; the 6 Take B foreign tracks all settle Unknown (raw 12 @29.2s, 25 @122.2s, 26 @122.4s, 30 @123.4s, 28 @123.8s, 36 @165.0s); hard case @302.0s → `{71:'linked:1', 72:'unknown'}`; hard case @368.0s → `{88:'unknown'}`. This is the reference every challenger diffs against.

---

## What helped / what didn't

### Flip / rotation gallery augmentation (priority lever 1) — RAN 2026-07-16 → FALSIFIED

**Verdict: flip/rotation gallery augmentation does NOT fix the 368s flip case.** `flip` alone is
demo-safe on both takes but **inert** (raw 88 stays Unknown); `flip+rot180` also fails to fix it
**and regresses Take A**. The §8 augmentation hypothesis is falsified on these takes. Artifacts:
`runs/flip-aug/` (driver.py, flip_aug_method.py, crop_eval, 4 traces + 4 score JSONs, verify/ frames, NOTES.md).

| variant | crop re-ID±std | crop F | e2e A/B safe | raw 88 | foreign B | lost/wrong links |
|---|---:|---:|:---:|---|:---:|---|
| flip | 0.9333±0.0425 | 0.9733 | ✅ / ✅ | unknown (not fixed) | 6/6 | none (0 diffs both takes) |
| flip+rot180 | 0.9333±0.0425 | 0.9733 | ❌ / ✅ | unknown (not fixed) | 6/6 | Take A: raw70 linked:3→linked:9, raw77 linked:3→unknown |

**Why it fails (decisive, augmentation confirmed live via atom-counts 20→40→60):** at t=368 the true
identity scores **exactly 0.0** even with its flipped/rotated atoms, and is instead best reconstructed
by the visually-similar (active, ineligible) identity 4 (0.46) → correct fail-safe Unknown. A physical
"turn it over" exposes an aspect that a horizontal mirror or in-plane 180° rotation of upright front
photos **cannot synthesise**; and augmentation is symmetric (every gallery gains the same atoms) so it
never preferentially boosts the true id. **flip+rot180's Take A break:** the rot180 atoms drop
instrument3's razor-thin Take A bind 0.3115 → 0.2793 (below τ=0.30) — exactly the knife-edge bind
demo-validation.md flagged — dropping specimen 3 to session-only and breaking two instrument3 returns.
No threshold was lowered anywhere.

> **⚠️ TWO ERRORS IN `model/docs/demo-validation.md` FOUND & VERIFIED (fold into the canonical doc):**
> 1. The 368s flipped instrument is **instrument3**, NOT "identity 6" as the doc states. Verified by
>    cropping raw 88's detector box from the real Take B frame and matching it to instrument3's
>    reference photo (and to checklist step 7 "remove instrument3, turn it over"). Evidence in
>    `runs/flip-aug/verify/`. So the correct target for that return would be `linked:3`.
> 2. It is a **K=8 SCI decision, not K=1**: every trace shows `comparison_galleries=8` at t=368; the
>    `cos_tau=0.60` K=1 gate never fires on these takes. (This also corrects the "K=1 cos_tau" claim in
>    an earlier draft of this file and in BRIEF.md — that premise was wrong.)

### DINOv3-B backbone swap (priority lever 2) — RAN 2026-07-16 → crushes crops, FAILS end-to-end

**Verdict: DINOv3-vitb16 wins the crop eval decisively but is NOT demo-safe end-to-end on either take. Rejected; SRC/DINOv2-B stays champion.** Artifacts: `../matcher-autoresearch/runs/dinov3-vitb16/` (sanity_check, sweep, cos_tau_calib, best_params, NOTES.md) + `runs/dinov3-vitb16/` (driver, traces, scores).

- **Embedder sanity: PASS.** DINOv3 `last_hidden_state` = `[CLS, 4 register, 196 patch]`; index 0 IS the CLS (== HF `pooler_output`, cos 1.0). Non-degenerate norms. The shipped `[:,0]` pooling is correct for DINOv3 — no change needed.
- **Crop frozen eval (authoritative `run_cv`, 5-seed): re-ID 1.0000 ± 0.0000, foreign-reject 0.9867 ± 0.0267, 0 twin errors** — beats SRC (0.9333 ± 0.0425) by +0.067 (> seedBand) and clears the floor. The harness reproduced the DINOv2 champion exactly (0.9333) in the same run, so the numbers are trustworthy. Best re-swept op point: `alpha=0.0002, size_alpha=0.25, tau=0.24, margin=0.05` (a flat plateau — underdetermined at n≈120).
- **End-to-end: demo_safe A=false, B=false; `binding_matches_baseline=false` on BOTH takes.** Two independent failures:
  1. **Roster rebind.** DINOv3 drops instrument-3's bind score to 0.295/0.294 — just under the linker's FIXED `bind_tau=0.30` (`session_linker.py:107`) — so instrument 3 never binds, enrols as session 9, roster → `{1,2,4,5,6,7,8,9}`. This is the razor-thin instrument-3 bind demo-validation.md already flagged, tipped the wrong way. **Unfixable by the matcher** — binding uses the linker's own gate, not the matcher tau.
  2. **Take B wrong foreign link.** raw 28 (foreign) → `linked:6` at 0.2726, accepted because the re-swept tau=0.24 < 0.2726. Raising tau to reject it re-breaks the instrument-3 bind: under DINOv3 the foreign score (0.2726) and the genuine-hard score (0.294) sit only **0.017 apart**, so **no clean operating point exists** on these takes.
- **The real bottleneck it exposes:** DINOv3 is genuinely a stronger backbone (perfect crop re-ID), but its embedding geometry compresses the foreign-vs-genuine-hard margin at the linker's operating point AND tips the already-fragile instrument-3 bind. The blocker is **instrument-3's thin reference coverage**, not the backbone. Post-demo, DINOv3-B is worth revisiting *together with* better instrument-3 reference capture (more/turned-over poses) + a bind_tau re-verification — not as a drop-in swap.

### Novel matcher families — four cleared the crop bar, ALL failed end-to-end

The overnight session rotated its own rounds 1–6 of novel families on top of the ~25 families already in `TRIED.md` (rounds 0–9). Four challengers beat SRC on the 5-seed crop CV and earned an end-to-end run; every one was rejected. The failure was consistent: **strictly more conservative than SRC** (correct links lost to Unknown), except Poincaré which was worse (a wrong link).

| challenger | crop re-ID (CV) | crop F | e2e outcome | verdict |
|---|---:|---:|---|---|
| SRC (shipped champion) | 0.9333 ±0.0425 | 0.9733+ | A 15/15, B 19/19, 6/6 reject, both hardcases safe | **CHAMPION** |
| Poincaré-ball (r9 reconstruction) | **0.9833 ±0.0204** | 0.9867 | A lost raw70 (14/15); **B wrong link raw28→6, foreign reject 5/6** | rejected (wrong link) |
| NFST null-space collapse (r1-c1) | **0.9833 ±0.0204** | 0.9867 | 0 wrong, foreign 1.0 both; A lost raw70 (14/15), B lost raw37+raw60 (17/19) | rejected (lost links) |
| Nearest convex-hull (r5-c0) | 0.9667 ±0.0312 | 0.9733 | 0 wrong; B safe 19/19; A lost raw70 (14/15) | rejected (lost link) |
| Tangent-distance (r6-c1, query-only injectable form) | 0.9583 (two-sided 0.9750 honest) | 0.9867 | 0 wrong, foreign 6/6; A lost raw63+raw70 (13/15), B lost raw37 (18/19) | rejected (lost links + not injectable) |
| k-reciprocal re-ranking (r5-c1) | 0.8833 ±0.0612 | 0.9733 | e2e not triggered (crop loss) | rejected at crop stage |

Notes per challenger:

- **Poincaré-ball (r9 reconstruction).** The lost round-9 `0.9833` claim is now legitimately reconstructed and leak-clean on crops — but it is the *worst* end-to-end failure mode: a **wrong foreign link on Take B** (`raw28→linked:6`, foreign reject dropping to 5/6) plus a lost link on A. A wrong link on camera is exactly what the fail-safe design forbids.
- **NFST (Null Foley-Sammon Transform).** Cleanest of the losers (0 wrong, foreign 1.0 on both takes) but drops 3 correct links. **Transferable finding:** the crop eval offers balanced 8×5 photo folds; the linker offers *mixed* photo+session galleries (bound identities ~18 atoms, unbound ~3), so the dominant within-class direction becomes the **photo-vs-video domain gap**, and the candidate-set-relative distance scale moves per batch — a `tau` calibrated on crops is not the same `tau` inside the linker. A caution for any future family whose score is normalised by a statistic of the candidate set.
- **Nearest convex-hull.** Barely cleared the crop bar (+0.0022, mean reject at the floor, one seed at 0.8667). Take B strictly safe, Take A lost one link. Fragile crop evidence, conservative e2e loss.
- **Tangent-distance.** The strongest crop challenger since SRC (honest nested 0.9750, a 0.0008 near-miss on the 0.9758 seed-band bar, +0.042 over champion point). **Transferable finding:** the win needs *two* tangent planes (gallery + query); the shipping linker rebuilds a **points-only** `ChampionGallery` (`session_linker.py:337,486,546`) so only the query plane is injectable = the sub-bar 0.9583 variant. Promoting tangent distance would require a **gallery-schema change** (carry per-atom crops / precomputed tangent bases), not a matcher swap.
- **k-reciprocal re-ranking.** Raw ranking (reject off) hit 0.9833, but the same reciprocity signal that rejects foreigners also rejects the harder genuine (flipped / low-overlap) returns; every feasible operating point collapsed to 0.8833, below champion. The "strong ranking, weak reject-at-floor" shape that sank most prior families.

Beyond these, ~25 families across the full log (Siamese, ECOC, SNFM, social-choice rank aggregation, analogical gallery hallucination, episodic max-margin verification, Extra-Trees, GP, DP-vMF, Grassmannian, MMD, Hopfield, SPRT, …) were tried and none promoted. SRC remains champion.

NOTE on wasted slots: the Codex-forwarding challenger slots (r1-c0, r2-c0, r3-c0, r4-c0, r5-c0, r6-c0) only *dispatched* background Codex tasks and returned "pending" stubs — they contributed no completed results within this run (the known Codex-bridge caveat). The substantive challengers above all came from the direct (Opus) slots.

---

## The two documented hard cases

Both were kept as correct fail-safe Unknowns throughout; **no threshold was ever lowered** and the K=1 `cos_tau=0.60` was copied verbatim in every driver.

- **raw 72 @302s — twin ring-forceps swap. Verdict: correct fail-safe Unknown, keep.** No challenger fixed it and none was expected to without a forbidden threshold drop. At that batch raw 71 correctly links to instrument1 and raw 72 stays Unknown — the system refuses to guess between the two ring forceps rather than risk a wrong ID. Correct behaviour on camera.
- **raw 88 @368s — flipped instrument. Verdict: correct fail-safe Unknown, remains open to flip-aug.** Stayed Unknown across every challenger. It is a **K=1 decision** gated by the absolute cosine `cos_tau=0.60`, so no matcher swap can move it without lowering that gate (forbidden). Its designated remedy — flip/rotation gallery augmentation — was not run; that is the one legitimate lever left and the recommended next experiment.

No `hardcase_changes` were auto-counted as wins anywhere; every documented change would have been flagged for human review.

---

## The cross-cutting finding

**The crop frozen eval and the end-to-end linker disagree at the top end.** Four families beat SRC on 5-seed crop CV (Poincaré, NFST, NCH, tangent, all ~0.9667–0.9833 vs SRC's 0.9333) and **every one failed end-to-end** — losing correct links or making a wrong one. On crops even plain nearest-class-mean edges SRC (0.9417 > 0.9333), i.e. the crop eval's top end does not resolve real differences at n≈120. The end-to-end two-take gate was the only signal that separated the field, and it separated cleanly: **only SRC keeps 15/15 + 19/19 links, 6/6 foreign reject, both hardcases safe, roster `{1..8}`, binding stable.** This is exactly what the brief predicted ("treat crop-eval numbers as directional; the end-to-end result is the decision-relevant signal").

---

## Honest caveats

- **Tiny data → directional, not conclusive.** Crop eval = 8×15 instrument crops + 60 adversarial-foreign crops, 5-seed leave-frame-group-out CV re-splitting the *same* ~120 crops (nested CV understates optimism). Locked holdout is n=8 re-ID / n=6 foreign — resolves nothing on its own.
- **Only two takes are the decision signal.** E2E ground truth is the scripted `demo-validation.md` checklist on two July-15 films (COCO `annotations.json` are detection/mask GT only, not re-ID GT). Two videos is the whole end-to-end evidence base; treat it as a demo-safety gate, not a population estimate.
- **Foreign negatives are a lower bound.** `other_objects` is the adversarial tool-like 60-crop set; easy phone/pen/tape negatives were never added, so every foreign-reject number is a lower bound against easy negatives.
- **Priority levers 1 and 2 are now both run** (flip-aug falsified; DINOv3-B wins crops but fails e2e). The open post-demo thread is DINOv3-B **combined with** better instrument-3 reference capture + a bind_tau re-verification — the instrument-3 bind fragility, not the backbone, is the blocker.
- **SRC real-time cost unprofiled.** `MultiTaskLasso`'s solve scales with total dictionary width (summed gallery sizes at the live roster). Fine for cached replay; flagged open risk for the live app, not a demo blocker.
- **No shortcuts taken.** No shipping/`frozen/` file edited; every e2e variant injected by monkeypatching `matching.ChampionMethod` before `load_tracker()`; no `tau/margin/cos_tau` lowered anywhere; every leak-check `clean=true`.
