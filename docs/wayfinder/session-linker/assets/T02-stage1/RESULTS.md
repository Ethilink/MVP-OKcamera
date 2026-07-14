# T02 stage 1 — embedding benchmark, interim results

**Date:** 2026-07-14 · **Ticket:** [T02 — Benchmark embedding models](../../tickets/T02-benchmark-embeddings.md) (stays **open**; closes after stage 2) · **Machine:** M3 Max, MPS

## TL;DR

- **The method holds on the real set.** 6 of your 8 demo instruments are cleanly
  re-identified (~95–100%). The whole-set number (DINOv2-B, 5 enrolment views):
  **90% top-1**, rising to **92% at 8 views**.
- **One near-twin pair drags everything down: `instrument1` ↔ `instrument2`.**
  Both are ring-handled forceps ([see image](pair_instrument1_vs_2.jpg)). Mean
  embedding cosine **0.895** vs ≤0.71 for every other pair. They cause **~70% of
  all errors** (25 of ~35). This is the Kocher/Pean-twin risk the map warned
  about — now confirmed present *inside the chosen set*. **Decision for Bram +
  Constantijn (T01):** swap one out, or accept it and lean on the one-missing
  prior + multi-frame voting for that pair only.
- **Latency & memory are non-issues.** Even embedding 10 crops at once takes
  <100 ms — the ≤1.0 s link budget has ~10× headroom. Model sits in ~90–350 MB
  next to the detector.
- **Model pick: ViT-B, not ViT-S.** The ticket said "prefer ViT-S if margins
  hold" — they don't quite (86% vs 90%, thinner separation), and latency is not
  a constraint, so ViT-B wins.
- **Blocker — DINOv3 is gated.** Bram's HF account (Bramdc99) gets 403 on all
  `facebook/dinov3-*`. Numbers below are **DINOv2** (ungated, same method) as a
  stand-in. See "Blocker" for the two ways out.

> **What's a proxy and what isn't:** the **data is real** — your 8 demo
> instruments, 15 frames each, shot through the capture tool at 1920×1080 with
> rf-detr masks (same camera + detector as the live demo). Only the **model** is
> a proxy: DINOv2 stands in for the gated DINOv3.

## Method

From `matching/notebooks/embeddings.ipynb`: `AutoImageProcessor` + `AutoModel` →
`last_hidden_state[:,0]` (CLS token) → cosine 1-NN. Each frame is mask-cropped
(rf-detr segmentation → apply mask → crop bbox) before embedding. Standalone
`bench.py` (no dependency on the `matching` package). 8 instruments × 15 crops =
120 crops. Re-ID is evaluated by splitting each instrument into *n* enrolment
("gallery") views + the rest as returning ("query") crops — the shape of the
real linker — averaged over 5 seeds.

## Model comparison (real 8)

| | DINOv2-S (ViT-S/14) | DINOv2-B (ViT-B/14) |
|---|---|---|
| dim / params | 384 / 22M | 768 / 87M |
| offline load | 1.0 s | 1.1 s |
| memory (MPS / RSS) | 88 / 272 MB | 346 / 275 MB |
| latency @batch 1 | 9.2 ms | 15.4 ms |
| latency @batch 10 | 3.8 ms/crop (38 ms total) | 9.0 ms/crop (90 ms total) |
| **re-ID top-1, 5 views** | 86.0% | **90.0%** |
| re-ID top-1, 8 views | 87.5% | **92.1%** |
| margin (self−other), 5 views | 0.161 | **0.199** |

Latency is warm, per link-event batch shapes (1/4/10). The ≤1.0 s budget covers
collect + embed + compare + emit; embed is ≤90 ms here → not the bottleneck.

## Re-ID vs enrolment gallery size (DINOv2-B)

| enrolment views | top-1 8-way | own-gallery-wins |
|---|---|---|
| 3 | 84.8% | 84.8% |
| 5 | 90.0% | 90.0% |
| 8 | 92.1% | 92.1% |

More Start-time views help. Gallery aggregation at 5 views: nearest 89.2%,
**mean 90.8%**, top-3-mean 90.8% — mean/top-K slightly beats nearest-view.

## The near-twin pair

Confusion (DINOv2-B, 5-view gallery, 5 seeds summed → 50 queries/instrument):

| true → | correct | main leak |
|---|---|---|
| instrument1 | 34/50 | → instrument2 (15) |
| instrument2 | 40/50 | → instrument1 (10) |
| instrument3,4,5 | 50/50 | — |
| instrument6 | 46/50 | → instrument3 (4) |
| instrument7 | 41/50 | → instrument5 (5), instrument3 (4) |
| instrument8 | 49/50 | → instrument1 (1) |

`instrument1`↔`instrument2` = 25 of ~35 total errors. Drop that pair and the set
is ~95%+. Secondary, minor: 6→3 and 7→3/5 (both resolve mostly correctly).
[Full 8×8 similarity matrix in `summary_dinov2_base.json`.] Sample of each
instrument: [`montage_one_per_instrument.jpg`](montage_one_per_instrument.jpg).

## Making it robust — fuse mask size (measured, `size_probe.py` + `fuse.py`)

The embedding can't see the twins' size difference because DINO's preprocessor
**resizes every crop to a fixed square** — absolute scale is normalised away, so
it compares shape only. The detector mask still carries scale:

| | mask area (px) | mask length (px) | ranges overlap | 1-D split acc |
|---|---|---|---|---|
| instrument1 | 6796 ± 378 | 299 ± 38 | | |
| instrument2 | 3478 ± 257 | 177 ± 13 | **no** | **100%** |

instrument1 is ~2× the area / ~1.7× the length of instrument2; a single size
threshold separates them perfectly. Fusing z-scored log(area,length) onto the
L2-normalised embedding and re-running re-ID:

| descriptor | top-1 8-way | instrument1↔2 errors |
|---|---|---|
| embedding only | 90.0% | 25 |
| **embedding + size** | **96.8%** | **0** |

So the twin pair is **resolvable in software** — no need to drop an instrument.
Caveats: (1) it's a *fusion* — too much size weight (≥2×) re-collides same-size /
different-shape instruments, so weight it moderately; (2) pixel-size is a valid
physical-size proxy only with the **fixed overhead demo camera + instruments flat
on the table**; a tool held up toward the lens changes apparent size; (3)
absolute thresholds reset at Start-time enrolment (the demo re-measures each
object live), so the mechanism transfers even though these exact pixel numbers
won't.

## Open-set: rejecting foreign objects (`openset.py`, `diagnose_openset.py`)

Bram added 60 foreign-object crops (`model/data/other_objects/`). Two facts up
front: (1) the detector fired on **all 60** at ~0.96 conf, labelling them
`surgical_instrument` — it does **not** filter foreign objects today (Bram: the
detector will improve at skipping them, but never perfectly), so rejection is on
the matcher; (2) the negatives are largely **other tool-like objects** (thin
drivers, rods, a bolt) — a hard adversarial test, not phones/pens/keys.

Single-crop cosine does **not** give a clean accept/reject threshold:

| | genuine self-match | foreign impostor |
|---|---|---|
| mean cosine | 0.764 | 0.519 |
| tail | p10 = 0.589 | **p90 = 0.844, max = 0.925** |

| threshold | true-accept | foreign-reject |
|---|---|---|
| cos ≥ 0.65 | 77% | 72% |
| cos ≥ 0.70 | 66% | 78% |

Best-vs-second-best **margin doesn't rescue it** — the impostors match
*confidently* (large margins), not ambiguously. **All 10 worst false-accepts
match `instrument8`** (0.81–0.92) — the thin, generic wire-holder shape is a
magnet for anything thin/elongated. [`foreign_top10_by_score.jpg`](foreign_top10_by_score.jpg).

Reading: against instrument-*shaped* intruders the single-crop matcher rejects
~70–78%; genuinely non-tool clutter would reject better. The real fix is not a
better still-crop threshold — it's **multi-frame voting over the return window**
(a foreign object rarely matches consistently across ~15 frames) evaluated in the
T04 scenario harness, plus size-fusion, plus reconsidering whether the
generic-shaped `instrument8` stays in the set. **This is the load-bearing open-set
risk for the demo and the main thing stage 2 / T04 must close.**

## Rotation / flip robustness (synthetic stand-in for the flipped-return face)

DINOv2-B, cosine of a rotated/mirrored crop back to its original, and whether it
still 1-NN-matches its own instrument:

| transform | mean cos to orig | still correct 1-NN |
|---|---|---|
| rot 90° | 0.804 | 100% |
| rot 180° | 0.887 | 100% |
| rot 270° | 0.814 | 100% |
| mirror | 0.968 | 100% |

Cosine drops under in-plane rotation, but with a diverse multi-view gallery the
**nearest match is still correct 100% of the time** → for T03, a multi-view
gallery looks sufficient; mask-based orientation canonicalization is likely
*not* needed. **Caveat:** these are synthetic rotations, not the genuine
opposite face. Real flipped-return coverage is a stage-2 gap (below).

## Blocker — DINOv3 gated access / offline

`facebook/dinov3-*` (vits16, vitb16, convnext, …) all return **403
GatedRepoError** for Bramdc99. The demo room may be offline, so weights must be
pre-cached on the machine. Two ways out:

1. **Accept the licence** at `huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m`
   (and vits16), then re-run — gives the apples-to-apples DINOv3 number.
2. **Commit to DINOv2** — ungated, loads offline *today*, already at 90–92%. Its
   only real weakness is the 1↔2 pair, and no frozen embedder cleanly separates
   two ring forceps — that's a choreography/gallery fix, not a model fix.

Recommendation: quick licence-accept so we have the DINOv3 number, but treat
**DINOv2-B as the safe default** that works offline regardless.

## To close T02 (stage 2 gaps)

1. **Resolve the 1↔2 pair** (T01 with Constantijn) — swap or accept-with-mitigation.
2. **Foreign-object negatives** — a handful of phone/pen/tape crops to pin the
   *reject* threshold (open-set). Not capturable from the current 8-instrument set.
3. **Genuine opposite-face views** — confirm the 15 views per instrument include
   both faces, or add them; today's flip evidence is synthetic.
4. **DINOv3 licence** (or formally choose DINOv2).
5. Ideally re-confirm at the actual UZ Leuven room lighting/distance.

## Reproduce

```
# env: uv venv + torch, transformers, pillow, scikit-learn, pycocotools, psutil, scipy
python bench.py --model facebook/dinov2-base  --tag dinov2_base  --out OUT
python bench.py --model facebook/dinov2-small --tag dinov2_small --out OUT
python analyze.py OUT .   # confusion matrix + montages
```

Raw: `summary_dinov2_base.json`, `summary_dinov2_small.json` (in OUT).
