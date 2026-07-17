# Matching features — what feeds the matcher, and how to change it

The champion matcher (`champion.py`) compares two things per view: an
**appearance embedding** (DINOv2) and a **size/shape cue** (a small vector
computed from the segmentation mask, no neural net). The two are
L2-normalized/z-scored separately, then concatenated into one "fused vector"
before anything else happens (`ChampionMethod._embed_fused`, `champion.py`).
This doc catalogs both, and says exactly which dial lives where.

## 1. Appearance embedding (DINOv2)

One backbone, swappable, not a per-feature toggle — the CLS token *is* the
appearance feature.

| model | params | embedding dim |
|---|---|---|
| `facebook/dinov2-small` | 21M | 384 |
| `facebook/dinov2-base` **(current default)** | 86M | 768 |
| `facebook/dinov2-large` | 300M | 1024 |
| `facebook/dinov2-giant` | 1.1B | 1536 |

Each also has a newer `facebook/dinov2-with-registers-{small,base,large,giant}`
variant (fixes attention-map artifacts in the original release).

**Dashboard control:** `[tracker.matcher] embedding_model` in `app/mvp.toml`.
Changing it invalidates every existing gallery embedding (they get
recomputed from the raw photos at next startup — see `persistent_gallery.py`)
and changes fused-vector dimensionality, so re-tuning `alpha`/`tau`/`margin`
is expected. Bigger backbones cost roughly 3-4x (large) to 13x (giant) the
per-embed compute of base — check this against the app's real-time budget
before switching in production.

## 2. Size/shape features (the fusion cue)

Implemented in `size_features.py`, computed straight from the instrument's
binary mask (no DINO involved). Every feature below is *implemented and
callable* today; only some are actually fused into matching decisions — see
§3 for the difference.

| name | dim | calibrated? | transform fit | active by default? | what it captures |
|---|---|---|---|---|---|
| `log_area` | 1 | yes | `log` | **yes** | log(mask pixel count) — overall size |
| `log_length` | 1 | yes | `log` | **yes** | log(PCA major-axis extent) — physical length, orientation-invariant |
| `aspect_ratio` | 1 | yes | `log` | no | major/minor axis ratio — elongation (thin scissors vs. squat rongeur), scale-invariant |
| `eccentricity` | 1 | yes | raw | no | elongation from PCA singular values — smoother/less outlier-sensitive cousin of `aspect_ratio` |
| `solidity` | 1 | yes | raw | no | mask area / convex-hull area — catches concave/hollow silhouettes (open jaws, rings) that pure size can't |
| `circularity` | 1 | yes | `log` | no | `4π·area / perimeter²` — 1.0 for a circle, →0 for thin/jagged shapes |
| `extent` | 1 | yes | `log` | no | mask area / bounding-box area — cheaper, cruder cousin of `solidity` |
| `hu_moments` | 7 | yes | raw* | no | classic rotation/scale/translation-invariant shape moments — more expressive, less interpretable |
| `width_profile` | 7 | yes | `log` | no | mask width sampled at 7 points along the major axis — captures tapering (e.g. a needle holder: wide handle, thin tip) that a single ratio collapses away |

\* `hu_moments()` itself already returns log-scaled values (`sign * log10(|hu|)`), so no further transform is applied before z-scoring.

**All nine are now calibrated.** Every `(mean, std)` was derived the same
way as the original `log_area`/`log_length`: fit only over the
development/CV pool (never the locked holdout) — 96 crops, 8 instruments ×
order-0..11 frames from `model/data/instruments/`, the exact population
`frozen.holdout.instrument_cv_selection_pool()` defines in
`model/matcher-autoresearch/`. "Transform fit" says whether the raw value
was `log`-transformed before fitting mean/std — decided per feature by
skewness of its raw distribution over that pool (a heavy right tail gets
logged, e.g. `aspect_ratio` ranged 1.08–54.6 raw but became roughly
symmetric under `log`; already-symmetric ones like `solidity` are fit
directly). Re-derive any of these with
`model/matcher-autoresearch/calibrate_shape_features.py` if the reference
instrument set or camera geometry changes materially — run it and paste the
printed constants into `size_features.py`'s `SHAPE_FEATURES` registry, same
as this round's derivation. Being calibrated doesn't mean tuned for
production: turning a new one on still means re-running the
`alpha`/`size_alpha`/`tau`/`margin` sweep (see `PARAMS.md`), since the fused
vector's composition changed.

## 3. How to enable/disable a feature — `app/mvp.toml`

Two dials, both under `[tracker.matcher]`, both operator-editable without
touching code:

```toml
[tracker.matcher]
# Weight of the ENTIRE size/shape block vs. DINO appearance. 0.0 = size/shape
# cue fully off regardless of active_features below.
size_alpha = 0.25

# WHICH of the features in the table above are actually computed and
# concatenated into that block. Edit this list to add/remove one.
active_features = ["log_area", "log_length", "aspect_ratio"]
```
(current live values in `app/mvp.toml` — `size_alpha` re-tuned for this
3-feature composition, see `PARAMS.md` "Round 3b". `solidity`/`extent`/
`width_profile` were tried as a swap-in on 2026-07-17 and swept out — a real
re-ID regression, not promoted; see `PARAMS.md` "Round 3c".)

- **`size_alpha`** is the master on/off + weight for the whole size/shape
  block. Set it to `0.0` and no shape feature affects matching, no matter
  what `active_features` contains.
- **`active_features`** picks *which* registered features (by name, from the
  table above) get computed and concatenated, in that order, whenever
  `size_alpha > 0`. Add a name to bring a feature in; remove it to drop it.
  An empty list (`[]`) means no size/shape vector at all, same effect as
  `size_alpha = 0`.

**Fail-fast validation:** an unrecognized name in `active_features` raises a
`ValueError` (listing the valid names) when the matcher is constructed at
process startup — same "fail fast, not on first frame" philosophy as the
rest of `[tracker]`. A typo never silently no-ops.

**Consequence of changing the list:** the fused vector's length changes
(each entry in the table adds its `dim` column count), so `alpha`
(the SRC sparsity penalty) and `tau`/`margin` (accept thresholds) were tuned
against whatever composition was active during the last CV sweep — treat
adding/removing a feature as "needs re-tuning," not a free lunch.

## 4. Adding a genuinely new feature

Toggling something in the table above is a config-only change. Adding a
feature that *isn't* in the table yet is still a code change: implement it
in `size_features.py`, register it in the `SHAPE_FEATURES` dict (compute
function, output dimensionality, `(mean, std)` — use `_UNCALIBRATED` until
you calibrate it), and it becomes selectable via `active_features`
immediately, no other wiring required. To calibrate it, add its name to
`model/matcher-autoresearch/calibrate_shape_features.py`'s feature list (or
just pass it as an argv name), run the script, and paste the printed
`mean`/`std` in place of `_UNCALIBRATED` — same recipe used to calibrate
every feature in the table above.
