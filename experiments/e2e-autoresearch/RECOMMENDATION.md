# orc-e2e-autoresearch — RECOMMENDATION

**The single config to run at the UZ Leuven demo (2026-07-20): KEEP THE SHIPPED CONFIG. No change.**

Run the shipped pipeline exactly as it stands: **SRC matcher (`ChampionMethod`) / DINOv2-B backbone / T08 gallery binding ON**, with the current `matching/champion.py` defaults and `load_tracker()` unmodified.

## Exact change to the shipped pipeline

**None.** Keep every `matching/champion.py` default as-is:

```
DEFAULT_ALPHA          = 0.0003
DEFAULT_SIZE_ALPHA     = 0.5
DEFAULT_TAU            = 0.30
DEFAULT_MARGIN         = 0.02
DEFAULT_MASK_DILATE_PX = 2
DEFAULT_MAX_ITER       = 2000
model_id               = "facebook/dinov2-base"
DEFAULT_COS_TAU        = 0.60
```

And keep `load_tracker()` with **T08 gallery binding enabled** (the default). Do not pass any override; do not monkeypatch `ChampionMethod`; do not disable binding. The foundation confirmed binding-disabled is correctly flagged `rebound` / `demo_safe:False`, so binding must stay ON for `binding_matches_baseline` to hold. Net diff against what ships today: **zero lines.**

## Why

The shipped SRC config is the **only** configuration that is `demo_safe` on BOTH takes. Every challenger that beat it on the directional crop eval failed the end-to-end gate:

- **Poincaré-ball** (crop 0.9833) made a **wrong foreign link** on Take B (`raw28→linked:6`) and dropped foreign reject to 5/6.
- **NFST** (crop 0.9833), **nearest convex-hull** (crop 0.9667), and **tangent-distance** (crop 0.9750 honest) each turned correct links into Unknowns (Take A retention 13–14/15; NFST also 17/19 on B).

A wrong link on camera is worse than a safe Unknown, and the shipped config makes **zero wrong links** while retaining **every** correct link (15/15 Take A, 19/19 Take B) and rejecting **every** foreign object (6/6 on B). Nothing cleared the promotion bar (crop win > seed-band AND foreign-reject floor held AND leak-check clean AND `demo_safe` on both takes with no lowered threshold). See `FINDINGS.md` for the full field.

## End-to-end demo-safety evidence on BOTH takes (the config being kept)

Reproduced 2026-07-16, `replay_session.py --from-cache`, shipped config, no extra flags; scored by `score_e2e.py` vs `gt_events.json`:

| metric | Take A (`001`) | Take B (`002`) |
|---|---:|---:|
| roster | `{1..8}` ✅ | `{1..8}` ✅ |
| binding_matches_baseline | true | true |
| foreign_reject_rate | 1.0 (0 foreign) | 1.0 (6/6 reject) |
| link_retention | 1.0 (15/15) | 1.0 (19/19) |
| wrong_links | 0 | 0 |
| regressions | 0 | 0 |
| hardcase_changes | none | none |
| **demo_safe** | **true** | **true** |

## What to expect on camera (and the honest live answer)

The two documented hard cases will show as **Unknown**, which is **correct, intended fail-safe behaviour, not a bug**:

- **@302s** twin ring-forceps swap → `{71: linked:1, 72: unknown}` — the system refuses to guess between the two identical ring forceps.
- **@368s** flipped instrument → `{88: unknown}` — the system refuses to ID a tool it only sees flipped.

If asked live: the linker is tuned to say "Unknown" rather than risk a wrong instrument ID; both refusals are the safety margin working as designed.

## After the demo

1. **Flip/rotation gallery augmentation — RAN, FALSIFIED (2026-07-16).** Does NOT fix the flipped-instrument
   return. `flip` is demo-safe but inert (the flip stays a correct Unknown); `flip+rot180` also fails and
   regresses Take A (tips instrument3's razor-thin bind below τ). A physical "turn it over" exposes an
   aspect a mirror/in-plane-rotation of front photos can't synthesise. See `FINDINGS.md`. **This is good
   news for the demo narrative:** the flipped-instrument Unknown is a genuine appearance limit, not a
   cheaply-fixable bug — so showing it as Unknown is honestly correct. The real fix would need extra
   captured poses (turn each specimen over during reference capture), not synthetic augmentation.
   - **Also found — two errors in `model/docs/demo-validation.md` to correct:** the 368s flip is
     **instrument3**, not "identity 6" (verified against the frame + reference photo); and it is a **K=8
     SCI decision, not K=1** (`comparison_galleries=8` at t=368, the `cos_tau` K=1 gate never fires on
     these takes).
2. **DINOv3-B backbone swap — RAN, REJECTED for the demo, but promising post-demo (2026-07-16).** Crushes
   the crop eval (re-ID **1.000 ± 0.000** vs SRC 0.9333, foreign-reject 0.9867) — it IS a stronger backbone.
   But it FAILS the end-to-end gate on both takes: (a) it tips instrument-3's already razor-thin bind below
   the fixed `bind_tau=0.30`, rebinding the roster to `{1,2,4,5,6,7,8,9}`; and (b) it makes a wrong foreign
   link on Take B (raw28→6), with foreign and genuine-hard scores only 0.017 apart so no clean operating
   point exists. **The blocker is instrument-3's thin reference coverage, not the backbone.** The right
   post-demo experiment is DINOv3-B *together with* better instrument-3 reference capture (more/turned-over
   poses) and a bind_tau re-verification — not a drop-in swap. See `FINDINGS.md`.

Neither lever changes today's recommendation, and neither should be applied to the 2026-07-20 demo without
passing the same two-take `demo_safe` gate the shipped config passes.

## The through-line

Three independent search directions — 6 rounds of novel matcher families, flip/rotation augmentation, and
a DINOv3-B backbone that scores a perfect 1.000 on crops — all converged on the same answer: **the shipped
SRC / DINOv2-B / T08 config is the only one that is demo-safe on both takes.** The recurring failure mode is
not weak matching; it is that stronger crop-level matchers either lose the hard genuine returns to safe
Unknowns or rebind the fragile instrument-3 slot. The single highest-leverage improvement for the future is
better reference capture of the thin/twin instruments (esp. instrument 3, and turned-over poses), which
would unblock both DINOv3-B and the flip case — not a new matcher.
