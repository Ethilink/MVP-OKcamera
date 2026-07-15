# Build brief — T08 (gallery binding) + T10 (app-side Unknown)

**Written 2026-07-15** after a full audit of docs-vs-code. Every claim below was
verified against the working tree at commit `c9690b9`. This file is
self-contained: it is the brief, not a pointer to a chat that will vanish.

**Goal:** build the two remaining unbuilt pieces of the session linker. Bram runs
the live camera test himself **after** this lands — do not attempt it.

**Deadline context:** UZ Leuven demo **2026-07-20**. Bias toward the smallest
thing that works; prefer session-only fallback over a clever wrong bind.

---

## Scope

| Build | Ticket | Where |
|---|---|---|
| §3 gallery binding — make "Instrument N" mean the specimen | [T08](../../tickets/T08-gallery-binding.md) | `model/` |
| §7 Unknown & Pending encoding | [T10](../../tickets/T10-app-side-unknown.md) | `app/` |

**Explicitly NOT in scope — do not build:**

- **T09 live camera test** — Bram does this. Do not run `--camera 0`.
- **T07** — a confirmation question only Bram can answer (are the photographed
  specimens the ones on the demo tray?). Does **not** block T08.
- **§5 stability-gated refresh, §8 rotation/mirror augmentation** — deliberately
  deferred; §3's 15 real poses largely subsume §8.
- **§9 async decision execution** — **closed by measurement**, twice. Do not build
  threading. See "Latency" below.
- **Catalog identification / `CATALOG.md`** — out of scope by Bram's decision.
- **The labelled eval harness** on video `09-07-26-003` — real and valuable, but
  post-demo. See MAP.md "Not yet specified".

## Read first, in this order

1. [`docs/wayfinder/session-linker/MAP.md`](../../MAP.md) — the map. Decisions and
   conventions. Append there; don't duplicate.
2. [`tickets/T08-gallery-binding.md`](../../tickets/T08-gallery-binding.md) and
   [`tickets/T10-app-side-unknown.md`](../../tickets/T10-app-side-unknown.md) — the
   two builds, each with its corrections and gotchas already worked out.
3. **[`model/docs/linker-design.md`](../../../../model/docs/linker-design.md) — CANONICAL** for
   linker behaviour. §3 (binding), §6/§6.5 (match rule), §7 (app encoding).
4. [`model/docs/demo-validation.md`](../../../../model/docs/demo-validation.md) — the
   replay procedure, acceptance checks, and measured latency.

> ## ⛔ [`model/docs/session-linker-SPEC.md`](../../../../model/docs/session-linker-SPEC.md) is SUPERSEDED
>
> It carries a banner saying so, but it reads like a build contract and it sits
> right next to the design doc. **Three of its clauses were deliberately reversed**
> — Hungarian assignment, the Missing-only dictionary, and a mandatory second
> re-score round. Building to it **reintroduces a measured, fixed bug.** It is a
> historical build record. `linker-design.md` wins on every disagreement.

---

## Verified facts — do NOT re-derive these

| Fact | Evidence |
|---|---|
| 73 model tests pass at `c9690b9` | `cd model && .venv/bin/python -m pytest tests/ -q` |
| Linker is built and replay-validated on two takes | `model/docs/demo-validation.md` |
| Specimen photos exist: `model/data/instruments/instrument{1..8}/images/`, 15 views each + COCO masks | listed |
| Negatives exist: `model/data/other_objects/`, 60 tool-like foreign crops | listed |
| Detector weights: `model/weights/checkpoint_best_regular.onnx` — **`.onnx`, not `.pt`**, absolute path | on disk |
| fps is pinned at `DEFAULT_PROCESSING_FPS = 3.0`, threaded into **both** OC-SORT and the linker | `tracking.py:27` |
| Windows at 3 fps: enrolment 2 frames, evidence 2, death 5 | replay trace `windows_frames` |
| Detection caches for both takes: `/private/tmp/orc-take-{a,b}-final-uncached.dets.npz` — **untracked and volatile**; regenerate if missing (~25 min each) | `/private/tmp` |

### Latency — measured, do not re-litigate

Timed on the Take B cached replay, **both halves** of the synchronous path:

| cost | counter | n | median | max |
|---|---|---:|---:|---:|
| SRC solve (score + assign) | `total_ms` | 46 | 40.1 ms | 70.1 ms |
| embed at track death | `build_ms` | 25 | 41.0 ms | 67.3 ms |
| embed at enrolment freeze | `build_ms` | 1 | — | 260.4 ms |

Worst case both on one frame ≈ 137 ms of linker work atop the detector's ~330 ms.
Fine at 3 fps; three orders inside the 1.0 s resolve contract. **§9 is closed.**

⚠️ **T08 will move these numbers** — a bound gallery embeds **15 persistent views +
Start crops** instead of ≤3 crops. See "Trap 4".

Batch **count** (46 on Take B) reproduces exactly; absolute **milliseconds** drift
with machine load. Compare counts across runs, not ms.

---

## The five traps

These are the things that already bit someone. Each is load-bearing.

### Trap 1 · K=1 and `cos_tau` — the open-set gate everyone gets wrong

SRC's SCI statistic is **relative to the candidate set**. With a single candidate
(K=1) SCI is `0/0` and `sim` floors at ~0.47 — above `tau=0.30`. **Nothing can be
rejected.** Measured: all seven other instruments force-linked into instrument1's
slot. `DEFAULT_COS_TAU = 0.60` (`champion.py:49`) is an absolute cosine gate that
covers the genuine K=1 roster. Do not remove it. Do not "simplify" it away.

### Trap 2 · Comparison set ≠ eligibility set

`_comparison_galleries()` (`session_linker.py:345`) scores against the **full frozen
roster**, Active identities included — they are the distractors SCI needs.
`_eligible_missing_ids()` (`:362`) then filters to Missing-only **after** the gate.
**Compare against everything; let only the missing ones win.** Collapsing these two
into "score against Missing" is exactly Trap 1's cause. §3 binding must respect the
same split.

### Trap 3 · Hungarian is not reachable through the matcher seam

`interface.py` is `score() -> dict[candidate, float]` then
`accept(scores) -> one id | REJECT` — each row collapses to a single decision. That
is why the shipped batch path is **greedy per-identity**, not
`linear_sum_assignment`. §3's enrolment binding needs a real matrix, so either
assemble the 8×8 yourself from `score()` calls and gate cells on τ/margin before
`scipy.optimize.linear_sum_assignment` (scipy is already a dep — Deep OC-SORT uses
it in `association.py`), **or** do greedy one-to-one with the confidence gates.
Either is defensible. **State the choice and the reason in `linker-design.md` §3.**

### Trap 4 · Atom-count asymmetry — measure it, don't assume it

SRC is sensitive to how many atoms each candidate contributes: more atoms span a
larger subspace, reconstruct *any* query (including foreign) at lower residual, and
attract more L1 mass — inflating **both** factors of `score = sim * sci`.

Today: Missing carries ≤6 atoms (Start ∪ last-seen) vs Active's ≤3. **The CV sweep
that tuned `tau=0.30` / `margin=0.02` never saw this** — `frozen/eval.py` gives every
gallery an equal `n_gallery_groups=5`.

§3 sharpens it: a **bound** identity carries ~18 atoms (15 persistent + 3 Start), an
**unbound** one carries 3.
- All 8 bind → symmetric → fine.
- **Mixed bind → worse imbalance than today**, biasing every decision toward the
  identities that happen to be bound.

**This must be measured.** Options if it bites: cap atoms per gallery, weight the
dictionary by atom count, renormalize per-candidate. Record the measurement and the
mitigation (or the evidence that none is needed) in the T08 resolution.

### Trap 5 · Embed the persistent galleries ONCE, at enrolment, and cache

The 41 ms death-embed figure is for ~3 crops. 15 persistent views per identity is
5× that. If binding re-embeds persistent views per event, the enrolment freeze grows
from 260 ms toward **seconds** and §9 reopens. Embed persistent galleries once and
cache the vectors. **Re-measure `build_ms` after T08 and update
`demo-validation.md`.**

### Also, still true

- **BGR→RGB**: the live seam delivers BGR (cv2); the research pipeline tuned on RGB.
  `_crop_for_row` (`session_linker.py:187`) converts. Covered by
  `test_b6_2_crops_handed_to_matcher_are_rgb_not_bgr`. Don't break it.
- **fps**: never let `frame_rate` default to 30. At real ~3 fps that stretches a
  "1.0 s" coast to ~10 s of wall time.

---

## T08 — gallery binding (`model/`)

Full spec in [the ticket](../../tickets/T08-gallery-binding.md) and
`linker-design.md` §3. Summary:

- At enrolment close, embed each identity's Start crops, match against the ≤8
  persistent specimen galleries, solve **one-to-one** — no two identities claim the
  same specimen.
- **Bind only when confident**: absolute threshold **and** best-vs-second margin.
- Bound gallery = `persistent ∪ Start`. **Uncertain → session-only**, still a
  first-class identity. **Prefer session-only over a wrong bind.**
- **Bind once, at enrolment. Never re-bind mid-recording** (late bind poisons).
- **Degrade gracefully**: no persistent galleries → everyone session-only, linker
  still works. This path must never crash.
- The match rule is **SRC, not cosine** — §3's "cosine-match" text predates §6's
  rewrite. Do not reintroduce crop-to-crop cosine; the champion computes none.
- Delete the stale `session_linker.py` TODO claiming T07's photos don't exist.

**Unverified assumption to respect:** nobody has confirmed the photographed
specimens are the demo-tray specimens (T07). Binding must work or degrade either
way; T07 decides whether it *helps*, not whether it *runs*.

## T10 — app-side Unknown & Pending (`app/`)

Full spec in [the ticket](../../tickets/T10-app-side-unknown.md) and
`linker-design.md` §7.

> ### ⚠️ Resolve this WITH BRAM before writing app code
>
> §7 opens *"Nothing new crosses the seam. The app already has the frozen roster
> `{1…N}`."* **Both halves are false.** The app has no roster
> (`grep -i roster app/backend/backend/` → nothing), and ids are OC-SORT's raw
> counter, not `{1…N}` — a tray of 8 yields e.g. `{3,5,7,9,10,11,12,14}`.
>
> Two routes:
> - **(a) App derives the roster** from the ids present at Start. Keeps the seam
>   closed; replace every `> N` test with set membership. ⚠️ The app's Start snapshot
>   and the linker's enrolment freeze are **different instants** (~2 frames ≈ 0.67 s
>   apart). If they disagree on the set, every Unknown decision disagrees.
> - **(b) Expose the roster across the seam** — the linker already holds it
>   (`linker.roster`, a frozenset, tested). Costs the seam change §7 avoided;
>   removes the coordination risk entirely. **Recommended.**
>
> Note T08 does **not** make this evaporate: if some identities bind and others fall
> back to session-only, the roster is *mixed* (`{1,2,5,47,52}`) and contiguity never
> returns.
>
> **This changes the contract D8 and `tracker-interface.md` pin — run `/grilling`
> with Bram and record the answer in §7 and D8 before building.**

Then build: resolving spinner + gray mask on young not-in-roster tracks; gray
**Unknown** at debounce, out of Usage/Completeness; 8 roster ids → 8 distinct
colours, not-in-roster → gray; a returned instrument regains its original colour.
Frontend tests per D12 (Vitest + RTL + MSW).

---

## Acceptance

- `cd model && .venv/bin/python -m pytest tests/ -q` → **73+ passed**.
- Replay **both** takes from cache; compare against `demo-validation.md`:
  - roster freezes at **8** on both;
  - **0** full-roster comparison violations;
  - the six Take B foreign tracks (~29 s, 122–124 s, 165 s) **all still settle
    Unknown**;
  - no regression in the decision-batch counts (24 / 46).
- **T08:** atom-asymmetry measurement recorded with its mitigation; `build_ms`
  re-measured and written into `demo-validation.md`; §3 TODO removed; §3 updated
  with the SRC correction and the assignment choice.
- **T10:** a foreign object renders gray Unknown, is never labelled
  `"Instrument N"`, and does not count toward completeness; a returned instrument
  regains its colour; D8a updated from "not yet built" to shipped behaviour.
- **Two known fail-closed rejects on Take B** (302.0 s swapped ring-handled forceps
  scoring 0.0688; 368.0 s flipped instrument staying Unknown instead of identity 6).
  These fail **safe**. **Do not lower `tau` to make them pass.** If T08's persistent
  views fix the 368 s flip case, say so — that is evidence §8 is subsumed.

## Reproduce a replay

```
model/.venv/bin/python model/scripts/replay_session.py \
  --video matching/data/testing/15-07-26-002/videos/15-07-26-002.mp4 \
  --out <prefix> --cache /private/tmp/orc-take-b-final-uncached.dets.npz --from-cache
```

`--from-cache` skips the detector (~2 min vs ~25 min) and still runs real OC-SORT
and real `SessionLinker`. Read `build_ms` / `total_ms` from the trace's `logs`
array. Take A is `15-07-26-001` with the `take-a` cache.

## When done

- One line per ticket under MAP.md **Decisions so far**; `## Resolution` on each
  ticket; `status: closed`.
- Do **not** close [T09](../../tickets/T09-first-live-check.md) — Bram runs it.
- Flag anything you decided that Bram didn't grill. The last build silently reversed
  three grilled decisions; they were right to, but the map found out days later.
