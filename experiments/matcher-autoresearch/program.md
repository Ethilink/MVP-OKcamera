# Overnight matcher auto-research — `program.md`

**Launch this in a fresh chat.** It drives an autonomous overnight loop that
*invents and tests new techniques* to make the ORC **session matcher** better —
then keeps only what measurably beats the current champion on **held-out** data.
Karpathy-`autoresearch` shape: a frozen harness + one guarded metric + a mutable
method module + a champion/challenger loop. Feeds **T02 stage 2** and finalizes
the open parameters in [`model/docs/linker-design.md`](../../../../../model/docs/linker-design.md).

> **Read first (do not skip):**
> [`linker-design.md`](../../../../../model/docs/linker-design.md) (what the
> matcher must do, § 4–6), and [T02 stage-1 `RESULTS.md`](../T02-stage1/RESULTS.md)
> (what's already known — the near-twin `instrument1↔2`, the open-set risk, the
> mask-size fusion result, DINOv2-B vs gated DINOv3). Reuse the stage-1 code in
> [`../T02-stage1/`](../T02-stage1/) as the harness base.

---

## 1 · Mission

Make the **session matcher** better on two axes at once:

1. **Re-identification** — a returning instrument matches its own missing gallery
   (open-set, multi-candidate). Session re-ID only: *"same physical object as one
   that left?"* — never catalog identity.
2. **Open-set rejection** — a foreign object (audience phone/pen/tool-like
   clutter) is *rejected*, never force-linked. **This is the load-bearing demo
   risk** (stage-1: single-crop cosine does not separate cleanly, and best-vs-
   second margin does *not* rescue it — impostors match confidently).

You may **invent new techniques and write new code** to do this. Keep only what
beats the champion on held-out data by more than seed noise.

## 2 · Data & context (real, fixed)

- **Instrument crops:** `model/data/instruments/` — 8 demo instruments × ~15
  frames, 1920×1080, rf-detr masks (same camera+detector as the live demo).
- **Foreign negatives:** `model/data/other_objects/` — for the reject metric.
  (Stage-1's negatives were adversarially *tool-like*; see setup checklist —
  add a few easy phone/pen/tape crops so the reject number is honest.)
- **Videos (for multi-frame / temporal work):**
  `matching/data/testing/09-07-26-00{1..4}/` — video + frames + `dataset.json`.
- **Demo facts that license certain tricks:** fixed **overhead** camera,
  instruments **flat on the table** (so pixel mask-size is a valid physical-size
  proxy); **no duplicate types except the `1↔2` forceps** (~2× size difference);
  returns are observed over a **window of frames** (temporal voting is available).
- **Embedder:** **DINOv2-B** (ViT-B/14) is the offline default (90–92% stage-1);
  DINOv3 is gated (optional ablation if the licence is accepted — see checklist).

## 3 · The one rule that keeps this honest: frozen vs mutable

**If you touch the frozen surface, the results are void.** Copy the stage-1 code
into a fresh working dir and split it:

- **FROZEN — never edit, never let the method read from it:**
  - the **data loader** and the **held-out split protocol**,
  - the **eval / metric** code,
  - the **locked test holdout** (see § 4).
  The method is called only through a **fixed interface**, e.g.
  `build_gallery(crops, masks, meta) -> Gallery`,
  `score(query_crops, query_masks, meta, galleries) -> {candidate: score}`,
  `accept(scores) -> decision | REJECT`.
- **MUTABLE — invent freely here:** one `method/` module implementing that
  interface. Add files, new descriptors, new scoring — anything, as long as the
  eval only ever calls the fixed interface and the method **never** sees query
  labels, leaks a query crop into its own gallery, or reads the holdout.
- Any suspiciously large jump (e.g. +15% in one step) is **presumed a leak** until
  a Codex review clears it.

## 4 · The metric (guarded, held-out) — do NOT maximize re-ID alone

A loop that maximizes re-ID top-1 will overfit 8×15 crops and **destroy
rejection**. So:

- **Selection score (single scalar):** `re-ID top-1` **subject to a hard floor**
  `foreign-reject ≥ F` at the operating threshold. A challenger that breaches the
  floor **loses**, regardless of re-ID. (Set `F` from the champion's current
  reject rate; it may only go up.)
- **Always report, per experiment:** re-ID top-1, `instrument1↔2` error count,
  foreign-reject rate + true-accept rate at threshold, per-instrument breakdown,
  and **±CI across CV seeds**.
- **Protocol:** leave-frame-group-out **cross-validation** over the crops for
  *selection* (keep stage-1's 5-seed averaging); a **locked holdout** (e.g. one
  video's returns, or a reserved instrument-frame split) scored **only for the
  champion**, never optimized against. Multi-frame voting is evaluated over real
  return windows from the videos.
- **Promotion rule:** a challenger becomes champion only if it beats the champion
  by **more than the seed-variance band**. Log the band. Numbers are
  **directional** (tiny set) — final confirmation is stage-2 with T07 photos +
  real room.

## 5 · Loop (champion / challenger)

1. **Champion = the current best** — start it at the stage-1 baseline: DINOv2-B,
   masked crops, **mask-size fusion** (moderate weight), **top-3-mean**
   aggregation, **multi-frame voting** over the return window. Its held-out score
   is the bar.
2. **Each round:** propose **2–4 diverse** challenger hypotheses → implement each
   as a `method/` variant → run the frozen eval → measure.
3. **Keep-if-better-by-margin** → new champion. Append to `LEADERBOARD.md` and a
   per-experiment record (hypothesis, code diff, full metrics, verdict).
4. **`TRIED.md`** logs every idea so none repeats; **rotate idea families** each
   round to avoid tunneling on one technique.
5. Repeat until the **budget target** is hit.

## 6 · Technique menu — seeds; the point is NEW METHODS, not a knob search

**This is not a 4-parameter search.** Fusion weight / voting / aggregation /
threshold are the *champion baseline's* knobs — re-tuning them is the **least**
valuable thing you can do. The job is to invent **genuinely different method
families / paradigms** that plug into the same `build_gallery / score / accept`
interface and beat the champion on held-out data. Each challenger must **declare
its family**; prefer a family **not yet in `TRIED.md`** over a tweak of the
current champion. The seeds below are a *floor*, not a ceiling — invent past them.

*The two boundaries (only these):* (a) it must fit the general
`build_gallery/score/accept` plug (so it drops into the linker); (b) **no
fine-tuning the backbone on the 8 demo specimens** (that overfits a frozen
embedder and won't survive live re-enrolment) — but *generic, low-capacity
learned combiners* (a linear/logistic head on cues, a calibrator, a re-ranker)
trained with held-out CV are fair game.

Seed families (go beyond these):
- **Different representations entirely:** classical local-feature matching
  (SIFT/ORB/keypoint correspondence — viewpoint-robust), colour/texture histograms,
  contour/Fourier/shape-context descriptors, SAM or detector-backbone features,
  hand-crafted geometric signatures — *not just DINO CLS cosine*.
- **Different matching paradigms:** set-to-set (gallery set vs query set) instead
  of crop-to-crop; part/segment-based matching; one-class model per identity;
  a discriminative head (metric learning / logistic) over generic features;
  ensembles / mixture-of-experts fusing several cues.
- **Temporal / voting:** mean-of-per-frame-decisions, SPRT (early accept/reject),
  window consensus, trajectory-of-appearance models. *(Stage-1 named voting the
  real open-set fix — a strong family to develop, not just a knob.)*
- **Multi-cue fusion:** embedding + size + colour + texture + geometry, combined
  by a learned generic head (watch stage-1's caveat: ≥2× size weight re-collides
  same-size/different-shape tools).
- **Open-set-specific scoring:** `other_objects` as a **"none-of-the-above"
  cohort**, energy/OOD scores, Mahalanobis / LOF, one-class SVM, cohort
  (z-/t-norm) normalization, calibration; and the **`instrument8` magnet** problem
  (thin generic shape absorbs thin intruders — fix with a shape/size prior, a
  dedicated reject rule, or conclude it should leave the set).
- **Embedding-space tricks (cheaper wins):** patch-token pooling vs CLS,
  multi-layer concat, PCA/whitening, re-ranking (k-reciprocal NN).
- **Backbone swaps:** DINOv2-S/B, **DINOv3 if licensed**, CLIP — keep everything
  **offline-runnable**.

## 7 · Orchestration — Codex + subagents + credit pacing

Run as a **budget-paced Workflow** ("slow but steady to maximise credits"):

- **Fan out challengers** with `parallel` — **2–4 per round, not 16** (protects
  both credits and the tiny-data signal; more parallelism just adds noise-chasing).
  Each challenger subagent implements + runs one hypothesis and returns a
  **structured metric** (JSON schema: technique, diff summary, metrics, seed CIs).
- **Codex (the `codex:rescue` skill / GPT-5) plays two roles:** (a) an
  **alternating idea proposer** each round for genuine diversity (different model,
  different priors), and (b) a **reviewer/debugger** — any challenger that errors
  goes to Codex rescue; any *promotion* must survive a Codex **leak-check** review
  before it becomes champion.
- **`loop-until-budget`:** stop clean at the credit target set at launch (e.g.
  `+300k`). **Write results continuously** (leaderboard + per-experiment records)
  so a stop/crash loses nothing and the run is **resumable**.
- Each experiment is **time-boxed** (the set is small — a run is seconds/minutes);
  a hung experiment is killed and logged, never fatal to the loop.

## 8 · Guardrails

- Isolation + timeout per challenger; failures logged, **non-fatal**.
- **Offline:** weights pre-cached; no network mid-run (DINOv3 only if the licence
  was accepted *before* launch).
- **Never** edit the frozen eval/split/loader; **never** read the holdout during
  selection; flag suspicious jumps for the Codex leak-check.
- Treat all numbers as **directional** — the deliverable is *ranked techniques +
  recommended parameters*, confirmed later at stage-2, not final production values.

## 9 · Deliverables (write to this dir)

- **`LEADERBOARD.md`** — ranked techniques with held-out metrics + CIs.
- **`champion/`** — the winning `method/` module, clean, with the
  `build_gallery / score / accept` interface **matching `linker-design.md` § 6**
  (so it drops straight into the linker build).
- **`FINDINGS.md`** — what worked / what didn't across method **families**, and:
  (a) if the champion is the baseline family, the **recommended param values**
  (fusion weight, voting scheme+window, aggregation, accept threshold,
  canonicalization y/n); (b) **if a different method family won, describe it and
  how it plugs into the linker** — it may *replace* `linker-design.md` §6's match
  rule, not just parameterize it, in which case say so explicitly so the design
  gets rewritten, not just filled in. Either way, give the **`instrument8` + `1↔2`
  verdicts** with the held-out numbers. This is the T02-stage-2 (partial) output
  that reconciles back into `linker-design.md`.
- **`TRIED.md`** — full history, so a later run doesn't repeat dead ends.

## 10 · Setup checklist (before launch)

- [ ] Copy [`../T02-stage1/`](../T02-stage1/) code into a fresh working dir; split
      into **frozen** (loader, eval, split, holdout) and **mutable** (`method/`).
- [ ] Confirm **DINOv2-B** weights cached & load offline (stage-1 confirms). To get
      the DINOv3 ablation, **accept the licence** at
      `huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m` first.
- [ ] Add a handful of **easy foreign crops** (phone/pen/tape) to
      `model/data/other_objects/` — stage-1's negatives were adversarially
      tool-like, so the reject floor is currently pessimistic/uneven.
- [ ] Set the **budget target** (`+Nk`) at launch — that's the pacing knob.
- [ ] Env: `uv venv` + `torch, transformers, pillow, scikit-learn, pycocotools,
      psutil, scipy` (stage-1's deps).

---

### The one-line launch (paste into the fresh chat)

> Run the overnight matcher auto-research in
> `docs/wayfinder/session-linker/assets/T02-stage2/program.md` as a budget-paced
> Workflow with Codex + subagents, target `+___k`. Read the program, set up the
> frozen/mutable split, and iterate champion/challenger until budget — invent new
> techniques, keep only held-out wins, write the leaderboard + findings + champion
> module as you go.
