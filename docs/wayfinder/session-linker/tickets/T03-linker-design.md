---
id: T03
title: Pin the session-linker design
type: wayfinder:grilling
status: open
assignee: bram
blocked-by: []
---

## Question

The linker's behavior, pinned decision-by-decision (HITL grilling with Bram),
producing `model/docs/linker-design.md`. Consumes T02's **stage-1** numbers
(post them as interim asset first); threshold *values* stay parameters filled
in by T02 stage 2, so this design does not wait for the photo session. The
open branches:

- **Identity model + Start-time enrolment:** distinguish disposable raw tracker
  ids from stable session ids. The objects present during the enrolment window
  after Start form a frozen known roster; later unmatched tracks are not added
  to it. Pin when enrolment begins relative to `tracker.reset()`, how long the
  table must remain still, how readiness is known, and how each initial raw
  track becomes one session identity.
- **Hybrid gallery policy:** each known session identity may combine (a)
  persistent multi-view references for that exact physical demo specimen and
  (b) fresh same-camera crops from Start-time enrolment. Pin masked vs padded
  bbox crops, number/diversity/quality of views, nearest/top-K aggregation, and
  the fallback when a Start track cannot be confidently mapped to a persistent
  specimen (it must still be usable as a session-only identity). Cached crops
  must be gathered while visible even if embedding is deferred until a death or
  birth event.
- **Gallery safety:** decide whether confidently active tracks refresh their
  session gallery. Never learn from Pending, Unknown, or ambiguous recoveries;
  one wrong match must not poison future identity.
- **Rotation/flip handling** (driven by T02's rotation-margin numbers): if raw
  embeddings degrade under rotation, canonicalize orientation from the
  instance mask (principal axis → rotate crop upright) before embedding, and
  resolve the remaining 180°/mirror ambiguity by embedding 2–4 variants and
  taking max similarity. No training on demo instruments, ever — the model
  stays frozen; robustness comes from gallery diversity + canonicalization +
  multi-frame voting.
- **Match rule (open-set, multi-candidate):** candidate set = ALL
  currently-missing tracks — the demo removes several at once, so this is the
  normal case, not an edge. Embedding score decides: best missing gallery
  above threshold wins; the one-missing prior is at most a tie-break.
  Require an absolute threshold, best-vs-second-best margin, and multi-frame
  evidence over the new track's first ~0.5 s. If several raw tracks return in
  the same decision window, use a one-to-one global assignment so two tracks
  cannot claim the same session identity. Starting values come from T02.
- **Below threshold — rejection:** new track matches no missing gallery → new
  id remains Unknown (foreign object or genuinely new instrument — the demo
  audience will place random objects on the table). Confirm no forced linking,
  ever. The product behavior is fixed by [Unknowns in the demo story — what the
  app shows for unmatched new tracks](T06-unknown-objects-ux.md); the exact
  model/backend encoding stays to be pinned here without widening the public
  tracker seam unnecessarily.
- **Tracker interplay:** ByteTrack `track_buffer` sized so brief occlusions
  never reach the linker (tracker handles them); linker only sees real
  departures. A track reactivated under the same raw id after a meaningful gap
  is still an identity-risk event: validate it against its assigned session
  gallery so a raw-tracker id switch cannot silently pass through. Strip
  `tracker_id == -1` immature tracks. Budget accounting for the ≤ 1.0 s resolve
  (collect + embed + compare + emit) at the measured fps.
- **Where it composes:** wrapper in `load_tracker()` — raw-track ids never
  cross the seam; re-emitted ids are the original, forever, no rewrites.

Resolution = the design doc committed in `model/docs/`, this ticket closed
with a gist.
