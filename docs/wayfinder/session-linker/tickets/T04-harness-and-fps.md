---
id: T04
title: Linking eval scenarios in the harness + real fps pin
type: wayfinder:prototype
status: open
assignee:
blocked-by: []
---

## Question

How do we *know* the linker works before the demo? Item 2 (scenarios) runs on
**cached detections** and is buildable now — only items 1 and 3 wait for real
weights (T01); note the partial dependency instead of blocking the whole
ticket.

1. **Real fps pin** (needs weights from T01): measure end-to-end
   `tracker.update()` fps with real weights at 1080p on the demo machine, and
   pin THAT number in the autoresearch harness (it currently replays cached
   detections at `TARGET_FPS=30`, which is free offline) — association gates
   and the linker must be tuned at the motion gaps the demo will actually see
   (tracker-interface open item, 2026-07-08).
2. **Link scenarios:** add leave/return scenarios with ground-truth identity
   to the harness, matching the real demo choreography — single instrument
   out-and-back; long exclusion (> track buffer); **several removed at once,
   one returned** (which one?); occlusion vs true departure; instrument
   swapped position on return; instrument leaves showing one face and returns
   flipped to a visually different face; a temporarily-lost raw track resumes
   under the same id but with the wrong physical object (identity validation
   must catch it); two tracks return near-simultaneously (one-to-one assignment);
   **foreign object placed on the table** while instruments are missing (must
   NOT link and must remain outside the frozen known roster). Include Start-time
   enrolment against the persistent multi-view galleries and the allowed
   session-only fallback when that initial mapping is uncertain. Score:
   fraction of returns re-emitting the original id within ≤ 1.0 s, zero wrong
   links, zero duplicate assignment, zero leaked provisional ids ≥ 1.0 s, and
   zero Unknown objects entering Usage/Completeness.
3. **Detector-on-foreign-objects evidence** (needs weights from T01): put a
   phone / pen / tape roll on the table in front of the real detector — does
   the single-class detector fire on them at all, and at what confidence? This
   fact determines how often the already-decided Unknown path is exercised (if
   they're never detected, the pipeline correctly ignores them).

**Dependency surfaced by [T03](T03-linker-design.md) (linker design, 2026-07-14):**
the harness must **replay cached *frames* (or pre-extracted crops) alongside the
cached detection boxes** — the linker crops + DINOv3-embeds from frames, so
box-only replay cannot exercise linking at all. Item 2's scenarios need frame (or
crop) payloads, not just `xyxy`/`tracker_id`. T04 also owns tuning the design's
fps-sensitive parameters (`max_age ≈ 1 s`, the reactivation re-validation gap) and
the go/no-go on keeping the stability-gated gallery refresh **on** (does
`persistent ∪ Start` already handle opposite-face returns, or is live refresh
load-bearing?). See [`model/docs/linker-design.md`](../../../../model/docs/linker-design.md).

Resolution = harness scenarios merged, fps number pinned, and a pass/fail
readout the implementation (post-map) will be accepted against.

## Progress (2026-07-15) — items 2 & 3 answered on real video; item 1 still blocked

Ran a real re-ID + open-set benchmark on Bram's 4 test videos
(`matching/data/testing/09-07-26-00{1..4}`). Full write-up + harness in
[`../assets/T04/RESULTS.md`](../assets/T04/RESULTS.md). Headlines:

- **Open-set (item 3 / the load-bearing risk): looks solved via multi-frame
  voting.** ~69 real foreign objects (incl. thin cables/pens/markers) reject at
  ~100% at a voted-cosine threshold ≈ 0.60–0.65. Single frames DO false-accept
  (hand 0.88, foreign ≤0.83 on lucky frames); voting over the return window
  collapses them. The detector fires on foreign objects (pliers ≈ 0.93), so
  rejection is the matcher's job — and it does it.
- **Re-ID (item 2): 71% top-1 on 003's teardown→re-lay (31 returns), 89%
  excluding the pituitary rongeur.** The **twin pair does NOT confuse on returns**
  (0 `instrument1`↔`instrument2`, labels human-verified) — real evidence for the
  T01 item-3 twin decision. Failure concentrates on `instrument8` (thin rongeur
  shaft); it's an **enrollment-coverage** problem (multi-view enrollment → 82% /
  rongeur 79%), not a wall → **Start-time enrollment must be multi-view per
  instrument, especially long thin ones.**
- **Still open:** item 1 (real-tracker fps pin) — blocked on the real-time tracker
  existing (stub today), not on data. Re-ID not yet scored on 001/004. Causal
  multi-view enrollment diagnosed but not yet productised. Ticket stays **open**.
