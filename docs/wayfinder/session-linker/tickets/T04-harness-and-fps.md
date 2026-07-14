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

Resolution = harness scenarios merged, fps number pinned, and a pass/fail
readout the implementation (post-map) will be accepted against.
