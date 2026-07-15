---
id: T01
title: Sync with Constantijn — build blessing, weights delivery, demo set
type: wayfinder:task
status: open
assignee:
blocked-by: []
---

## Question

Three things only Constantijn can settle, in one conversation (HITL — Bram
talks to him; the agent preps the agenda and records the answers):

1. **Blessing:** Claude agents under Bram build the session linker inside
   `model/` (composed in `load_tracker()`), on his turf. OK? Any design
   constraints he wants respected? (Charting decision 2026-07-14 — he was the
   assumed owner in tracker-interface's open items.)
2. **Weights delivery:** how does the trained detector weights file reach the
   demo machine (path is all a consumer needs — Drive / git-LFS / scp), and
   *when* — T04's fps pin and any live-camera linker tuning are blocked on it.
3. **Demo instrument set:** confirm the 5–8 demo instruments are visually
   distinct — no duplicate types, no near-twins (the embeddings notebook shows
   Kocher/Pean-style twins collapse to ~20% recall). Who physically picks the
   set and by when?

Resolution = the three answers recorded here, plus updating
`model/docs/tracker-interface.md` open items (linker owner, weights delivery)
to match.

---

**Input for item 3 (from T02 stage 1, 2026-07-14):** the embedding benchmark on
Bram's real captured set found a near-twin pair **`instrument1`↔`instrument2`**
(both ring-handled forceps; cosine 0.895 vs ≤0.71 for every other pair; ~70% of
all re-ID errors). The "no near-twins" assumption is currently **violated** —
but there's an engineering fix, so this may not force a set change:
**the two forceps differ ~2× in mask size, which the embedding discards but the
detector mask keeps; fusing mask-size into the matcher drops the pair's errors
to 0 and lifts overall re-ID 90%→97% (measured).** So the options for Constantijn
are: (a) keep both, add mask-size to the linker (preferred — no set change),
(b) swap one out (simplest, if size fusion proves fragile at the demo camera),
or (c) keep both and lean on the one-missing prior + multi-frame voting. See
[assets/T02-stage1/RESULTS.md](../assets/T02-stage1/RESULTS.md).
