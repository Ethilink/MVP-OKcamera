---
id: T09
title: First live camera check — the whole pipeline against a real tray
type: wayfinder:task
status: open
assignee:
blocked-by: []
---

## Question

Does `load_tracker()` actually work on a live camera, end-to-end, against a real
tray of instruments — and what breaks first?

This is the **biggest un-ticketed risk on the map**, so it is now a ticket. Every
validation to date is replay (cached detections, recorded video) or fake mode
(`ScenarioTracker`). HITL — Bram runs it, the agent preps and records.

⚠️ **One thing to confirm before this ticket is trusted:** T08 cites the symptom
*"Live on the camera, the app showed 'Instrument 10' on a tray of 8"*, which
implies a live run already happened, while MAP.md says real camera mode has never
run end-to-end. The likely reconciliation is that an **informal live glance**
surfaced the "Instrument 10" symptom, but no **validated** run exists — no live
fps pin, no pre-flight cache check, no recorded acceptance. This ticket is the
validated run either way. **Bram: correct this paragraph if the informal run went
further than that.**

## Pre-flight (do this FIRST — it is the demo-day killer)

Both model downloads are lazily fetched on first use. An offline demo room with a
cold cache means `load_tracker()` **hangs at startup**, and no amount of linker
correctness saves you.

- [ ] `facebook/dinov2-base` present in `~/.cache/huggingface/hub/`
- [ ] `mobilenet_v3_small-047dcff4.pth` present in `~/.cache/torch/hub/checkpoints/`
      (Deep OC-SORT's `TorchvisionEmbedder` pulls it from download.pytorch.org)
- [ ] Verify by running with the network **off**, not by listing the directory.
- [ ] Weights path is `model/weights/checkpoint_best_regular.onnx` — **`.onnx`,
      not `.pt`**, and absolute.

## The run

```
cd app/backend && uv run python -m backend.main --camera 0 \
  --weights ../../model/weights/checkpoint_best_regular.onnx
```

Lay the tray, hit Start, let enrolment freeze, then: pick one instrument up and
put it back; remove several at once and return them out of order; put a foreign
object (phone, keys) on the table.

## What to record

1. **Live end-to-end fps.** `DEFAULT_PROCESSING_FPS = 3.0` is pinned from
   *offline replay* (2.38–3.13 fps), never from a live camera. Every `*_s` window
   in the linker and OC-SORT's `max_age` is derived from this number, so if live
   fps differs materially, coast windows stretch or shrink against wall clock.
   Pin the real number.
2. **Does the tile keep its colour/id** across a pickup and return, within
   ≤ 1.0 s? (The seam contract.)
3. **Do foreign objects stay Unknown**, or do they borrow a missing identity?
4. **Enrolment roster size** — should be exactly the tray count. The
   `workspace_max_center_y_ratio = 0.88` boundary was added because enrolment
   froze 11 identities on a tray of 8 (tripod/table objects below the mat). Live
   camera geometry may differ from the recorded takes — this is the most likely
   thing to need retuning.
5. **`tau` / `margin` behaviour.** Every guarded number comes from replay of two
   takes. Expect to retune. **Do not lower `tau` to hide a false reject** — see
   the two known fail-closed cases in `model/docs/demo-validation.md`
   (302.0 s swapped ring-handled forceps, 368.0 s flipped instrument).

## Acceptance

Resolution records: the live fps number, whether the roster froze at the tray
count, whether returns re-linked ≤ 1.0 s, whether foreign objects stayed Unknown,
and any parameter changed. Fold the fps number back into
`DEFAULT_PROCESSING_FPS` and `model/docs/demo-validation.md`.

## Note on ordering vs T08

T09 and T08 are independent — neither blocks the other. T09 validates what is
already built; T08 builds what is left. If live fps turns out materially different
from 3.0, **T09's number lands first**, because T08's binding thresholds would
otherwise be tuned against a wrong clock.
