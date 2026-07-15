# Session linker

This document describes the code that currently ships in
`orc_model.pipelines`. It is the durable reference for session identity; old
research loops, implementation tickets, and agent handoffs are intentionally
not part of the runtime documentation.

## Goal

Within one recording, an instrument that leaves the workspace and returns must
resume its original integer `tracker_id`. A new or ambiguous object must remain
Unknown rather than borrowing an enrolled identity.

The pipeline is:

```text
BGR frame
  -> RF-DETR segmentation
  -> fixed-camera workspace filter
  -> Deep OC-SORT raw tracks
  -> SessionLinker stable session identities
  -> sv.Detections
```

`InstrumentTracker` is the only interface exposed to the app. Raw OC-SORT IDs,
appearance crops, and galleries remain private to the model package.

## Runtime configuration

`load_tracker()` composes the pipeline. Its important defaults are:

| setting | default | reason |
|---|---:|---|
| detector confidence | `0.50` | current operating threshold |
| processed fps | `3.0` | measured 2.87–3.13 full-pipeline fps on the M3 Max recordings |
| OC-SORT maximum age | `1.0 s` | short occlusions remain one raw track |
| linker death threshold | `1.0 s` | never shorter than OC-SORT maximum age |
| workspace maximum centre-y | `0.88 × frame height` | excludes tripod/table clutter below the blue mat |
| enrollment window | `0.5 s` | freezes the starting roster |
| return evidence window | `0.5 s`, at most 3 quality crops | resolves before the app's entry debounce |

The workspace boundary is installation-specific. Pass
`workspace_max_center_y_ratio=None` for an uncropped camera, or recalibrate it
when the overhead framing changes. It is applied before OC-SORT, so excluded
objects cannot enter the frozen roster or perturb association.

The `fps` argument means processed frames per second, not camera capture fps.
Offline replay passes `source_fps / stride`; the live app uses the measured
M3 Max default.

## Identity lifecycle

### Enrollment

The first non-empty processed frame opens the enrollment window. A raw track is
promoted into the roster when it appears in a strict majority of the window's
frames. Its session ID initially equals its raw OC-SORT ID. Up to three
quality-ranked RGB mask crops form its immutable Start gallery.

The roster never grows after enrollment. New raw tracks are Pending and then
either link to a Missing roster identity or settle as Unknown.

### Active and Missing

An enrolled identity is Active while it has an `active_raw_id`. After more than
the configured death window without that raw ID:

1. its recent quality crops replace its previous last-seen gallery;
2. the old raw-to-session mapping is retired;
3. the identity becomes Missing.

Retiring the mapping is important. If OC-SORT later recycles or reactivates the
same raw integer, it is treated as a new Pending track and must pass the matcher
again. There is no unvalidated late-reactivation shortcut.

### Pending and Unknown

A post-enrollment raw track buffers up to three quality crops. Its decision is
made when the buffer fills or the evidence window expires. Until then, its raw
ID passes through. A rejected or collision-losing track settles Unknown and
keeps its raw ID; it is not repeatedly reconsidered on every frame.

## Galleries: comparison is not eligibility

This is the central invariant.

- **Comparison galleries:** every enrolled identity with Start and/or
  last-seen views, whether Active or Missing.
- **Eligible link targets:** only identities currently Missing and possessing a
  usable comparison gallery.

The matcher always scores against the complete comparison dictionary. Only
after its open-set gates accept a winner does the linker check whether that
winner is eligible.

```python
scores = matcher.score(query, galleries=complete_roster)
winner = matcher.accept(scores)

if winner == REJECT or winner not in eligible_missing_ids:
    settle_unknown()
else:
    link_to(winner)
```

With eight enrolled instruments and one Missing instrument, the matcher still
operates at `K=8`. The seven Active identities remain valuable distractors but
can never receive a second raw track.

The comparison dictionary does not shrink during a batch. If two Pending rows
accept the same Missing identity, the higher score wins and the other row
becomes Unknown. It is deliberately not re-scored after deleting the winning
gallery: that would change SCI's geometry and push an ambiguous row toward an
unrelated second choice.

## Matcher

`ChampionMethod` uses masked DINOv2-B CLS embeddings plus a moderate size cue.
All roster gallery atoms form one normalized dictionary. A query window is
jointly reconstructed with `MultiTaskLasso`, and candidate `c` receives:

```text
similarity(c) = 1 / (1 + reconstruction_residual(c))
score(c)      = similarity(c) * SCI(c)
```

SCI is the Sparsity Concentration Index: coefficient mass concentrated on one
identity approaches 1; mass spread across the roster approaches 0. Acceptance
requires both an absolute score threshold and a best-versus-second margin.

Selected parameters:

| parameter | value |
|---|---:|
| `MultiTaskLasso alpha` | `0.0003` |
| size-feature weight | `0.5` |
| score threshold `tau` | `0.30` |
| winner margin | `0.02` |
| mask dilation | `2 px` |
| solver iteration cap | `2000` |
| query window | up to `3` crops, caller-side |

### Genuine one-gallery fallback

SCI is undefined at `K=1`. The former implementation substituted `SCI=1`,
which made the residual score too permissive and force-linked every foreign
instrument in the smoke test. A truly one-gallery roster now uses an absolute
nearest-gallery cosine gate before the normal score threshold.

The fallback cosine threshold is `0.60`. On the captured 8×15 still set this
gave 93.5% genuine accept and 98.6% foreign reject. It is a fallback for an
entire installation with one usable gallery, not the normal one-Missing path.

## Crop and gallery rules

- Frames enter as BGR; matcher crops are converted to RGB.
- Boxes are rounded and clamped for slicing.
- A crop is admitted only when its mask area is at least 200 pixels and its box
  does not touch the frame edge.
- Confidence ranks enrollment crops but does not change crop admission.
- Start views remain frozen for the recording.
- Last-seen views are replaced at each death, not appended indefinitely.
- Gallery embedding occurs only at enrollment, death, and Pending decisions;
  DINO is not run on every frame.

## Detector and compute path

The actual ONNX input is static `(1, 3, 768, 768)`. On Apple Silicon the
detector requests ONNX Runtime's CoreML execution provider with:

- `ModelFormat=MLProgram`;
- `MLComputeUnits=ALL`;
- `RequireStaticInputShapes=1`;
- a persistent compiled-model cache beside the weights;
- CPU fallback for unsupported graph fragments.

Measured on the M3 Max at 1920×1080, detector inference was about 0.33 s/frame
versus 0.84 s/frame on CPU. End-to-end uncached replay measured 2.87 and
3.13 fps on the two July 15 recordings. Cached-detection replay is only a fast
linker/tracker regression tool and must not be reported as detector throughput.

RF-DETR's second output channel is untrained. It is discarded before top-k
selection so it cannot crowd valid channel-0 queries out of the result.

## Recorded matcher evidence

The promoted SRC configuration was selected on 8 instruments × 15 captured
crops and 60 foreign crops using five seeded leave-frame-group-out splits:

| metric | result |
|---|---:|
| CV re-identification top-1 | `0.9333 ± 0.0425` |
| CV foreign rejection | `0.9733 ± 0.0533` |
| instrument1/instrument2 twin errors | `0` |
| locked holdout re-identification | `8/8` |
| locked holdout foreign rejection | `5/6` |

Ablations established the important mechanisms:

| variant | re-ID | foreign reject |
|---|---:|---:|
| full joint SRC + size + SCI | `0.9333` | `0.9733` |
| no SCI | `0.9583` | `0.6133` |
| independent per-frame sparse codes | `0.9500` | `0.9200` |
| single-frame window | `0.9143` | `0.7500` |

These are directional results from a small still-image set. The durable video
procedure and latest replay results live in
[`demo-validation.md`](./demo-validation.md).

## Known limitations

- Session galleries are recording-local; there is no catalog identity or
  cross-session history.
- The workspace boundary assumes the fixed overhead demo framing.
- Start and last-seen galleries do not yet use synthetic rotation/mirroring.
- DINO and MobileNet weights must already be cached for an offline demo.
- Matching is synchronous inside `update()`; measured event latency is within
  the one-second budget, but a future slower platform may need a worker.
- Unknown raw IDs pass through the model seam. The app must avoid adding them
  to roster-based completeness/usage reporting.

## Verification

Fast tests:

```bash
model/.venv/bin/python -m pytest model/tests -q
```

Recorded-video replay and the operator choreography are documented in
[`demo-validation.md`](./demo-validation.md).
