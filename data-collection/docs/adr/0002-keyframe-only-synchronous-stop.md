# Keyframes saved live; /record/stop writes synchronously — no offline post-pass, no queue

## Status

accepted (2026-07-09) — **supersedes
[`0001-idle-draining-postpass-queue.md`](0001-idle-draining-postpass-queue.md)**
and the "record live, detect offline" core decision of
[`../RECORDING.md`](../RECORDING.md). Reviewed independently by Codex + Fable,
both "do it". Spec: [`../tasks/SIMPLIFY-keyframe-only.md`](../tasks/SIMPLIFY-keyframe-only.md).

## Decision

Stop detecting every frame after a recording. Save the video, and save each
keyframe annotation **at the moment SPACE is pressed** — from the live detection
already on screen, pinned to the exact displayed frame via the capture ring —
exactly like an image-mode `/flag`, written into the video-project layout.
`/record/stop` writes `selected_frames.json` + keyframe JPEGs + `annotations.json`
**synchronously** (sub-second for K ≈ 1–20 keyframes) and returns to idle. The
`full_frame_detections.json` sidecar is no longer produced by the live tool.

## Why

Every hard invariant ADR-0001 introduced existed downstream of **one fact**:
there was a second, slow consumer of the single shared detector — the all-frames
offline post-pass (~0.6 fps → a 1-min clip ≈ 20–50 min of detect time). That one
consumer was the entire reason for the idle-draining FIFO queue + background
worker, pause-within-one-frame / resume-from-index, halt-on-failure, retry,
drain-discard, the single-detector-owner machinery (`pause_inference` /
`resume_inference` / `predict_now` / a shared `detector_lock` /
`inference_paused`), `eta_seconds` / `detect_fps`, and the status chip.

**The sidecar moved offline, so the queue's raison d'être vanished.** Once the
tool no longer runs a slow all-frames pass, the capture loop is the *sole*
caller of `detector.predict`. There is nothing to share, park, or drain, and
that entire apparatus collapses by construction. The new stop path is bounded by
K keyframes: ~20 `imwrite`s + 2 JSON dumps, sub-second, synchronous, on the
threadpool exactly like `/flag` already runs.

This is a real simplification, not a shuffle: ~700 lines of production code and
~2000+ lines of tests deleted, replaced by a few hundred lines.

## The annotation dashboard loses nothing

Per [`../IMPORT_FORMAT_VIDEO.md`](../IMPORT_FORMAT_VIDEO.md): the discovery rule
needs only `annotations/annotations.json` + a non-empty `video/` dir — no
sidecar, no `selected_frames.json` required to open a project, so we stay
conformant. The dashboard reviews **keyframes only** and never reads
`full_frame_detections.json`; a repo-wide search found no production consumer of
that sidecar (only its producer, its tests, and the docs).

The full-frame candidate pool is not lost, just **moved offline** where it
belongs: a deferred `scripts/mine_video.py` with its own detector can regenerate
the §5 schema from any MP4 — overnight, on a GPU box, against the *current best*
model rather than whatever checkpoint was loaded at record time (frozen
detections go stale the moment you retrain). Coordinate the sidecar's removal
with whoever builds mining tooling to `IMPORT_FORMAT_VIDEO.md` §5 —
`IMPORT_FORMAT_VIDEO.md` itself is the annotation-repo contract and is **not**
edited by this change.

## Considered and rejected

- **Keep the queue but drop only the all-frames pass** — pointless: the queue
  exists *only* to schedule that pass against the shared detector. Remove the
  pass and the queue has no work.
- **Seek the MP4 to the keyframe indices at stop** (to keep AC2's literal "JPEG
  == MP4 decoded at frame_number") — rejected: it re-introduces a decode step
  for no benefit. The editor serves keyframe *display* from the MP4 anyway; the
  saved JPEG is only for the crop tool + training export, and the live
  pre-encode frame is arguably *better* training pixels than a decode of a lossy
  encode. AC2 is relaxed accordingly (settled, Bram).
- **A mini-retry for a failed synchronous stop write** — rejected: recovery is
  "re-detect in the editor / run the offline script," by design. Rebuilding
  retry is the complexity creeping back in.

## Consequences

- **Threshold semantics change on purpose.** Keyframe annotations now filter at
  the live per-frame snapshot threshold (image mode's provenance rule), not a
  single operator threshold frozen at `/record/start`. More consistent; reverses
  a settled `RECORDING.md` §Thresholds decision (stated there). One
  nonrecoverable loss: keyframe detections below the live threshold at press
  time (the old low-threshold mining pool kept them). Weight low — editor review
  adds missed instruments anyway.
- **Partial-write safety with zero machinery.** `/record/stop` writes
  `selected_frames.json` before the JPEGs/annotations, so a mid-write failure
  leaves exactly the contract's valid partial entry (video + selected_frames, no
  annotations → "needs detection"). INV-5 (the MP4 finalized before anything
  else; the only sanctioned delete is an explicit Discard) is preserved.
- **Memory bound at press time.** The keyframe frame is `imencode`'d to JPEG
  bytes when SPACE is pressed (~tens of KB), so the stored keyframe dict stays
  small even under rapid pressing.
- **No background progress to report.** The frontend loses the queue chip,
  Retry, and the processing veil; `/record/status` is just
  `{state: "idle" | "recording"}`.
