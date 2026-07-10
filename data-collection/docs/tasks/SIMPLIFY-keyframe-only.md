# Simplification — drop the all-frames post-pass + drain queue

**Status:** proposal (2026-07-09). Supersedes the core decision of
[`../RECORDING.md`](../RECORDING.md) ("record live, detect offline") and
[`../adr/0001-idle-draining-postpass-queue.md`](../adr/0001-idle-draining-postpass-queue.md).
Reviewed independently by Codex + Fable — both say **do it**.

## The change in one line

Stop detecting every frame after a recording. Save the video, and save a
keyframe annotation **at the moment SPACE is pressed** (using the live
detection already on screen, pinned to the exact frame via the existing ring
buffer) — exactly like image-mode `/flag`, just written into the video-project
layout.

## Why this is a real simplification, not a shuffle

Every hard invariant in today's recording code exists downstream of **one
fact**: there is a second, slow consumer of the single shared detector — the
all-frames post-pass (measured at ~0.6 fps → a 1-min clip ≈ 20–50 min of
detect time). That one consumer is the entire reason for:

- the idle-draining FIFO **queue** + single background worker,
- **pause-within-one-frame / resume-from-index**, halt-on-failure, retry,
  drain-discard,
- the **single-detector-owner** machinery: `pause_inference` /
  `resume_inference` / `predict_now` / a shared `detector_lock` /
  `inference_paused`, and the U4 "flag pauses the drain" rule,
- `eta_seconds` / `detect_fps`, the queue status chip.

Delete the all-frames pass and **all of that collapses by construction** — the
capture loop becomes the only thing that ever calls `detector.predict`. The
new stop path is bounded by K keyframes (K ≈ 1–20): ~20 `imwrite`s + 2 JSON
dumps, sub-second, synchronous, on Starlette's threadpool exactly like `/flag`
already runs.

## Does the annotation dashboard lose anything? No.

Per [`../IMPORT_FORMAT_VIDEO.md`](../IMPORT_FORMAT_VIDEO.md):

- **Discovery rule** only needs `annotations/annotations.json` + a non-empty
  `video/` dir. No sidecar, no `selected_frames.json` required to open a video
  project. We stay conformant.
- The dashboard **reviews keyframes only** and **never reads**
  `full_frame_detections.json`. That sidecar is purely for downstream
  "frame-mining for missed detections."
- **Confirmed by both reviewers:** a repo-wide search (Codex) and an
  independent grep (Fable) found **no production consumer** of the sidecar
  anywhere — only its producer, its tests, and the docs.

The sidecar capability isn't lost, just **moved offline** where it belongs: a
small `scripts/mine_video.py` with its own private detector instance can
regenerate the exact §5 schema from any MP4, overnight or on a GPU box, against
the whole corpus — and against your *current best* model, not whatever
checkpoint was loaded at record time (frozen-at-collection detections go stale
the moment you retrain). Fable's point, and it's a good one: the capture-time
sidecar was arguably the *wrong* artifact for its own use case.

**One genuinely nonrecoverable loss:** keyframe detections that were below the
live threshold at press time. The sidecar kept a low-threshold candidate pool;
live capture filters at the on-screen threshold. Weight: low — the editor's
review adds missed instruments anyway.

## What gets deleted / redesigned

### Deleted outright
| File | Lines | Fate |
|---|---|---|
| `backend/postpass.py` | 207 | **delete whole** (`PostPassJob`, the all-frames runner) |
| `backend/app.py` — queue half | ~350 of 910 | `DrainItem`, `_drain_worker` + thread, `_new_postpass_job`, `_wait_for`, `_drain_runnable_locked`, `_take_runnable_locked`, `_resume_parked_current_locked`, `/record/retry`, the drain halves of `/record/discard` + `/record/status`, the 3-phase park/reclaim dance in `/record/start`, the `RecordingController` queue fields |
| `backend/capture.py` — detector-sharing | ~65 of 483 | `pause_inference` / `resume_inference` / `predict_now` / `inference_paused` / `_detector_lock` / `detector_lock` property, and the detector-lock guards inside `_tick` / `_tick_recording` |
| `backend/video_writer.py` — sidecar | ~40 of 169 | `_sidecar_frames` accumulation in `add_frame` + the `full_frame_detections.json` write in `finalize` |

### Redesigned (small, reuses existing `/flag` plumbing)
- **`POST /keyframe {generation}`** (plain `def`): 409 if not recording;
  `snap = capture.snapshot_at(generation)` (already exists,
  `capture.py:273`); 409 "aged out" on a ring miss; store
  `rec.keyframes[snap.frame_number] = (snap.frame, snap.dets, snap.threshold)`.
  Dict keying gives dedup for free. `Latest` already pairs
  frame/dets/frame_number atomically → no pairing race.
  *(Swaps the current `X-Frame-Number` echo for the `generation` echo `/flag`
  already uses.)*
- **`POST /record/stop`** (plain `def`): `stop_recording()` → probe MP4 →
  write `selected_frames.json` **first**, then JPEGs + `annotations.json` from
  the stored tuples via a shrunken `VideoEntryWriter`. Writing
  selected-frames-first means a mid-write failure leaves exactly the contract's
  valid **partial** entry (video + selected_frames, no annotations → "needs
  detection") — INV-5 preserved with zero machinery.
- **`VideoEntryWriter`** keeps its COCO / `track_id` / filename-contract logic
  (and its tests); `add_frame` → `add_keyframe`, sidecar code removed.
- **Frontend**: delete the queue chip + retry/discard-drain controls; keyframe
  press needs a non-fatal "frame aged out" path (today it's treated as
  infallible).

### Rough totals
- **Production:** ~700 lines deleted, replaced by ~250–350 new lines. Net
  strongly negative.
- **Tests:** ~2000–2500 lines deleted (`test_postpass.py` 912 whole; the
  queue/retry/pause/eta bulk of `test_recording_api.py` 1340,
  `test_discard_cancel.py` 304, sidecar tests in `test_video_writer.py`, queue
  chip in `test_u3_frontend.py`, drain bits of `test_recording_e2e.py` /
  `test_capture.py`). **Triage test-by-test, don't rip by file** — some tests
  in these files guard contracts that *survive* (writer-index == MP4-index,
  keyframe filename contract, COCO-VID shape, generation echo, recording e2e).

## Decisions

1. **JPEG source — SETTLED (Bram, 2026-07-09): save the live pre-encode
   frame.** The frame captured at SPACE-press time is written directly as the
   keyframe JPEG — simplest, and arguably *better* training pixels than
   decode-of-lossy-encode. This **drops AC2's literal** "JPEG == MP4 decoded at
   `frame_number`" (accepted: the editor serves keyframe *display* from the MP4
   anyway; the JPEG is only for the crop tool + training export). Guard the
   SPACE-spammer: `imencode` at press time or cap the stored-frame dict
   (~6 MB/frame; fine at 20 presses, not at 2000). The rejected alternative was
   seeking the MP4 to the keyframe indices at stop to keep AC2 literal.
2. **Threshold semantics** change and should change *on purpose*: today
   keyframe annotations = mining dets filtered at the operator threshold frozen
   at `/record/start`; new model filters at the live per-frame snapshot
   threshold — image-mode's exact provenance rule. More consistent, but it
   reverses a settled `RECORDING.md` §Thresholds decision. State the reversal in
   the doc.

## Suggested landing order (both reviewers: one code pass, staged landing)

The queue and the post-pass are one inseparable feature — a half-removed queue
is worse than either end state. So:

1. **Rewrite the fake-camera e2e test** to the new contract first (SPACE → live
   tuple → stop → synchronous artifacts; ACs 1–3, 5 unchanged, AC4 dropped).
   Safety net before deleting ~2500 test lines; fits the repo's spec-first
   workflow.
2. **Single swap PR:** new `/keyframe` + `/record/stop`, delete `postpass.py` +
   the drain worker + retry + drain-halves + capture pause/lock surface +
   frontend chip; **docs + a superseding ADR-0002 in the same PR** (RECORDING.md
   inverts "detect offline"; ADR-0002's honest reason = "the sidecar moved
   offline, so the queue's raison d'être vanished," not "the queue was wrong").
3. **Alongside:** `scripts/mine_video.py` emitting the exact §5 sidecar from
   MP4 + model — this is what makes "we lose nothing" true rather than
   aspirational. Land before the docs claim mining is offline.

## Caveats worth repeating
- `IMPORT_FORMAT_VIDEO.md` is **copied from the annotation-tool repo** ("that
  doc is the contract") — the discovery rule keeps us conformant, but §5 reads
  as "the pipeline emits this." Coordinate the sidecar removal with whoever
  builds mining tooling to that doc; don't just silently stop emitting.
- Don't rebuild a mini-retry for the synchronous stop failure path — "re-detect
  in the editor, or run the offline script" is the recovery, and that's the
  point. Rebuilding retry is the complexity creeping back in.
