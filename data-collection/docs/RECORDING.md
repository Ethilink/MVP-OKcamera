# Recording mode — technical spec

The **🎬 Video** side of the unified capture tool (the **📷 Image** side is
[`DASHBOARD.md`](DASHBOARD.md) — one tool, one mode toggle): record the camera
stream as an H.264 MP4 at full camera rate, let the operator mark a handful of
**keyframes** with SPACE while recording, and at Stop save each keyframe
annotation **from the live on-screen detection**. Output is a *video project*
entry per [`IMPORT_FORMAT_VIDEO.md`](IMPORT_FORMAT_VIDEO.md) (copied from the
annotation tool repo — that doc is the contract; this doc is how we produce it).

> **2026-07-09 — keyframe-only (ADR-0002).** The all-frames offline post-pass
> and the idle-draining post-pass **queue** are gone. A keyframe is now saved at
> the moment SPACE is pressed, exactly like an image-mode `/flag`, written into
> the video-project layout. See
> [`adr/0002-keyframe-only-synchronous-stop.md`](adr/0002-keyframe-only-synchronous-stop.md)
> for the why, and [`../CONTEXT.md`](../CONTEXT.md) for the vocabulary
> (**Dataset**, **Entry**, **Take**, **Base name**). ADR-0002 **supersedes**
> ADR-0001 (the queue) and the "record live, detect offline" decision below.

Unified controls: **SPACE always marks** — "save one still" in Image mode
([`DASHBOARD.md`](DASHBOARD.md)), "mark keyframe" here while recording, a no-op
hint ("Press Record first") here while idle. The Record control is a single
**toggle** button (`● Record` ↔ `■ Stop`), not two separate buttons.

> **Core design decision (settled): record live, save keyframes live.** The
> live overlay the operator watches to decide when to press SPACE **is** the
> detection that gets saved for that keyframe — pinned to the exact displayed
> frame via the capture ring, at that frame's own snapshot threshold. There is
> no second, slow consumer of the detector: the capture loop is the sole
> predictor, so there is nothing to share, park, or drain. The MP4 is saved so
> the annotation editor and any offline re-detection have the full video.

## What the operator experiences

1. Configure Settings as today (camera, output path, confidence). The name
   field relabels to "Recording session name" in Video mode — it's the
   **Base name** (`entry_base`); the same field relabels back to "Dataset
   name" on toggling to Image mode.
2. **Record** toggle (`● Record` → `■ Stop`) → recording starts against
   `output_path/videos/<entry_base>_NNN/` (server auto-suffixes the next
   zero-padded counter — max existing + 1, gaps never backfilled; see
   §Storage layout). The live overlay looks and feels exactly like today (same
   FPS — it was always detector-limited, and the encoder is hardware, near-zero
   CPU).
3. SPACE marks the frame **currently on screen** as a keyframe. The live
   detection on that frame is captured then and there (the frame is pre-encoded
   to a JPEG in memory, its detections and threshold stored). Instant; typical
   use is 1–20 presses per video. A repeat press on the same frame dedups. If
   the frame has already aged out of the ring, the mark is a non-fatal error the
   operator can just repeat.
4. **Stop** (the toggle again) → the MP4 is finalized, then the reviewed
   video-project artifacts are written **synchronously** from the stored
   keyframes (K ≈ 1–20 → ~20 `imwrite`s + 2 JSON dumps, sub-second) and the
   tool returns to idle. No background job, no progress chip.
5. Done → entry folder is a complete video project; `/validate` advisory runs.

## Storage layout

Each **Take** produces one **Entry**, auto-suffixed under the shared Output
path, always disjoint from any Image-mode **Dataset** of the same **Base
name** (vocabulary: [`../CONTEXT.md`](../CONTEXT.md)):

```
output_path/
├── images/<base>/            ← Dataset (Image mode, DASHBOARD.md)
└── videos/<base>_001/        ← Entry   (this take)
    videos/<base>_002/        ←   next take, same base — counter auto-suffixes
```

- `POST /record/start {entry_base}` scans `videos/<entry_base>_*` on disk and
  mints `<entry_base>_{NNN}` at **max existing counter + 1**, zero-padded to
  3 — never first-free (a gap like `_001`/`_003` still yields `_004`) and
  never reuses or overwrites an existing Entry. The response returns the
  resolved `entry_name`.
- Base-name validation is unchanged: single path component, no leading `.`;
  applied to `entry_base`, and the resolved `<base>_NNN` is always a valid
  component.
- A Dataset and an Entry are **never** the same folder, by construction of
  the `images/` / `videos/` split.

## Runtime — two threads while recording

```
reader/encoder thread (30 fps, owns the camera while recording):
    ok, frame = cap.read()                    # every frame, full camera rate
    writer.write(frame)                       # hardware H.264 (see §Encoder)
    frame_number += 1                         # THE frame-number ground truth:
                                              # index-as-written == index in MP4
    with slot_lock:
        newest = (frame, frame_number)        # hand-off slot for inference

inference thread (detector rate, skips frames freely — sole detector caller):
    frame, frame_number = newest              # newest wins; in-between frames
    threshold = detector.confidence_threshold #   are simply never displayed
    dets = detector.predict(frame)
    overlay = render(frame.copy(), dets)
    publish Latest(frame, dets, threshold, jpeg, count, frame_number)
```

When **not** recording, the loop degrades to today's single-loop behaviour; the
idle-mode observable behaviour of `DASHBOARD.md` (`/flag`, generation pinning,
ring buffer) is unchanged.

### The frame-number / generation contract (load-bearing)

- The reader thread stamps each frame with its index **as it writes it into the
  encoder**. Same thread, same counter — frame 212 in memory *is* frame 212 in
  the MP4 (cv2 0-based indexing, matching the import doc).
- `Latest` carries both the `generation` (published every tick, idle or
  recording) and, while recording, the `frame_number` it was detected on.
  `/frame` returns the generation in its `X-Frame-Generation` header.
- SPACE sends the **generation** of the overlay on screen → `POST /keyframe
  {generation}`. The endpoint resolves it through the ring to the exact
  displayed `Latest` and takes *that* frame's `frame_number`, detections, and
  threshold — never "newest at press time" (which lags by detector latency +
  reaction time, 5–15 frames). A generation that has aged out of the ring is a
  non-fatal 409 ("aged out — try again").

Consequence (accepted): only frames that were actually displayed (~every
2nd–4th frame) are selectable live. Irrelevant at 1–20 keyframes on a
slow-changing tray scene, and keyframes can be adjusted later in the annotation
editor, which has the full video.

## Stop — synchronous keyframe write

`/record/stop` (plain `def`, on the threadpool):

1. `capture.stop_recording()` finalizes and releases the MP4, returning the
   authoritative `frames_written`. The raw recording survives independent of
   everything below (INV-5).
2. The tool returns to **idle immediately** (state-wise) — a new recording can
   start right away; the sub-second artifact write below touches only this
   entry's own dir.
3. Probe the finished MP4 (`fps`, `width`, `height`); prefer the recording's own
   `frames_written` for `frame_count` (import doc §7 re-encode gotcha).
4. Write, in this order, via `VideoEntryWriter`:
   - **`selected_frames.json` FIRST** (`manual_review: true` — the operator
     chose these). Writing it first means a mid-write failure leaves exactly the
     contract's valid **partial** entry (video + `selected_frames.json`, no
     `annotations.json` → "needs detection").
   - one **JPEG per keyframe** — the pre-encoded bytes captured at press time,
     named `<entry>_f{frame_number:06d}.jpg` (§3 filename contract — exact),
   - the COCO records into `annotations.json` via the **same serialization
     helper as image mode** (`backend/coco.build_annotation`: clip → drop
     degenerate → RLE mask → `float()` casts), filtered at each keyframe's own
     snapshot threshold, plus the video-mode fields.
   - `finalize()` writes `annotations.json` (COCO-VID: top-level `video` block,
     `video_id` + `frame_number` on images, `video_id` + unique `track_id` on
     annotations, `review_status: "pending"`).

There is **no retry machinery** for a failed stop write. Recovery is
deliberate: re-detect the entry in the annotation editor (it has the full MP4),
or run an offline mining script against the video. Rebuilding a mini-retry is
exactly the complexity ADR-0002 removed.

### Keyframe-write specifics

- **`track_id`**: the video import doc requires it on annotations. We are
  detector-only (no tracker) → give **every annotation a unique `track_id`** (a
  running counter); the editor's propagation tools link them later. (Image mode
  omits `track_id`; video mode must not.)
- **`video` block values**: probe the finished file; the writer's own frame
  counter is the ground truth for `frame_count`.
- **Memory bound**: the keyframe frame is `imencode`'d to JPEG bytes **at press
  time**, so the stored keyframe dict is ~tens of KB per mark, not ~6 MB —
  bounded even under SPACE-spamming.

## Thresholds

**Per-keyframe snapshot threshold.** Each keyframe's annotations are filtered at
the live per-frame threshold captured **into that keyframe's snapshot at press
time** — image-mode's exact provenance rule (`snap.threshold`). The slider
moves the live overlay freely during recording; whatever value filtered the
detection the operator saw and marked is the value recorded for that keyframe.

> This **reverses** the pre-ADR-0002 rule (a single operator threshold frozen at
> `/record/start`, applied to mining detections). The reversal is intentional:
> keyframes now come from the live overlay, so they inherit the live overlay's
> exact filter — consistent with image mode, one code path. The one
> nonrecoverable loss vs the old low-threshold mining pool is keyframe
> detections that were below the live threshold at press time; weight low — the
> editor's review adds missed instruments anyway.

## Encoder (settled, with fallback)

- First choice: `cv2.VideoWriter(path, fourcc('a','v','c','1'), fps, (W, H))` —
  on macOS this is AVFoundation/VideoToolbox **hardware** H.264. Verify the
  writer opened (`isOpened()`) and that the output is genuinely H.264 MP4.
- Fallback if `avc1` is unavailable: pipe raw BGR frames to an `ffmpeg`
  subprocess stdin with `-c:v h264_videotoolbox`.
- **CFR by construction**: every captured frame is written sequentially at the
  nominal fps. If the camera delivers slightly under nominal, playback runs
  marginally fast and `selected_frames_with_time` seconds drift accordingly —
  accepted; the frame_number mapping (what annotation correctness depends on) is
  exact regardless. No VFR/timestamp encoding (the import doc warns against it).
- **Crash risk (accepted for MVP)**: if the process dies before stop, the MP4
  has no moov atom and the recording is lost. No mitigation now.
- `cv2.VideoWriter.release()` runs on stop, before the keyframe write.

## FPS — 30 vs 60 (settled: default 30, configurable)

`capture_fps` is a config/Settings value, default **30**. The whole pipeline is
fps-agnostic (everything derives from the real file), so re-record at 60 by
flipping the setting if tracking ever proves motion-limited — no code change.

- **Measured (2026-07-08, Camo, `scripts/spike_fps.py --fps 60`):** the stream
  negotiates **1920×1080**, `CAP_PROP_FPS` **claims 30**, delivered rate is
  **exactly 30.0 fps**. Camo does not provide 1080p60 through OpenCV/AVFoundation
  — requesting 60 yields 30. Do NOT flip `capture_fps` to 60.

> Historical note: the pre-ADR-0002 bottleneck was the **offline post-pass**
> (~0.6 fps → ~50× clip length). That is gone — the keyframe write is
> sub-second — so this is no longer a concern for the live tool.

## API & state

**Foreground state is `idle ↔ recording` only.** There is no background worker
and no `processing`/`failed`/drain state. One recording at a time;
`/record/start` 409s only when already `recording`.

```
POST /record/start    {entry_base}    -> mint output_path/videos/<entry_base>_NNN/
                                         (scan for max existing counter + 1),
                                         open encoder, start recording. Returns
                                         {ok, entry_name} (resolved <base>_NNN).
                                         409 if already recording / no output
                                         path / no frame yet; 422 on bad base;
                                         400 on unreadable/unwritable path.
POST /keyframe        {generation}    -> resolve the on-screen generation via the
                                         ring, capture that frame's live
                                         detection + threshold (pre-encoded JPEG),
                                         store deduped by frame_number. Returns
                                         {generation, n_keyframes}. 409 if not
                                         recording; 409 (non-fatal) if the frame
                                         aged out of the ring.
POST /record/stop                     -> finalize MP4, then synchronously write
                                         selected_frames.json + keyframe JPEGs +
                                         annotations.json. Returns
                                         {ok, frames_written}, state == idle.
POST /record/discard                  -> recording -> abort: stop+release the
                                         encoder, delete the half-written Entry
                                         (the one sanctioned delete). 409 from a
                                         fully idle machine.
GET  /record/status                   -> {state: "idle" | "recording"}
```

- `/record/*` and `/keyframe` are plain `def` — folder mint, encoder open,
  reader-thread join, and the synchronous keyframe write stay off the event
  loop.
- Image-mode endpoints (`/flag`, `/discard`, `/validate`, `/settings`) are
  untouched; `/flag` during `recording` → 409 ("stop recording to snapshot
  stills"). `/status` reports `recording_state` (`idle` / `recording`).
- Frontend: a segmented mode toggle and the Record toggle button; the SPACE
  handler branches on mode + recording state; `/keyframe` sends `generation`
  and surfaces the aged-out 409 non-fatally. No queue chip, no processing veil.
  Plain JS, no build step.

## Acceptance criteria

1. A recorded entry opens in the annotation dashboard as a **video project**
   (`Project.from_directory` discovery rule: `annotations.json` + non-empty
   `video/`).
2. Every SPACE press during recording yields exactly one entry in
   `selected_frames`, one image record in `annotations.json`, and one JPEG. The
   JPEG is the **live pre-encode frame** captured at press time (equal to the
   video decoded at that `frame_number` within JPEG re-encode tolerance — it is
   not a literal MP4 decode; settled AC2 relaxation, ADR-0002).
3. The keyframe recorded is the frame that was **displayed** at press time
   (assert via the `generation` echo in a test with a fake camera producing
   numbered frames; writer index == MP4 index).
4. Keyframe annotations are byte-identical in structure to image-mode output
   for the same `sv.Detections` input (shared serialization helper), plus the
   video-mode fields (`video_id`, `frame_number`, `track_id` unique).
5. Live overlay FPS while recording ≈ live overlay FPS when idle (encoder adds
   no meaningful load).
6. `scripts/validate_import.py` advisory passes.

> The pre-ADR-0002 "every frame in `full_frame_detections.json`" AC is dropped
> — that sidecar moved offline. If a full-frame candidate pool is needed later,
> a `scripts/mine_video.py` regenerates the §5 schema from any MP4 + the
> current best model (deferred).

## Gotchas (import-doc §7, restated)

- **Off-by-one is silent and fatal**: `frame_number` is cv2 0-based; one
  slipped index puts every annotation on the wrong frame with no error. The
  writer-side counter is authoritative; test with a numbered fake video.
- All dimensions everywhere (`video.width/height`, `images[].width/height`, RLE
  `size`) must equal the real stream dims — 1080p end-to-end keeps this
  trivially true; assert it anyway.
- The keyframe JPEG is written **verbatim** from the bytes captured at press
  time — no re-encode in the writer, so idempotent rebuild is byte-identical.
