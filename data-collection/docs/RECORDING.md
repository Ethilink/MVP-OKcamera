# Recording mode — technical spec

An **additive** mode on the existing data-collection dashboard: record the
camera stream as an H.264 MP4 at full camera rate, let the operator mark a
handful of **keyframes** with SPACE while recording, then run the detector over
**every frame** of the finished video in an offline post-pass. Output is a
*video project* entry per [`IMPORT_FORMAT_VIDEO.md`](IMPORT_FORMAT_VIDEO.md)
(copied from the annotation tool repo — that doc is the contract; this doc is
how we produce it).

Image-snapshot mode ([`DASHBOARD.md`](DASHBOARD.md)) is **untouched**. SPACE
means "save one still" when idle and "mark keyframe" while recording.

> **Core design decision (settled): record live, detect offline.** The detector
> runs well under 30 fps, so it cannot be in the 30 fps path. During recording
> it keeps doing exactly what it does today — overlay feedback so the operator
> knows when to press SPACE — but it is **UX only**. The authoritative
> detections (all-frames sidecar *and* keyframe annotations) come from a
> post-pass over the recorded MP4 after stop. This is not a workaround: the
> post-pass detects on the decoded post-encode frames — the identical pixels
> the annotation editor displays and the training export ships.

## What the operator experiences

1. Configure Settings as today (camera, output path, confidence).
2. **Record** button → enter an entry name → recording starts. The live overlay
   looks and feels exactly like today (same FPS — it was always
   detector-limited, and the encoder is hardware, near-zero CPU).
3. SPACE marks the frame **currently on screen** as a keyframe. Instant — it
   appends one integer, no disk I/O. Typical use: 1–20 presses per video.
4. **Stop** → recording finalizes, the post-pass starts, a progress bar shows
   `frame 812/1996`. Live inference is paused while it runs (stream may show
   raw frames or a "processing" state); starting the next recording is
   **blocked** until the job finishes.
5. Done → entry folder is a complete video project; `/validate` advisory runs.

## Runtime — two threads while recording

```
reader/encoder thread (30 fps, owns the camera while recording):
    ok, frame = cap.read()                    # every frame, full camera rate
    writer.write(frame)                       # hardware H.264 (see §Encoder)
    frame_number += 1                         # THE frame-number ground truth:
                                              # index-as-written == index in MP4
    with slot_lock:
        newest = (frame, frame_number)        # hand-off slot for inference

inference thread (unchanged shape, detector rate, skips frames freely):
    frame, frame_number = newest              # newest wins; in-between frames
    threshold = detector.confidence_threshold #   are simply never displayed
    dets = detector.predict(frame)
    overlay = render(frame.copy(), dets)
    publish Latest(frame, dets, threshold, jpeg, count, frame_number)  # NEW field
```

When **not** recording, the reader thread degrades to today's single-loop
behaviour (or simply: the inference thread reads from the slot which the reader
fills at camera rate but nothing is encoded — implementer's choice, but the
idle-mode observable behaviour of `DASHBOard.md` must not change: `/flag`,
generation pinning, ring buffer all keep working).

### The frame-number contract (the load-bearing invariant)

- The reader thread stamps each frame with its index **as it writes it into the
  encoder**. Same thread, same counter — frame 212 in memory *is* frame 212 in
  the MP4 (cv2 0-based indexing, matching the import doc).
- `Latest` carries the `frame_number` it was detected on; `/frame` returns it
  in a new `X-Frame-Number` header alongside the existing
  `X-Frame-Generation`.
- SPACE sends back the frame number of the overlay **on screen** →
  `POST /keyframe {frame_number}`. The keyframe is the frame the operator was
  looking at — never "newest at press time" (which lags by detector latency +
  reaction time, 5–15 frames).
- The post-pass seeks exactly those frame numbers for keyframe JPEGs and
  annotations.

Consequence (accepted): only frames that were actually displayed (~every
2nd–4th frame) are selectable live. Irrelevant at 1–20 keyframes on a
slow-changing tray scene, and keyframes can be adjusted later in the
annotation editor, which has the full video.

## Post-pass (background job, after stop)

Reopen the recorded MP4 and, per frame:

1. `detector.predict(frame, confidence_threshold=MINING_THRESHOLD)` →
   accumulate into `full_frame_detections.json` (§5 of the import doc):
   `bbox` + `score` + `label` only — **no masks** (they'd bloat the sidecar).
   **Every processed frame appears, including empty ones** (empty list =
   "model ran, found nothing"; that distinction is what frame-mining needs).
2. If the frame is a keyframe: filter the same detections at the **operator
   threshold** (frozen at record start, see §Thresholds), and write:
   - the JPEG, extracted 1:1 from the video, quality ~95, named
     `<entry>_f{frame_number:06d}.jpg` (§3 filename contract — exact),
   - the COCO records into `annotations.json` via the **same serialization
     path as image mode** (clip to bounds → drop degenerate boxes → RLE mask,
     empty mask ⇒ box-only → `float()` casts). Extract that per-detection
     logic out of `DatasetWriter.flag` into a shared helper rather than
     duplicating it — "keyframes look exactly like image-mode output" is a
     requirement, and one code path is how it stays true.

Then write `annotations.json` (COCO-VID: top-level `video` block, `video_id` +
`frame_number` on images, `video_id` on annotations, `review_status:
"pending"`), `selected_frames.json` (`manual_review: true` — the operator
chose these), and finalize.

The post-pass is faster than live inference per frame (no overlay render, no
JPEG encode, no stream). Ballpark: a 1-minute 30 fps video ≈ 1800 frames ≈
~3 min at 10 fps detect. Report progress via `/record/status`.

### Post-pass specifics

- **`track_id`**: the video import doc requires it on annotations. We are
  detector-only (no tracker) → give **every annotation a unique `track_id`**
  (a running counter), exactly as the import doc §2 prescribes; the editor's
  propagation tools link them later. (Image mode omits `track_id`; video mode
  must not.)
- **`video` block values**: probe the finished file (fps, frame_count, width,
  height) — do not hardcode 30/1996. The writer's own frame counter is the
  ground truth for `frame_count`; assert probe ≈ counter and prefer the
  counter if a re-encode-style mismatch appears (import doc §7 gotcha).
- **`model` block** (sidecar): `name`, `version` (from dashboard startup
  config, as in image mode — the Detector has no such field),
  `conf_threshold` = the mining threshold.
- **Failure**: a post-pass that dies must leave the entry as a valid *partial*
  project (video + `selected_frames.json`, no `annotations.json` → shows as
  "needs detection" in the annotation dashboard) and surface the error in
  `/record/status`; a **Retry** re-runs the job from the MP4 (it's all
  derivable from the video + keyframe list — idempotent by construction:
  rebuild the three JSONs and JPEGs from scratch, never append).

## Thresholds (settled)

Two thresholds, both recorded:

- **Mining threshold** for the all-frames sidecar: fixed low (default
  `0.25`, config value). The sidecar is a candidate pool — more candidates is
  the point; downstream filters on `score`.
- **Operator threshold** for keyframe `annotations.json`: the live slider
  value **frozen at `/record/start`**. What the reviewer opens matches what
  the operator saw and vetted. The slider still moves the live overlay during
  recording (UX), but does not affect the recorded outputs.

## Encoder (settled, with fallback)

- First choice: `cv2.VideoWriter(path, fourcc('a','v','c','1'), fps, (W, H))` —
  on macOS this is AVFoundation/VideoToolbox **hardware** H.264. Verify the
  writer opened (`isOpened()`) and that the output is genuinely H.264 MP4.
- Fallback if `avc1` is unavailable in the installed OpenCV build: pipe raw
  BGR frames to an `ffmpeg` subprocess stdin with `-c:v h264_videotoolbox`.
- **CFR by construction**: every captured frame is written sequentially at the
  nominal fps. If the camera actually delivers slightly under nominal (e.g.
  28 instead of 30), playback runs marginally fast and
  `selected_frames_with_time` seconds drift accordingly — accepted; the
  frame_number mapping (what annotation correctness depends on) is exact
  regardless. Do not attempt VFR/timestamp encoding (the import doc warns
  against it).
- **Crash risk (accepted for MVP)**: if the process dies before stop, the MP4
  has no moov atom and the recording is lost. One line in the UI docs; no
  mitigation now.
- `cv2.VideoWriter.release()` on stop **before** the post-pass opens the file.

## FPS — 30 vs 60 (settled: default 30, configurable)

`capture_fps` becomes a config/Settings value, default **30**. Rationale:

- 60 fps would help tracking association only marginally on slow tray scenes,
  while **doubling the post-pass time** (the actual bottleneck) and the
  sidecar size.
- Camo *claims* 1080p60, but virtual cameras routinely lie to
  `CAP_PROP_FPS` and OpenCV/AVFoundation may deliver less. Requested fps must
  be **measured**, not trusted: at recording start, the delivered rate is
  whatever the reader thread actually gets; the probed file values are what
  goes in the JSONs (see §Post-pass specifics).
- The whole pipeline is fps-agnostic (everything derives from the real file),
  so if tracking later proves motion-limited, re-record at 60 by flipping the
  setting — no code change.
- Spike task before relying on 60: time 300 `cap.read()`s at requested
  1080p60 and report the delivered rate.

**Open items** (filled by TR7's spike — do not flip the default before this
has a number): measured delivered fps at requested 1080p60 via Camo: _TBD_.

## API & state

State machine: `idle → recording → processing → idle` (`processing → failed →
processing` on retry). One recording at a time; `/record/start` while
`recording` or `processing` → 409.

```
POST /record/start    {entry_name}    -> mint <output_path>/<entry_name>/ (video-
                                         project layout), open encoder, freeze
                                         operator threshold. 409 on collision or
                                         wrong state; 422 on bad name (same rules
                                         as dataset_name).
POST /keyframe        {frame_number}  -> append to keyframe set (dedup repeat
                                         presses on the same frame). 409 if not
                                         recording; 422 if frame_number is
                                         negative or beyond frames written.
POST /record/stop                     -> finalize MP4, kick post-pass job.
POST /record/discard                  -> delete the entry folder. Allowed from
                                         recording (abort) or after a completed/
                                         failed job (reject the take).
POST /record/retry                    -> re-run a failed post-pass from the MP4.
GET  /record/status                   -> {state, entry_name, frames_written,
                                         n_keyframes, postpass: {done, total},
                                         error}
```

- `/keyframe` is trivially cheap (append an int under a lock) — safe as plain
  `def`.
- `/record/stop` returns immediately; the post-pass runs on a worker thread
  (same rule as everything else: blocking work stays off the event loop).
- Image-mode endpoints (`/flag`, `/discard`, `/validate`, `/settings`) are
  untouched; `/flag` during `recording` → 409 ("stop recording to snapshot
  stills"). `/status` gains `recording_state`.
- Frontend: Record/Stop button, keyframe counter, progress bar during
  processing, and the SPACE handler branches on recording state. Plain JS,
  no build step, as ever.

## Detector sharing (settled)

One detector instance. During `processing`, live inference **pauses** (the
post-pass owns the detector); the stream shows the last overlay or raw frames
with a "processing" badge. No lock-sharing/interleaving — it halves both sides
and complicates provenance for no operator value between takes.

## Acceptance criteria

1. A recorded entry opens in the annotation dashboard as a **video project**
   (`Project.from_directory` discovery rule: `annotations.json` + non-empty
   `video/`).
2. Every SPACE press during recording yields exactly one entry in
   `selected_frames`, one image record in `annotations.json`, and one JPEG —
   and the JPEG's pixels equal the MP4 decoded at that `frame_number`.
3. The keyframe recorded is the frame that was **displayed** at press time
   (assert via `X-Frame-Number` echo in a test with a fake camera producing
   numbered frames).
4. `full_frame_detections.json` contains **every** frame `0..frame_count-1`,
   empty frames included, coordinates in original video pixel space.
5. Keyframe annotations are byte-identical in structure to image-mode output
   for the same `sv.Detections` input (shared serialization helper), plus the
   video-mode fields (`video_id`, `frame_number`, `track_id` unique).
6. Live overlay FPS while recording ≈ live overlay FPS when idle (encoder adds
   no meaningful load).
7. Kill the post-pass mid-run → entry is a valid partial project; retry
   completes it; the result is identical to an uninterrupted run.
8. `scripts/validate_import.py` advisory passes (noting it doesn't yet check
   the video-specific contract — import doc §7).

## Gotchas (import-doc §7, restated for the coder)

- **Off-by-one is silent and fatal**: `frame_number` is cv2 0-based; one
  slipped index puts every annotation on the wrong frame with no error
  anywhere. The writer-side counter is authoritative; test with a numbered
  fake video.
- All dimensions everywhere (`video.width/height`, `images[].width/height`,
  RLE `size`) must equal the real stream dims — 1080p end-to-end keeps this
  trivially true; assert it anyway.
- Don't name the sidecar `raw_detections.json`, don't inline it into
  `annotations.json` (both explicitly forbidden by the import doc).
- Keyframe detections intentionally appear in **both** `annotations.json`
  (mutable, human-corrected) and the sidecar (frozen raw output at mining
  threshold) — the diff is itself a training signal; do not "deduplicate".

## Build plan (suggested task cut)

- **T-R1** — split reader/encoder thread out of `CaptureLoop`; `Latest` gains
  `frame_number`; idle-mode behaviour regression-tested unchanged.
- **T-R2** — encoder wrapper (`avc1` probe + ffmpeg fallback) + probe helper.
- **T-R3** — shared COCO serialization helper extracted from
  `DatasetWriter.flag`; `VideoEntryWriter` producing the four artifacts.
- **T-R4** — post-pass job runner (progress, failure → partial, retry).
- **T-R5** — API endpoints + state machine + `/status` extension.
- **T-R6** — frontend (Record/Stop, SPACE branch, progress, keyframe count).
- **T-R7** — integration: fake camera with numbered frames → record → verify
  ACs 1–5, 7 end-to-end; 1080p60 spike measurement.
