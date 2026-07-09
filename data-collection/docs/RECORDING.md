# Recording mode — technical spec

The **🎬 Video** side of the unified capture tool (the **📷 Image** side is
[`DASHBOARD.md`](DASHBOARD.md) — one tool, one mode toggle, see below): record
the camera stream as an H.264 MP4 at full camera rate, let the operator mark a
handful of **keyframes** with SPACE while recording, then run the detector over
**every frame** of the finished video in an offline post-pass. Output is a
*video project* entry per [`IMPORT_FORMAT_VIDEO.md`](IMPORT_FORMAT_VIDEO.md)
(copied from the annotation tool repo — that doc is the contract; this doc is
how we produce it).

> **2026-07-09 redesign.** This doc and `DASHBOARD.md` merged into **one
> tool** behind a segmented mode toggle (📷 Image / 🎬 Video), and the
> post-pass moved from a per-recording **blocking** job to an idle-draining
> background **queue**. See [`tasks/REDESIGN.md`](tasks/REDESIGN.md) for the
> full plan, [`../CONTEXT.md`](../CONTEXT.md) for the vocabulary (**Dataset**,
> **Entry**, **Take**, **Post-pass**, **Base name**) this doc uses precisely,
> and [`adr/0001-idle-draining-postpass-queue.md`](adr/0001-idle-draining-postpass-queue.md)
> for why the post-pass is a queue now — it **supersedes** §"Detector sharing"
> below, kept in place (not deleted) so the old constraint is visible next to
> what replaced it.

Unified controls: **SPACE always marks** — "save one still" in Image mode
([`DASHBOARD.md`](DASHBOARD.md)), "mark keyframe" here while recording, a
no-op hint ("Press Record first") here while idle. The Record control is a
single **toggle** button (`● Record` ↔ `■ Stop`), not two separate buttons.

> **Core design decision (settled): record live, detect offline.** The detector
> runs well under 30 fps, so it cannot be in the 30 fps path. During recording
> it keeps doing exactly what it does today — overlay feedback so the operator
> knows when to press SPACE — but it is **UX only**. The authoritative
> detections (all-frames sidecar *and* keyframe annotations) come from a
> post-pass over the recorded MP4 after stop. This is not a workaround: the
> post-pass detects on the decoded post-encode frames — the identical pixels
> the annotation editor displays and the training export ships.

## What the operator experiences

1. Configure Settings as today (camera, output path, confidence). The name
   field relabels to "Recording session name" in Video mode — it's the
   **Base name** (`entry_base`); the same field relabels back to "Dataset
   name" on toggling to Image mode.
2. **Record** toggle (`● Record` → `■ Stop`) → recording starts against
   `output_path/videos/<entry_base>_NNN/` (server auto-suffixes the next
   zero-padded counter — max existing + 1, gaps never backfilled; see
   §Storage layout). If an earlier take's post-pass is mid-drain, starting
   **pauses** it (within one frame) and reclaims the shared detector before
   recording begins — see §Queue model. The live overlay looks and feels
   exactly like today (same FPS — it was always detector-limited, and the
   encoder is hardware, near-zero CPU).
3. SPACE marks the frame **currently on screen** as a keyframe. Instant — it
   appends one integer, no disk I/O. Typical use: 1–20 presses per video.
4. **Stop** (the toggle again) → recording finalizes the MP4 and **enqueues**
   the post-pass job, returning to idle **immediately** — it never blocks.
   Progress shows in a **top-bar status chip** (`current done/total · N
   queued · ~ETA`), not a full-screen veil; a new recording can start right
   away and will pause the job if it's still mid-drain. See §Queue model.
5. Done → entry folder is a complete video project; `/validate` advisory
   runs. A job that **fails** halts the queue on it — the chip shows the
   error plus Retry/Discard until the operator resolves it (§Queue model).

## Storage layout (2026-07-09 split)

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
  the `images/` / `videos/` split — see `DASHBOARD.md` §Dataset layout &
  config for the Image-mode side of the same Base name.

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

## Post-pass (queued background job, after stop)

`/record/stop` finalizes the MP4 and **enqueues** a post-pass job (built at
that moment, so it carries its own `frame_count`) rather than blocking on it
— see §Queue model right below for how the job actually gets drained. What
follows is what each job does once the worker picks it up: reopen the
recorded MP4 and, per frame:

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

### Queue model (idle-draining, single shared detector)

Full rationale: [`adr/0001-idle-draining-postpass-queue.md`](adr/0001-idle-draining-postpass-queue.md).
Mechanics, in brief:

- **FIFO queue, one worker.** Each Stop enqueues a job; a single background
  worker drains the queue strictly in enqueue order, one job at a time, using
  the **one** shared detector — never two instances (doubling memory + CPU
  contention would slow both the live overlay and the post-pass).
- **Drains only while idle.** The worker predicts/writes frames **only**
  while the foreground `state == idle`. The foreground state itself
  collapses to `idle ↔ recording` — `processing`/`failed` are no longer
  `state` values; they're reported under `drain`/`error` instead (see
  §API & state).
- **Pause within one frame, resume from index, never from scratch.** Starting
  a new recording (`/record/start`) sets a cooperative flag the worker checks
  **between** frames — never mid predict/write. The worker releases the
  detector, parks the job at its current frame index, and hands the detector
  to the live overlay before recording begins. When the tool returns to idle
  with that job still current, the worker resumes it **from the parked
  index** — the output is identical to an uninterrupted run (the writer is
  deterministic; resuming continues at the parked frame number). This is
  different from **Retry** after a failure, which always rebuilds from frame
  0.
- **No data loss, ever (INV-5).** The MP4 is finalized at `/record/stop`
  *before* enqueue, so the raw recording survives independent of the
  post-pass. A failed, paused, interrupted, or killed job **never deletes its
  Entry folder** — it leaves a valid **partial** Entry (video +
  `selected_frames.json`, no `annotations.json`) that Retry rebuilds from the
  MP4. The **only** path that deletes an Entry folder is an explicit operator
  **Discard**.
- **Halt-on-failure.** When a job fails mid-run, the queue **halts on it**:
  the failed job stays as the head (`drain.current`, `done < total`), the
  top-level `error` is set to its message, and jobs queued behind it wait
  untouched (their MP4s are safe on disk — INV-5 again). The operator
  resolves it from the chip: **Retry** re-drains that Entry from frame 0 (the
  rebuild is idempotent — same inputs, same outputs); **Discard** drops the
  failed head, deletes its folder (the one sanctioned delete for a failed
  job), and advances the queue. Because the worker halts on the first
  failure, at most one Entry is ever in the failed state at a time.
- **Single-owner detector invariant (INV-3).** The live capture loop and the
  drain worker never call `detector.predict` concurrently. Mechanism: the
  capture loop skips inference and publishes the raw frame when
  `capture.inference_paused` is `True`; the drain worker sets
  `capture.pause_inference()` while it owns the detector and
  `capture.resume_inference()` when it yields or finishes. Enforced by a
  lock/flag around detector access, not two detector instances.
- **Cross-mode consequence — flagged for the post-U2/U3 e2e/hardware pass.**
  The shared-detector invariant is global, not per UI mode: whenever the tool
  is **idle** (regardless of whether Image or Video is selected in the
  toggle) *and* a job is draining, the live overlay temporarily loses
  detection boxes (raw frames only) until the queue empties or a recording
  reclaims the detector. A SPACE press in Image mode during that window still
  flags successfully, just with an empty detection list — a "the detector
  didn't run on this one" gap, not a genuine empty-frame result. The exact UI
  treatment of this window (e.g. a badge on the Image-mode stream while the
  chip is active) is a U3 frontend detail not yet nailed down here — verify
  against the built UI, don't take this doc's word for the pixels.
- **`eta_seconds`.** `remaining_frames / detect_fps`, where
  `remaining_frames = (current.total − current.done) + Σ queued[i].total` and
  `detect_fps` is a running estimate (default `0.6`, per the ADR's measured
  RF-DETR/CPU rate) exposed as a plain settable value so it's deterministic
  to test. Zero when nothing is queued or draining.

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
  rebuild the three JSONs and JPEGs from scratch, never append). Since the
  redesign this is the queue's **halt-on-failure** behavior, not a one-off —
  see §Queue model above for how the failed job blocks the queue behind it
  until Retry or Discard.

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

**Open items** (filled by TR7's spike):

- **Measured delivered fps at requested 1080p60 via Camo: 30 fps.** On Bram's
  Mac (2026-07-08, `scripts/spike_fps.py --fps 60`, 300 blocking `cap.read()`s):
  the stream negotiates **1920×1080**, `CAP_PROP_FPS` **claims 30**, and the
  delivered rate is **exactly 30.0 fps** (300 reads / 10.005 s). A dashboard-style
  open with **no** `CAP_PROP_FPS` request delivers the same 30 fps. **Conclusion:
  Camo does not provide 1080p60 through OpenCV/AVFoundation — requesting 60 yields
  30. Do NOT flip `capture_fps` to 60; it would not increase the delivered rate.**
  The pipeline stays fps-agnostic (everything derives from the probed file), so
  this is a measurement, not a blocker.
- **Real bottleneck is the post-pass, not capture.** RF-DETR ONNX on this CPU at
  1080p runs the post-pass at **~0.6 fps** (58 frames took ~102 s end-to-end;
  ~15× slower than the spec's 10 fps ballpark). Post-pass wall-time ≈ **50× the
  record duration** (30 captured fps ÷ 0.6 detect fps) — a 1-min clip ≈ ~50 min.
  A CoreML/GPU execution provider for the detector is the lever if that time
  matters; 60 fps would only double it.

## API & state

> **Superseded (2026-07-09):** `idle → recording → processing → idle`
> (`processing → failed → processing` on retry) was the pre-redesign state
> machine — kept here as a struck-through note so the "before" is visible,
> not because it's current:
> ~~State machine: `idle → recording → processing → idle`. One recording at a
> time; `/record/start` while `recording` or `processing` → 409.~~
> The endpoint table below is current.

**Foreground state is `idle ↔ recording` only.** Draining is an orthogonal
background worker (§Queue model) that runs iff `state == idle` and the queue
is non-empty; `processing` and `failed` never appear as `state` — they're
reported under `drain` / `error`. One recording at a time; `/record/start`
409s **only** when already `recording` — a draining queue does not block a
new recording, it gets paused instead (§Queue model).

```
POST /record/start    {entry_base}    -> mint output_path/videos/<entry_base>_NNN/
                                         (scan for max existing counter + 1, see
                                         §Storage layout), open encoder, freeze
                                         operator threshold. Pauses a mid-drain
                                         job and reclaims the detector first
                                         (bounded wait). Returns {entry_name}
                                         (the resolved <base>_NNN). 409 only if
                                         already recording; 422 on bad base;
                                         400 on unwritable path.
POST /keyframe        {frame_number}  -> append to keyframe set (dedup repeat
                                         presses on the same frame). 409 if not
                                         recording; 422 if frame_number is
                                         negative or beyond frames written.
POST /record/stop                     -> finalize MP4, ENQUEUE the post-pass
                                         job, return {ok, frames_written} with
                                         state == idle immediately. Never blocks.
POST /record/discard                  -> recording -> abort (as before).
                                         Otherwise -> cancel + drop the CURRENT
                                         drain job (drain.current, failed or
                                         not), delete its Entry folder, advance
                                         the queue. 409 from a fully idle
                                         machine (nothing recording, nothing
                                         draining).
POST /record/retry    {entry_name}    -> valid only when drain.current is a
                                         failed head whose entry_name matches;
                                         re-drains that Entry from frame 0.
                                         409 otherwise.
GET  /record/status                   -> {state: "idle" | "recording",
                                         drain: {current: {entry_name, done,
                                                            total} | null,
                                                 queued: [entry_name, ...],
                                                 eta_seconds: number},
                                         error: string | null}
```

- `/keyframe` is trivially cheap (append an int under a lock) — safe as plain
  `def`.
- `/record/stop` returns immediately; the queue worker (§Queue model) does
  the post-pass work off the event loop — same rule as everything else.
- Image-mode endpoints (`/flag`, `/discard`, `/validate`, `/settings`) are
  untouched; `/flag` during `recording` → 409 ("stop recording to snapshot
  stills"). `/status` continues to report `recording_state` (`idle` /
  `recording` only, post-redesign).
- Frontend (U3): a segmented mode toggle, the Record toggle button, and a
  **top-bar status chip** (`current done/total · N queued · ~ETA`, with Retry
  / Discard controls) replacing the old progress bar and full-screen veil;
  the SPACE handler branches on mode + recording state. Plain JS, no build
  step, as ever. Frozen UI contract: [`tasks/REDESIGN.md`](tasks/REDESIGN.md)
  §U3.

## Detector sharing — SUPERSEDED, see ADR-0001 + §Queue model

> ~~One detector instance. During `processing`, live inference **pauses** (the
> post-pass owns the detector); the stream shows the last overlay or raw
> frames with a "processing" badge. No lock-sharing/interleaving — it halves
> both sides and complicates provenance for no operator value between
> takes.~~
>
> This described the pre-redesign model, where the post-pass **blocked the
> whole tool** behind a `processing` state until it finished. That's gone —
> kept above (struck through) so the "why one detector instance" reasoning
> stays visible next to what replaced the blocking part of it.

**Current model:** still **one** detector instance — that constraint didn't
change — but the tool is never blocked waiting on it. See
[`adr/0001-idle-draining-postpass-queue.md`](adr/0001-idle-draining-postpass-queue.md)
for the full "why", and §Queue model above for the pause/resume mechanics
(`capture.pause_inference()` / `capture.resume_inference()`, the single-owner
invariant enforced by a lock/flag rather than two detector instances).

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

> **Queue-specific ACs** (FIFO ordering, pause-within-one-frame, resume-from-
> index, halt-on-failure, single-detector-owner) are enumerated separately in
> `tasks/REDESIGN.md` §U2 / the U2 blind-TDD brief, and exercised end-to-end
> by U4's fake-camera e2e test (§Build plan below is pre-redesign; U1–U4 in
> `tasks/REDESIGN.md` is the current task cut). Not duplicated here to avoid
> two sources of truth drifting.

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

> T-R1–T-R7 are **done** (pre-2026-07-09 redesign) and this build plan is kept
> as historical record. The active task cut — storage split (U1, done),
> idle-draining queue (U2), unified frontend (U3), integration + docs (U4,
> this doc) — lives in [`tasks/REDESIGN.md`](tasks/REDESIGN.md).
