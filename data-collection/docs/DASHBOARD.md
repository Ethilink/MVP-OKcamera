# Data-collection dashboard — technical spec

The **📷 Image** side of the unified capture tool (the **🎬 Video** side is
[`RECORDING.md`](RECORDING.md) — one tool, one mode toggle, see below): stream
Camo → run the **detector** live → operator **snapshots** bad frames → save
each as a still into a COCO-VID dataset (`review_status "pending"`) for
correction in the **separate** annotation dashboard. This tool never edits
anything; it captures stills. It consumes the model only via the `Detector`
(`model/src/orc_model/components/detector/detector.py`; design in
[`plan-first-detections.md`](../../model/docs/plan-first-detections.md)) and
emits datasets per [`IMPORT_FORMAT.md`](IMPORT_FORMAT.md).

> **2026-07-09 redesign.** This doc and `RECORDING.md` describe one tool now,
> not two — a segmented mode toggle switches the control set, and **SPACE
> always marks** (a still here, a keyframe there). See
> [`tasks/REDESIGN.md`](tasks/REDESIGN.md) for the full plan and
> [`../CONTEXT.md`](../CONTEXT.md) for the vocabulary (**Dataset**, **Entry**,
> **Base name**, ...) this doc uses precisely. The two modes still write to
> disjoint storage under the same Output path — see §Dataset layout & config.

> **Image mode is not a video recorder.** In this mode, SPACE saves **one
> still frame** (the current detector output) into the active Dataset folder
> — no start/stop, no clip, no `track_id`, just "snap this frame, it looks
> wrong." (Toggling to Video mode hands control to `RECORDING.md`'s Record
> button and keyframe model — a different mode of the same tool, not a
> separate app.)

> **Detector-only, for this first version.** It runs the RF-DETR `Detector`
> directly — boxes + masks per frame, **no tracking, no `track_id`**. A tracker is
> being built elsewhere in the repo but is not wired in here yet; when it lands it
> slots behind the same `sv.Detections` shape (adding `tracker_id`) without
> touching this tool.

## Scope (locked)

- **Watch:** live stream with overlay — box + **mask** per instrument (as the
  detector outputs them), colour by instance index, a live **count**, and a
  **confidence slider**.
- **Snapshot:** `space` (or a button) saves the current frame + the detector's
  annotations for that frame into the active dataset.
- **Settings:** camera index, output path, **dataset name** (relabels to
  "Recording session name" in Video mode — same field, same **Base name**;
  see [`../CONTEXT.md`](../CONTEXT.md)), confidence. Nothing else. (`dataset
  name` joined Settings when the tool went stateless — see *Dataset layout &
  config*; it is not a separate "start recording" step in Image mode.)
- **Not here:** any correction/editing — that's the separate annotation
  dashboard. Video recording is a different **mode** of this same tool,
  behind the mode toggle — see [`RECORDING.md`](RECORDING.md) — not a
  separate app.

## Stack (settled)

- **Backend:** Python **FastAPI** — `StreamingResponse` for MJPEG, Pydantic on
  `/confidence`. Depends on `orc_model` (uv path dep on `../model`).
- **Camera + inference:** OpenCV `VideoCapture(camera_index)`; one background
  **capture-infer thread** — `detector.predict()` called once per frame, in order.
  (The detector is stateless, but a single ordered thread keeps the stream simple
  and is what the future tracker will require anyway.)
- **Stream:** **MJPEG** (`multipart/x-mixed-replace`) — an `<img>` tag, no WebRTC.
  100–300 ms latency is fine for "SPACE when it looks wrong".
- **Frontend:** **plain HTML + vanilla JS** served by FastAPI (slider, buttons,
  `keydown` space → `/flag`). No build step, one process — `<img>` + slider +
  keydown + status poll is no state model to justify a toolchain.
- **Lives in:** `data-collection/dashboard/` (`backend/` + `static/`) — shared
  with `RECORDING.md`'s Video mode; one app, one process, one mode toggle.

## Runtime

```
capture-infer thread (loop):        # ONE resolution — capture, detect, save all at 1080p
    ok, frame = cap.read()                 # 1080p BGR (cap.set WIDTH/HEIGHT — see note); the SAVED image
    if not ok:                             # camera gone / Camo not streaming
        mark capture unhealthy; short sleep; continue   # never tight-spin, never crash
    threshold = detector.confidence_threshold           # snapshot the value USED
    dets = detector.predict(frame)         # sv.Detections in frame coords, boxes+masks, NO tracker_id
    overlay = render(frame.copy(), dets)   # draw on a COPY — never the frame /flag saves
    with lock:
        latest = (frame, dets, threshold, encode_jpeg(overlay), len(dets))
        heartbeat = <monotonic now>        # /status uses this to report capture health

GET  /stream        -> MJPEG of latest overlay jpeg (legacy; UI uses /frame)
GET  /frame?after=N -> newest overlay jpeg + `X-Frame-Generation` header; 204 if
                       gen==N (client already has it). The UI paints this so it
                       knows the generation of the frame on screen (freeze-capture).
POST /confidence    -> detector.confidence_threshold = value        (live, Pydantic)
POST /settings      -> {camera_index?, output_path, dataset_name}   set capture target;
                       dataset_name is the shared Base name (CONTEXT.md) — Image
                       mode resolves it to the Dataset at output_path/images/<dataset_name>/
                       -> REJECT (409) if that exact images/<dataset_name>/ folder
                       already exists on disk (collision) — not merely because the
                       shared images/ parent exists from a different Dataset
POST /flag          -> {generation?}: serialize the frame at that generation (the
                       EXACT frame the operator froze on) into the active dataset;
                       falls back to newest latest if omitted. 409 if no valid
                       target, or if that generation aged out of the ring buffer.
POST /discard       -> undo the most recent flag (drop its image + annotations +
                       jpeg); 409 if no dataset or nothing to discard
POST /validate      -> validate(<active dataset>) in-process -> advisory (errors, warnings)
GET  /status        -> {count, confidence, dataset_name, output_path, n_flagged, capture_health, camera_index}
```

There is **no separate start/stop step for Image-mode capture.** The tool is
stateless in this mode: the target folder lives in Settings, every SPACE
appends a still, and `/validate` runs the advisory on demand (typically once
you think a dataset is done). (Video mode — a different mode of the same
tool — does have `/record/start` / `/record/stop`; see `RECORDING.md`.)

The live overlay draws boxes **and** masks — the detector already hands us
full-frame masks, and showing them lets the operator judge **mask quality**, not
just the box (a box can look right while the mask is garbage). The expensive part
is **RLE-encoding** masks (`mask_utils.encode`) — that stays **flag-time only**,
never per stream frame.

### Threading rules (the ones that bite)

- **`/flag` is a plain `def`, not `async def`.** Its `cv2.imwrite` + RLE encode are
  blocking; on the async loop they would freeze `/stream` on every SPACE press —
  the exact worst moment. Plain `def` runs it on the threadpool, off the loop.
- **`latest` is rebound atomically under the lock, never mutated in place.**
  `/flag` grabs the tuple reference under the lock, then does its blocking work on
  that stable snapshot while the capture thread moves on. `cap.read()` returns a
  fresh array each call, so the frame `/flag` holds won't change under it — *as
  long as* `render()` draws on `frame.copy()` (above), not the raw frame.
- **`/flag` bodies serialize on their own `dataset_lock`** (also held by the
  dataset-mutating part of `/settings`), separate from the `latest` lock. Rapid
  SPACE presses land on different threadpool threads; unserialized they
  interleave `image_id`/`ann_id` assignment and the `annotations.json` rewrite —
  duplicate ids and a torn file. Flags are ~1/s at most, so holding it for the
  whole flag body is free.
- **`/flag` saves the EXACT frame the operator froze on (freeze-capture).** Earlier
  v1 saved the *newest* `latest`, which lagged the displayed frame by 100–300 ms —
  fine for most captures but wrong when detections **flicker** frame-to-frame, since
  the saved boxes could differ from the ones that triggered the SPACE. Now the UI
  drives the display by polling `/frame` (not a plain MJPEG `<img>`), so it knows
  the **generation** of the frame on screen; SPACE freezes that frame and posts
  `/flag {generation}`, and the capture loop serves it from a shallow **ring buffer**
  (`ring_size`, ~16 frames — each holds a full 1080p frame, so the window is
  deliberately a few seconds, not minutes). If the requested generation has aged
  out, `/flag` returns 409 rather than silently saving a different frame. A capture
  is held on screen ~1.5 s as confirmation with a **Discard** button (`/discard`)
  that undoes it — one keypress per capture, cheap to correct a mis-grab.
- **Provenance uses the threshold that PRODUCED the dets**, captured into `latest`
  at predict time — not the live slider value at flag time. The slider can move
  between the predict and the SPACE; recording the live value would mislabel the
  annotation.
- **One resolution — capture, detect, and save all at 1080p.** Camo streams
  **1080p (1920×1080)**. The RF-DETR `preprocess` resizes any input to **768×768**
  internally (`_rfdetr_postprocess.py:19-27`) and masks are born at **192×192**, so
  a higher capture resolution would buy **zero** detection accuracy anyway. So
  there is no capture/detect split and no scale factor: `cap.read()` gives the
  `frame`, and `predict()`, the overlay, the stream, and the saved image all share
  that one frame. (Historical note: an earlier draft assumed Camo streamed 4K and
  ran a two-resolution pipeline — capture 4K, downscale to 1080p to detect, scale
  detections back up to 4K at flag time. Camo streaming 1080p collapsed that to a
  single resolution on 2026-07-07; the saved stills are 1080p, an accepted
  tradeoff since detection quality is identical either way.)
- **Request the resolution, but stay agnostic:** OpenCV `VideoCapture` silently
  defaults to some resolution unless asked, so `cap.set(CAP_PROP_FRAME_WIDTH,
  1920)` / `CAP_PROP_FRAME_HEIGHT, 1080)` at open. The pipeline does **not** assert
  an exact resolution — it saves and annotates whatever frame the camera hands
  back — so it keeps working if Camo negotiates a different size. The only hard
  failure `start()` guards is a camera that returns **no valid frame at all**.
- **Near-live frames:** `predict()` is slower than 30 fps, so `cap.read()` outruns
  inference and OpenCV's internal buffer would serve **stale** frames. Set
  `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` (or grab-drain to the latest) so the
  stream and snapshots stay near-live. The capture-infer loop runs at detector
  throughput, not 30 fps — expected and fine for this workflow.
- **`/stream` sends each overlay once.** The MJPEG generator waits for a *new*
  `latest` (generation counter + short async sleep, or a Condition) — a naive
  `while True: yield jpeg` busy-spins the event loop and floods the socket with
  duplicate frames.
- **Single detector, single caller (2026-07-09, ADR-0002).** The capture-infer
  loop is the **only** thing that ever calls `detector.predict()`. The video
  side no longer runs an offline all-frames post-pass, so there is no second
  consumer to share the detector with — the drain queue, the detector-lock /
  `pause_inference` / `predict_now` machinery, and the "flag pauses the drain"
  rule are all gone (see
  [`adr/0002-keyframe-only-synchronous-stop.md`](adr/0002-keyframe-only-synchronous-stop.md)).
  A SPACE press in image mode always flags the live detection on the frozen
  frame; the overlay is never boxless waiting on a background job.

## `sv.Detections → COCO-VID` mapping (the crux — our side, not the model's)

Per flag, with `frame, dets, threshold, _, _ = latest` (all five slots — a
3-name unpack of the 5-tuple raises `ValueError`). **`dets` and `frame` share
one resolution** — the detector ran on the same frame we save — so boxes and
masks are written in the **frame's own pixel coordinates, unscaled**. There is
no work/full split and no scale factor to get wrong.

**Image** — write raw `frame` (BGR, no overlay) with `cv2.imwrite` to
`images/frame_{n:05d}.jpg` (`n` = per-dataset flag counter; we control the JPEG →
no EXIF trap), then:
```python
W, H = int(frame.shape[1]), int(frame.shape[0])   # frame dims

{ "id": image_id, "file_name": f"frame_{n:05d}.jpg",
  "width": W, "height": H, "review_status": "pending" }
```

**Annotations** — one per detection `i`. Note the explicit `float(...)` casts:
`dets.xyxy` is `np.float32` and the raw `json` module **cannot serialize numpy
scalars** — an uncast `bbox` raises `TypeError` on the first flag.
```python
x1, y1, x2, y2 = dets.xyxy[i]
# Clip to the frame — the detector emits boxes that run off-frame (no clamp in
# decode_predictions), and the import validator only checks w>0/h>0, not bounds.
x1 = min(max(float(x1), 0.0), W); x2 = min(max(float(x2), 0.0), W)
y1 = min(max(float(y1), 0.0), H); y2 = min(max(float(y2), 0.0), H)
bw, bh = x2 - x1, y2 - y1
if bw <= 0 or bh <= 0:
    continue                                         # drop degenerate box (clip first,
                                                     # then drop) — the box backstop

bbox = [x1, y1, bw, bh]                              # xywh, abs px in frame coords

# Encode the bool mask as-is (it's already at the frame resolution). The 192x192
# origin is the precision ceiling; it's a seed the reviewer corrects anyway.
rle = mask_utils.encode(np.asfortranarray(dets.mask[i].astype(np.uint8)))
rle["counts"] = rle["counts"].decode("utf-8")
mask_area = int(mask_utils.area(rle))

ann = {
  "id": ann_id,                                      # running counter, 1..N across ALL frames
  "image_id": image_id,                              # == this frame's image record id (== n)
  "category_id": 1,                                  # class_id 0 -> category 1
  # no track_id — the detector doesn't track; the annotation editor assigns one on edit
  "bbox": bbox,                                      # xywh, abs px (all Python floats)
  "iscrowd": 0,
  # provenance -> preserved in the format's extras bucket:
  "confidence": float(dets.confidence[i]),
  "model_version": model_version,                    # deliberate startup tag, e.g. "rfdetr-2026-07-07"
  "confidence_threshold": threshold,                 # snapshot at predict time, not live slider
}

# Box and mask are thresholded INDEPENDENTLY in the detector, so a valid box can
# carry an all-zero mask (area 0). Write it box-only in that case — the import
# format's SAM3-fill path — instead of an empty RLE the reviewer can't use.
if mask_area > 0:
    ann["segmentation"] = {"size": rle["size"], "counts": rle["counts"]}   # size == [H, W]
    ann["area"] = mask_area
else:
    ann["area"] = int(bw * bh)                       # box-only fallback; omit segmentation
```

- **Categories:** `[{"id": 1, "name": "surgical_instrument", "supercategory": ""}]`
  — single class, hardcoded (the detector is single-class; `class_id 0 → category 1`).
- **`info` block** (dataset-level provenance the format preserves round-trip):
  `{"description": <dataset_name>, "date_created": <iso8601 at folder creation>}`.
- **Two independent ID counters — do not reset them together:**
  - `image_id` increments **once per flagged frame** and equals the filename `n`.
  - `ann_id` is a **separate running counter** that accumulates across **all**
    frames and never resets per-frame. Resetting it per frame yields duplicate
    `annotations[].id`, which **corrupts editing** (`IMPORT_FORMAT.md` §5).
  - Implementation: hold `images[]` and `annotations[]` in memory, append per
    flag, assign `image_id = len(images)+1` and `ann_id = len(annotations)+1`
    *before* appending, then rewrite the file (`n` == `image_id`).
  - The live `count` (`len(dets)`) is UI-only and never written.
- **Empty frame** (detector found nothing) → image record with **zero
  annotations**. This is a first-class capture (the false negative), not an error.
- **`annotations.json` is rewritten on every flag** (cheap for hundreds of frames)
  so a crash never loses the pile. Rewrite atomically — write
  `annotations.json.tmp`, then `os.replace` — so a crash mid-write can't torch
  the previous good file either.

## Dataset layout & config

Stateless — the target folder lives in **Settings**, not a recording session.
One **Dataset** = one folder, nested under `images/` since the 2026-07-09
storage split — this is what keeps a Dataset and a video **Entry** of the
same **Base name** always disjoint (vocabulary + relationships:
[`../CONTEXT.md`](../CONTEXT.md); the Entry side lives at
`output_path/videos/<base>_NNN/`, see `RECORDING.md` §Storage layout). (No
tracker → no `reset()`, no `track_id` namespace.)
```
output_path/
└── images/<dataset_name>/
    ├── images/frame_00001.jpg ...
    └── annotations/annotations.json
```
- `output_path` + `dataset_name` are set via `/settings`; `dataset_name` is a
  single path component, no leading `.` (validator enforces both) and **is**
  the **Base name** in CONTEXT.md's vocabulary — Image mode resolves it
  straight to the Dataset name.
- **Collision → reject.** If `output_path/images/<dataset_name>/` already
  exists on disk, `/settings` rejects (409) and the operator picks a new name
  — this does **not** fire merely because the shared `images/` parent already
  holds a *different* Dataset (the nesting itself is never a collision). We
  never append into an existing folder: that would need ID counters to
  resume from the on-disk max, and a wrong resume produces duplicate ids
  that corrupt editing. **A dataset is therefore single-session — there is
  no resume-after-restart in v1** (documented limitation, not an oversight).
  Toggling between Image/Video mode mid-session does **not** trip this
  check — the writer state persists in memory for the running session and is
  reused rather than re-created (`tasks/REDESIGN.md` §"Session").
- The folder + `images/` are created lazily on the **first** flag; in-memory
  counters (`image_id`, `ann_id`, `n`, `n_flagged`) reset when `dataset_name` changes.
- **`/validate` is advisory, not a gate.** It `import`s `validate()` from
  [`scripts/validate_import.py`](scripts/validate_import.py) and calls
  `validate(<dataset>)` **in-process** — returns typed `(errors, warnings)`, no
  subprocess/stdout scraping. The folder always exists; validate just tells you if
  it's import-ready. Green = drop the folder into the annotation dashboard's
  `data/processed/` (manual — this tool doesn't write into the other app's tree).

## Failure-mode guardrails (required — no undefined behavior)

- **`/flag` with no valid target** (`output_path`/`dataset_name` unset or invalid)
  → **409**, not a silent no-op or crash, so a mis-timed SPACE is visible.
- **`/status` exposes `capture_health`.** If `cap.read()` fails or the capture
  thread dies, the stream freezes on the last frame; without a health signal the
  operator would flag a dead stream and capture nothing. `capture_health` is
  derived from the `heartbeat` timestamp (e.g. `stale` if no successful read in
  ~2 s).
- **Collision reject** (above) is the guard on the config side — there is no
  "already recording" state to guard, since the tool is stateless.

## Model / config

- **Weights are local:** `model/weights/checkpoint_best_regular.onnx` (RF-DETR
  instance-seg ONNX export; gitignored via `model/weights/*`). The dashboard takes
  the **ONNX path** + a **version tag** (for provenance) at **startup**, not in the UI.
- **`model_version` (decided):** a deliberate, date-stamped tag —
  `rfdetr-2026-07-07` for the current export — passed at startup next to the
  ONNX path. Not the filename stem: every future export is also named
  `checkpoint_best_regular`, so the stem can't tell two models apart.
- **Contract verified against the real export** — build directly on
  `Detector(weights_path)`. The export's I/O matches `decode_predictions`
  exactly: input `[1,3,768,768]`, outputs `dets [1,300,4]` / `labels [1,300,2]` /
  `masks [1,300,192,192]`. `predict()` returns `sv.Detections` (xyxy `float32`,
  bool full-frame masks, confidence `float32`, class_id `0`) with no dashboard-side
  changes. (Loaded + ran end-to-end on 2026-07-07.)

## Open items

- **Camo device index** — on Bram's Mac (2026-07-07), `scripts/find_camera.py`
  found **three** 1080p feeds at indices 0, 2, 3 (index 1 is a 720p FaceTime
  cam). Bram confirmed **index 3 is Camo** (2026-07-07). Indices can still shift
  between plug/unplug sessions, so re-run `find_camera.py` if the feed looks
  wrong — but index 3 is the working default with the phone attached.
- **Static assets were not served (fixed 2026-07-07)** — `create_app` had no
  `/static` mount, so `style.css` + `app.js` 404'd and the live app rendered as
  raw unstyled HTML with a dead script. Flagged in the T06 log, never wired;
  mounted now with a regression test (`test_static_assets_are_served`). The UI is
  styled + interactive, verified live in Chrome on the built-in webcam.
- **Real-instrument capture still pending** — the end-to-end flow (stream, apply
  settings, FLAG, validate, dataset-on-disk) is proven, but only against dark /
  empty frames. Detection quality on actual surgical instruments (T07 AC8) needs
  Bram to point Camo at instruments in decent light — a "aim the camera" step,
  not a code gap.
- **U4 hardware/visual pass — verified live on the pre-ADR-0002 build
  (2026-07-09).** Ran `backend.main` with the real RF-DETR ONNX weights on the
  CoreML EP and drove mixed one-base sessions both via the HTTP API (`u4base`)
  and through the rendered UI in Chrome (`u4visual`, Camo index 3 on real
  instruments). Confirmed and **still true today**: image-mode stills land in
  `images/<base>/` (JPEGs + `annotations.json`), video-mode clips land in
  `videos/<base>_NNN/` as valid COCO-VID projects, the **📷/🎬 mode toggle**
  relabels the control set (big button FLAG↔KEYFRAME, Record toggle Video-only,
  name field Dataset↔Recording session), and **unified SPACE** dispatches on
  mode (still in Image, keyframe while recording, "Press Record first" toast in
  Video-idle).

  > **The drain-queue half of that pass is obsolete (ADR-0002).** The same run
  > also exercised the idle-draining post-pass queue — `full_frame_detections.json`
  > sidecar, FIFO pause/resume, the top-bar queue chip, `eta_seconds`/`detect_fps`,
  > and a "flag-during-drain not reachable in the UI" discrepancy. **All of that
  > machinery was removed the next commit** (keyframe-only synchronous stop). Its
  > verification notes are dropped from this doc; the flag-during-drain gap is moot
  > because there is no drain. See
  > [`adr/0002-keyframe-only-synchronous-stop.md`](adr/0002-keyframe-only-synchronous-stop.md).
- **Keyframe-only stop path not yet hardware-verified.** The synchronous
  `/record/stop` (write `selected_frames.json` + keyframe JPEGs + `annotations.json`,
  no background job) is covered by the fake-camera e2e suite but has **not** been
  driven on real Camo hardware since it replaced the queue. A short live pass —
  record a take, mark a few keyframes, stop, confirm the entry opens as a video
  project — is the open item here.
