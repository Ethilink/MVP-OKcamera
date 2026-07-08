# Import format — pre-annotated video datasets

How an **external pipeline** (outside this repo — e.g. the collection dashboard
running the NF detection model) must lay out a recorded video with model
detections so the annotation dashboard opens it as a *video project*: the human
reviews **keyframes only** (deletes false positives, adds missed instruments),
and the corrected keyframes are exported for training.

Companion to [IMPORT_FORMAT.md](IMPORT_FORMAT.md) (independent images). The
image doc's rules for categories, masks (§3), `review_status` (§4), and EXIF
still apply; this doc covers only what video adds on top — plus the sidecar for
**all-frames detections** that the dashboard must NOT display.

---

## 1. Folder layout

```
data/processed/<entry>/
├── video/
│   └── <entry>.mp4                          # the recording (H.264 MP4)
├── images/
│   ├── <entry>_f000212.jpg                  # one JPEG per keyframe — strict naming
│   └── ...
└── annotations/
    ├── annotations.json                     # COCO-VID — KEYFRAMES ONLY (see §2)
    └── metadata/
        ├── selected_frames.json             # which frames are keyframes (see §4)
        └── full_frame_detections.json       # raw model output for ALL frames (see §5)
```

- `<entry>` — single path component, becomes the project name. The video file
  should be named `<entry>.<ext>` (the loader takes the first video file in
  `video/`, but every other tool assumes the entry name).
- Accepted video extensions: `.mp4 .avi .mov .mkv .webm`. **Use H.264 MP4** —
  frames are decoded server-side with OpenCV (`CAP_PROP_POS_FRAMES` seeks) *and*
  the file is streamed to the browser for playback; H.264 MP4 is the only
  container reliably good at both. Prefer a clean constant-frame-rate encode.

**Discovery rule** (`src/models/project.py` → `Project.from_directory`): a folder
with `annotations/annotations.json` **and** a non-empty `video/` dir opens as a
*video project* — `video/` wins over `images/`, so the same folder never opens
in image mode. A folder with `video/` + `annotations/metadata/selected_frames.json`
but **no** `annotations.json` shows as a *partial* entry ("needs detection",
Open disabled).

---

## 2. `annotations.json` — keyframes only

Same COCO schema as the image case, with three additions. **Only keyframes go
in `images[]`.** The dashboard renders exactly what is listed there — this is
the mechanism that keeps the all-frames detections (§5) out of the review UI.

```jsonc
{
  "info": { "description": "NF collection dashboard v1", "date_created": "2026-07-08T10:00:00" },

  "categories": [
    { "id": 1, "name": "surgical_instrument", "supercategory": "" }
  ],

  "video": {                          // NEW — singular, top-level
    "id": 1,
    "file_name": "<entry>.mp4",       // basename inside video/
    "width": 1920,                    // must match the actual video stream
    "height": 1080,
    "frame_count": 1996,
    "fps": 23.99823660252086,         // from the real file — drives the timeline
    "source_type": "video"
  },

  "images": [
    {
      "id": 1,
      "video_id": 1,                  // NEW — FK → video.id
      "frame_number": 212,            // NEW — 0-based frame index into the video
      "file_name": "<entry>_f000212.jpg",
      "width": 1920,                  // = video dimensions (frames are 1:1, no scaling)
      "height": 1080,
      "review_status": "pending"
    }
  ],

  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "video_id": 1,                  // NEW — same FK on annotations
      "category_id": 1,
      "track_id": 1,                  // same physical instrument across keyframes ⇒ same track_id
      "bbox": [1241.0, 500.0, 332.0, 359.0],
      "area": 17527,
      "iscrowd": 0,
      "segmentation": { "size": [1080, 1920], "counts": "..." }   // optional, see image doc §3
    }
  ]
}
```

- **`frame_number`** uses cv2's frame indexing: frame 0 is the first frame.
- **`track_id`** matters more than in the image case: keyframes of the same
  video share instrument identity through it. If your model doesn't track,
  give every detection a unique `track_id` — the editor's propagation tools
  can link them later.
- Box-only import is fine (omit `segmentation`); *Detect & Segment* / SAM
  propagation can fill masks in the editor.

---

## 3. Keyframe JPEGs — filename contract

Every image record needs a matching JPEG in `images/`, named

```
<entry>_f<frame_number:06d>.jpg        # e.g. tray_0421_f000212.jpg
```

This contract is the single source of truth in `src/shared/keyframes.py` and is
shared by the pipeline, the dashboard's *add keyframe*, and the send-time
backfill. A different naming scheme orphans files.

Extract the JPEGs 1:1 from the video (no resize, quality ~95). The editor
actually serves keyframe *display* from the video file on demand, but the crop
tool edits the JPEGs in place and the training export ships them — so emit
them.

---

## 4. `annotations/metadata/selected_frames.json`

Records which frames were chosen as keyframes. Not required to open a
fully-annotated project, but the weight panel reads it for frame↔time mapping
and it is what surfaces a not-yet-annotated entry as a partial project. Emit it:

```json
{
  "selected_frames": [0, 104, 212, 319],
  "manual_review": false,
  "timestamp": "2026-07-08 10:00:00",
  "fps": 23.99823660252086,
  "selected_frames_with_time": [
    { "frame": 0, "seconds": 0.0 },
    { "frame": 104, "seconds": 4.334 }
  ]
}
```

`selected_frames` must equal the set of `frame_number`s in `annotations.json`.

---

## 5. All-frames detections — `full_frame_detections.json`

The collection dashboard runs the model on (near-)every frame, but only
keyframes are reviewed. The raw per-frame output goes in a **sidecar** the
dashboard never reads:

```
annotations/metadata/full_frame_detections.json
```

Why this location:

- The dashboard only loads `annotations.json` (plus `selected_frames.json` /
  `weight_series.json` for the weight panel). Unknown files under
  `annotations/metadata/` are ignored — the pipeline already parks its own
  sidecars there (`raw_detections.json`, `tracks.json`, `run_meta.json`).
- Send-to-registry (`src/pipeline/send.py`) copies everything except `debug/`,
  so the sidecar **survives export** to `data/sent/` and reaches downstream
  consumers. (Anything you *don't* want exported goes in `debug/` instead.)
- Do **not** add it as an extra top-level key inside `annotations.json`. It
  would survive round-trip (unknown keys are preserved), but it would bloat
  every load, save, and the 5 rotating backups the dashboard writes on each
  open.
- Do **not** name it `raw_detections.json` — that name is the internal
  pipeline's artifact with different semantics.

### Schema

```jsonc
{
  "schema_version": 1,
  "video": {                          // copy of the annotations.json video block
    "file_name": "<entry>.mp4",
    "width": 1920, "height": 1080,
    "fps": 23.99823660252086,
    "frame_count": 1996
  },
  "model": {                          // provenance — free-form but keep these
    "name": "nf-detector",
    "version": "2026-07-01",
    "conf_threshold": 0.25
  },
  "frames": [
    {
      "frame_number": 0,              // 0-based, same indexing as annotations.json
      "detections": [
        { "bbox": [1241.0, 500.0, 332.0, 359.0], "score": 0.91, "label": "surgical_instrument" }
      ]
    },
    { "frame_number": 1, "detections": [] }
  ]
}
```

- **Include every frame the model processed, even with zero detections.** An
  empty list means "model ran, found nothing" — that distinction is exactly
  what frame-mining for missed detections needs. A frame absent from `frames`
  means "not processed".
- `bbox` is `[x, y, w, h]` in absolute pixels of the **original video frame** —
  the same coordinate space as `annotations.json`.
- Keyframe detections intentionally appear in **both** files: the
  `annotations.json` copy is mutable (human deletes water bottles / cables,
  adds misses), the sidecar copy is the frozen raw model output. The diff
  between them per keyframe is itself a training signal.

---

## 6. The correction loop

Identical to the image case (image doc §4): every machine-produced keyframe
enters as `review_status: "pending"`, the human corrects and marks it
reviewed, and send-to-registry can drop unreviewed rows. Never pre-mark
machine output `"reviewed"`. The sidecar has no review state — it is never
edited by a human.

---

## 7. Gotchas

- **Dimensions.** `video.width/height`, every `images[].width/height`, and
  every RLE `size` must all equal the real video stream dimensions. Keyframes
  are extracted 1:1 — no scaling anywhere.
- **Crop tool.** The editor's video crop stores a `crop_roi` that send bakes
  into the outgoing video/JPEGs/annotations — but it knows nothing about
  `full_frame_detections.json`, whose coordinates would silently stay in the
  un-cropped space. Avoid the crop tool on imported collection sessions, or
  remap the sidecar yourself afterwards.
- **Frame indexing.** `frame_number` is the cv2 0-based index. Off-by-one here
  puts every annotation on the wrong frame with no error anywhere.
- **Unreliable frame counts.** Re-encoded MP4s can report a wrong
  `CAP_PROP_FRAME_COUNT`. Probe the file you actually ship and write those
  values; don't hardcode.
- **Validator.** `scripts/validate_import.py` currently checks the image
  contract only — it does not validate the `video` block, `frame_number`
  consistency, the keyframe filename contract, or the sidecar.
