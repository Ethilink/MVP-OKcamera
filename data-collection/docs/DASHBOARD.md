# Data-collection dashboard — technical spec

The capture tool: stream Camo → run the model live → operator flags bad frames →
save as COCO-VID (`pending`) for correction in the annotation dashboard. It never
edits anything; it captures. Consumes the model only via `InstrumentTracker`
(`model/docs/tracker-interface.md`); emits datasets per
[`IMPORT_FORMAT.md`](IMPORT_FORMAT.md).

## Scope (locked)

- **Watch:** live stream with overlay — box + mask outline per instrument,
  **colour by `tracker_id`**, a live **count**, and a **confidence slider**.
- **Flag:** `space` (or a button) saves the current frame + the model's
  annotations for that frame into the active dataset.
- **Settings:** camera index, output path, confidence. Nothing else.
- **Not here:** any correction/editing — that's the separate annotation dashboard.

## Stack (proposed — confirm)

- **Backend:** Python **FastAPI**. Depends on `orc_model` (uv path dep on `../model`).
- **Camera + inference:** OpenCV `VideoCapture(camera_index)`; one background
  **capture-infer thread** (satisfies the contract: `update()` called once per
  frame, in order, single thread).
- **Stream:** **MJPEG** (`multipart/x-mixed-replace`) — an `<img>` tag, no WebRTC.
- **Frontend:** **plain HTML + vanilla JS** served by FastAPI (slider, buttons,
  `keydown` space → `/flag`). No build step, one process. *(Alternative: Vite+React
  — heavier; not worth it for one internal screen.)*
- **Lives in:** `data-collection/dashboard/` (`backend/` + `static/`).

## Runtime

```
capture-infer thread (loop):
    ok, frame = cap.read()                 # BGR uint8
    dets = tracker.update(frame)           # sv.Detections (+tracker_id)
    overlay = render(frame, dets)          # boxes+labels, colour by tracker_id
    latest = (frame, dets, encode_jpeg(overlay), count=len(dets))   # under lock

GET  /stream        -> MJPEG of latest overlay jpeg
POST /confidence    -> tracker.confidence = value        (live)
POST /recording/start {name} -> tracker.reset(); open dataset <name>
POST /flag          -> serialize latest (frame, dets) into the dataset
POST /recording/stop-> finalize; run validate_import.py as the gate
GET  /status        -> {count, confidence, recording, dataset, n_flagged}
```

Overlay is **boxes-only live** for speed; masks are encoded to RLE **only at flag
time** (the per-instance full-frame masks are the main cost — never render them
per stream frame).

## `sv.Detections → COCO-VID` mapping (the crux — our side, not the model's)

Per flag, with `dets = latest.dets` and `img = latest.frame`:

**Image** — write raw `img` (BGR, no overlay) with `cv2.imwrite` to
`images/frame_{n:05d}.jpg` (we control the JPEG → no EXIF trap), then:
```python
{ "id": image_id, "file_name": f"frame_{n:05d}.jpg",
  "width": img.shape[1], "height": img.shape[0], "review_status": "pending" }
```

**Annotations** — one per detection `i`:
```python
x1, y1, x2, y2 = dets.xyxy[i]                       # clip to [0,w]/[0,h]
rle = mask_utils.encode(np.asfortranarray(dets.mask[i].astype(np.uint8)))
rle["counts"] = rle["counts"].decode("utf-8")
{
  "id": ann_id,                                     # global counter
  "image_id": image_id,
  "category_id": 1,                                 # class_id 0 -> category 1
  "track_id": int(dets.tracker_id[i]),
  "bbox": [x1, y1, x2 - x1, y2 - y1],               # xywh, abs px
  "area": int(mask_utils.area(rle)),
  "iscrowd": 0,
  "segmentation": {"size": rle["size"], "counts": rle["counts"]},
  # provenance -> preserved in the format's extras bucket:
  "confidence": float(dets.confidence[i]),
  "model_version": tracker.model_version,
  "confidence_threshold": tracker.confidence,
}
```

- **Categories:** `[{"id": 1, "name": "surgical_instrument", "supercategory": ""}]`,
  built from `tracker.class_names`.
- **Empty frame** (model missed everything) → image record with **zero
  annotations**. This is a first-class capture (the false negative), not an error.
- **`annotations.json` is rewritten on every flag** (cheap for hundreds of frames)
  so a crash never loses the pile.

## Dataset layout & session

One recording = one `tracker.reset()` = one folder = one `track_id` namespace:
```
<output_path>/<dataset_name>/
├── images/frame_00001.jpg ...
└── annotations/annotations.json
```
- `dataset_name`: operator enters it at **start recording** (fallback:
  timestamp). Single path component, no leading `.`.
- On **stop**, run `validate_import.py <dataset>` — green = drop into the
  annotation dashboard's `data/processed/`.

## Open items

- **Frontend choice** — plain HTML/JS (rec) vs React.
- **Weights hand-off** — from Constantijn; dashboard needs only the ONNX path.
- **Camo device index** — discover on Bram's Mac.
