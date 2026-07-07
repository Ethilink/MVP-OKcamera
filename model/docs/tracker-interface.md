# Tracker interface — the contract for `orc_model`

The single thing the **data-collection dashboard** needs from the model: a stateful
tracker it can feed camera frames to, one at a time, that hands back detections
with a stable id per instrument. That's it. This doc is the contract; the type
lives in [`src/orc_model/pipelines/tracking.py`](../src/orc_model/pipelines/tracking.py)
as `InstrumentTracker` (Protocol) + `FakeInstrumentTracker` (stub).

> **Why a contract:** the two halves are built in parallel. You own everything
> behind `update()` — you can keep updating the detector, swap the ONNX, retune
> the tracker. As long as the output shape below holds, none of that reaches the
> dashboard. The dashboard builds against the fake stub today and swaps in your
> real tracker when the weights land.

---

## What you implement

A class matching `InstrumentTracker`:

```python
class InstrumentTracker(Protocol):
    confidence: float                       # read/write, live-adjustable
    def update(self, frame: np.ndarray) -> sv.Detections: ...
    def reset(self) -> None: ...
    @property
    def class_names(self) -> dict[int, str]: ...
    @property
    def model_version(self) -> str: ...
```

Realistically this is: **wrap the existing `Detector` + add a plain tracker
(`sv.ByteTrack` is fine) behind this interface.** Nothing fancy — the tracker
exists only to stamp an id per instrument. It never has to be *correct*; wrong
output is exactly what the human flags and fixes downstream.

---

## Input

`update(frame)` — a single **BGR** `np.ndarray`, shape `(H, W, 3)`, `uint8` —
exactly what OpenCV / Camo produces. No preprocessing on the caller's side.

> **The stream is subsampled well below camera fps.** The dashboard calls
> `update()` as fast as inference allows (~10–15 fps, not 60), on **one thread,
> in frame order**. Your tracker's motion model must tolerate a low, variable
> frame rate — don't assume 60 fps or a fixed `dt`.

## Output

An `sv.Detections` for **this frame only**, with these fields **guaranteed
populated**:

| field | type | meaning |
|---|---|---|
| `xyxy` | float32 `(N, 4)` | box in the **frame's own pixel coords** (not the 768² model input) |
| `confidence` | float32 `(N,)` | per detection |
| `class_id` | int `(N,)` | key into `class_names` |
| `tracker_id` | int `(N,)` | **stable, unique per instrument this recording** |
| `mask` | bool `(N, H, W)` | full-frame instance mask |

- Return **only** detections at or above `confidence`, and **every** returned
  detection carries a real `tracker_id`. There is no untracked / `-1` case —
  the threshold is a real setting, so the tracker only ever sees confident boxes.
- A frame with nothing to report → `sv.Detections.empty()`. Never `None`, never
  raise on a valid frame. (An empty frame is a legitimate, useful result — it's
  how "the model missed everything here" gets captured.)

`Detector.predict` already emits `xyxy` / `mask` / `confidence` / `class_id` in
exactly these shapes — your tracker's job is to **add `tracker_id` and preserve
the rest** (watch that `ByteTrack` doesn't drop `mask` / `data` on the way
through).

## `confidence` — a setting, not a slider hack

`confidence` is a plain read/write attribute. Set at startup; the dashboard may
change it between frames (its slider). It maps straight onto
`Detector.confidence_threshold`, so it's cheap — no reloading the session.
Changing it mid-recording may retire/spawn track ids; that's acceptable, every
captured frame is independent.

## `reset()` — session boundary

One recording = one `reset()` = one `tracker_id` namespace. The dashboard calls
it when a recording starts.

## `class_names` / `model_version`

- `class_names`: `{class_id: label}`, e.g. `{0: "surgical_instrument"}`. Single
  class for now; keep it a dict so multi-class is free later.
- `model_version`: any provenance string (weights tag/hash). Stashed with every
  saved frame for traceability.

---

## NOT your job

The dashboard owns all of this — do **not** build any of it into the tracker:

- Serializing to the COCO/`annotations.json` import format
- Assigning image/annotation ids, writing image files, RLE-encoding masks
- `review_status`, the output folder layout, the confidence slider UI

You emit `sv.Detections`. The mapping to the on-disk dataset is entirely the
consumer's side.

---

## How the consumer uses it

```python
tracker = load_tracker(weights_path, confidence=0.5)   # your factory
tracker.reset()                                         # start of a recording
while streaming:
    frame = camera.read()          # BGR uint8
    dets = tracker.update(frame)   # -> sv.Detections
    render(frame, dets)            # overlay, colour by tracker_id, live count
    # on flag: serialize (frame, dets) into the dataset  <-- consumer's side
```

Until your weights exist, the dashboard runs the exact same code against
`FakeInstrumentTracker()` — same interface, fake boxes. Swap it for `load_tracker(...)`
at integration. **Open item:** how you deliver the trained weights file (path is
all the dashboard needs — Drive / git-LFS / scp, your call).
