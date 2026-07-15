# InstrumentTracker contract

`orc_model.pipelines.tracking.InstrumentTracker` is the only model interface
used by the dashboard and demo backend. Consumers may change anything behind
this seam only by changing this contract deliberately.

## Interface

```python
class InstrumentTracker(Protocol):
    confidence: float
    def update(self, frame: np.ndarray) -> sv.Detections: ...
    def reset(self) -> None: ...
    @property
    def class_names(self) -> dict[int, str]: ...
    @property
    def model_version(self) -> str: ...
```

`load_tracker(weights_path, confidence=0.5, ...)` builds the real RF-DETR →
workspace filter → Deep OC-SORT → SessionLinker composition. A lightweight
`FakeInstrumentTracker` implements the same protocol without loading ML
dependencies.

## Input

`update(frame)` accepts one BGR `uint8` NumPy array with shape `(H, W, 3)`.
Frames arrive on one thread and in capture order. The tracker treats the array
as read-only and does not retain a mutable reference.

The stream is processed as fast as inference allows, not at camera capture
rate. On the demo M3 Max the complete uncached path measures about 3 fps.
Frame-based tracker and linker windows must be configured with that processed
rate; offline replay passes the effective sampled video rate explicitly.

## Output

`update()` returns an `sv.Detections` for the same input frame with all fields
row-aligned:

| field | required shape | meaning |
|---|---|---|
| `xyxy` | float32 `(N, 4)` | box in the input frame's pixel coordinates |
| `mask` | bool `(N, H, W)` | full-frame instance mask |
| `confidence` | float32 `(N,)` | detector confidence |
| `class_id` | int `(N,)` | key into `class_names` |
| `tracker_id` | int `(N,)` | stable session identity or Unknown raw ID |

An empty result is `sv.Detections.empty()`, never `None`. Valid frames do not
raise merely because nothing was detected.

Returned detections satisfy the configured confidence and workspace gates.
Detector boxes may extend outside the frame; consumers clamp before indexing.

## Identity semantics

The frozen Start roster defines the known physical objects for one recording.
An enrolled instrument that leaves and returns is re-emitted under its original
session ID after the link decision. A new, foreign, rejected, or ambiguous
track keeps its raw ID and is Unknown because that ID is absent from the frozen
roster.

The comparison/eligibility and open-set rules are specified in
[`linker-design.md`](./linker-design.md). Consumers must not maintain their own
alias map or retroactively rewrite IDs.

## Mutable confidence

`confidence` is a plain read/write startup setting and may be changed between
frames. It is forwarded to detector filtering. Changing it mid-recording can
spawn or retire raw tracks, so the demo should normally leave it fixed.

## Session boundary

`reset()` starts a new identity namespace. It clears OC-SORT and all
SessionLinker state while preserving loaded detector and embedding models.

## Metadata

- `class_names` is currently `{0: "surgical_instrument"}`.
- `model_version` combines the weights filename stem and a short SHA-256 hash.

## Consumer responsibilities

The model does not encode thumbnails, serve HTTP, write capture datasets, or
build reports. Consumers already own the frame and derive UI crops from
`frame + xyxy/mask/tracker_id`. The app owns roster-based usage/completeness
reporting and must not promote Unknown raw IDs into the frozen roster.

## Example

```python
tracker = load_tracker("model/weights/checkpoint_best_regular.onnx")
tracker.reset()

while streaming:
    frame = camera.read()
    detections = tracker.update(frame)
    render(frame, detections)
```
