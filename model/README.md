# model

## What this is

Create components (e.g. by wrapping trained object detection models into a `Detector`), and pipelines and transform those into an artifact, consumable by the `app/backend`.

No model training is done here.

## Structure

```
model/
├── src/orc_model/
│   ├── components/    # one subpackage per component — wraps pre-trained weights/ONNX
│   │   └── detector/    # Detector + model-specific pre/postprocessing (private)
│   └── pipelines/    # wired components → per-instrument track history
├── artifacts/         # packaged model artifact (gitignored, not code)
├── playground/        # notebooks, example scripts
├── docs/              # context/reference docs for humans and agents
└── tests/
```

> **`pipelines/` is currently a contract + stub, not the real tracker.**
> `pipelines/tracking.py` defines the `InstrumentTracker` interface and a
> `FakeInstrumentTracker` so `app/` can be built against a stable seam before
> real tracking exists. It is not the tracker implementation that will ship —
> that's still being designed (see `playground/trackers/`). Don't build on top
> of it as if it were production tracking logic. Full contract in
> [`docs/tracker-interface.md`](docs/tracker-interface.md).

## Setup

```
uv sync
```
