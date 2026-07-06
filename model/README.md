# model

## What this is

Create components (e.g. by wrapping trained object detection models into a `Detector`), and pipelines and transform those into an artifact, consumable by the `app/backend`.

No model training is done here.

## Structure

```
model/
├── src/orc_model/
│   ├── components/   # Detector, Tracker, Classifier — wrap pre-trained weights/ONNX
│   └── pipelines/    # wired components → per-instrument track history
├── artifacts/         # packaged model artifact (gitignored, not code)
├── playground/        # notebooks, example scripts
├── docs/              # context/reference docs for humans and agents
└── tests/
```

## Setup

```
uv sync
```
