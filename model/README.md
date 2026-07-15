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
├── scripts/           # reproducible smoke/replay tools
├── docs/              # context/reference docs for humans and agents
└── tests/
```

`pipelines/tracking.py` contains the production composition and the lightweight
`InstrumentTracker` protocol/fake. Runtime identity behavior is documented in
[`docs/linker-design.md`](docs/linker-design.md); the public seam is in
[`docs/tracker-interface.md`](docs/tracker-interface.md).

## Setup

```
uv sync
```

Run the model tests:

```bash
uv run pytest -q
```

The two-video demo procedure is in
[`docs/demo-validation.md`](docs/demo-validation.md).
