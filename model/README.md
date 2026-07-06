# model

Offline: detection/tracking training. Produces the artifact `app/backend` consumes — frame in, boxes and labels out. No dependency on `app/`.

```
model/
├── src/orc_model/
│   ├── components/   # Detector, Tracker, Classifier
│   └── pipelines/    # wired components → per-instrument track history
├── artifacts/         # trained output (gitignored, not code)
├── playground/        # notebooks, example scripts
└── tests/
```

## Setup

```
uv sync
```
