"""Startup config + process assembly for the ORC data-collection dashboard.

The weights path and the ``model_version`` provenance tag come from CLI args at
startup — never from the UI (spec §Model / config). The routes live in
``backend.app`` (T05); ``create_app`` is re-exported here so the frozen
``from backend.main import create_app`` import keeps working. ``main()`` builds
the real ``Detector`` + ``CaptureLoop`` and hands them to the app factory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from backend.app import create_app  # noqa: F401 - re-exported (frozen import path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="backend.main",
        description="ORC data-collection dashboard",
    )
    parser.add_argument(
        "--weights",
        required=True,
        type=Path,
        help="Path to the RF-DETR ONNX weights (e.g. model/weights/checkpoint_best_regular.onnx)",
    )
    parser.add_argument(
        "--model-version",
        required=True,
        help="Provenance tag baked into every annotation, e.g. rfdetr-2026-07-07",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    # Recording mode: capture rate the encoder is opened at. Threaded into
    # app.state below.
    parser.add_argument("--capture-fps", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # Heavy deps imported inside main so `from backend.main import create_app`
    # (what the tests use) stays light and camera/onnx-free.
    import onnxruntime as ort
    import uvicorn
    from orc_model.components.detector import Detector

    from backend.capture import CaptureLoop
    from backend.render import render

    # CoreML execution provider (M3 Max GPU/Neural Engine) with a persistent
    # on-disk compile cache next to the weights; fall back to plain CPU when the
    # CoreML EP isn't available. Output parity with CPU is verified;
    # MLProgram is mandatory for this graph.
    cache_dir = Path(args.weights).parent / ".coreml_cache"
    if "CoreMLExecutionProvider" in ort.get_available_providers():
        cache_dir.mkdir(exist_ok=True)
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        provider_options = [
            {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "ALL",
                "RequireStaticInputShapes": "1",
                "ModelCacheDirectory": str(cache_dir),
            },
            {},
        ]
    else:
        providers, provider_options = None, None
    detector = Detector(args.weights, providers=providers, provider_options=provider_options)

    # T02's DatasetWriter bakes model_version in; T03's CaptureLoop owns the
    # camera and calls render() per tick. T05 assembles them here.
    def writer_factory(output_path, dataset_name):  # noqa: ANN001
        from backend.dataset_writer import DatasetWriter

        return DatasetWriter(output_path, dataset_name, args.model_version)

    capture = CaptureLoop(detector, args.camera_index, render)
    try:
        # Inside the try so a start() failure (dead camera) still hits stop() —
        # start() also releases its own handle on that path, so this is belt and
        # suspenders, but it keeps shutdown uniform for every exit.
        capture.start()
        app = create_app(detector, writer_factory=writer_factory, capture=capture)
        # Recording-mode config: threaded into app.state after build.
        app.state.capture_fps = args.capture_fps
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        capture.stop()


if __name__ == "__main__":
    main()
