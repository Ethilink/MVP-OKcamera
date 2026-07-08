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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # Heavy deps imported inside main so `from backend.main import create_app`
    # (what the tests use) stays light and camera/onnx-free.
    import uvicorn
    from orc_model.components.detector import Detector

    from backend.capture import CaptureLoop
    from backend.render import render

    detector = Detector(args.weights)

    # T02's DatasetWriter bakes model_version in; T03's CaptureLoop owns the
    # camera and calls render() per tick. T05 assembles them here.
    def writer_factory(output_path, dataset_name):  # noqa: ANN001
        from backend.dataset_writer import DatasetWriter

        return DatasetWriter(output_path, dataset_name, args.model_version)

    capture = CaptureLoop(detector, args.camera_index, render)
    capture.start()
    try:
        app = create_app(detector, writer_factory=writer_factory, capture=capture)
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        capture.stop()


if __name__ == "__main__":
    main()
