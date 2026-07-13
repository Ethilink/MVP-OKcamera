#!/usr/bin/env python3
"""Probe camera indices to find the Camo device (T07 AC1).

Camo does not show up in ``system_profiler SPCameraDataType`` by name and only
registers while the Camo app is running. So we identify it the robust way: open
each OpenCV index, request **1080p** exactly as the capture loop does
(``CAP_PROP_FRAME_WIDTH/HEIGHT`` + ``BUFFERSIZE=1``), read a few frames to let
the resolution negotiate, and report ``index -> settled frame shape``. Camo
streams **1920x1080** — but so can iPhone Continuity, so resolution alone is not
enough to disambiguate: a preview JPEG is saved per working index and the
operator eyeballs which feed is the phone camera pointed at the instruments.

Usage:
    uv run python -m scripts.find_camera [--max-index 5] [--out <dir>] [--warmup 5]

Exit code 0 if at least one working camera index was found, else 1.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2

# (H, W, C) — the 1080p stream the capture loop requests. Camo settles here, but
# so can other virtual cameras: this is a hint for the printout, not a unique
# fingerprint. Final identification is by eyeballing the preview JPEG.
_CAMO_SHAPE = (1080, 1920, 3)


def probe_index(index: int, warmup: int, out_dir: Path) -> dict | None:
    """Open one index, negotiate 1080p, return a result dict or None if it won't open."""
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, _CAMO_SHAPE[1])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAMO_SHAPE[0])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # The first read(s) after a resolution change often return the previous
        # (smaller) size before the pipeline settles — warm up, keep the last.
        frame = None
        for _ in range(max(1, warmup)):
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
        if frame is None:
            return {"index": index, "shape": None, "preview": None, "is_camo": False}

        shape = tuple(int(v) for v in frame.shape)
        is_1080p = shape == _CAMO_SHAPE
        preview = out_dir / f"cam_{index}_{shape[1]}x{shape[0]}.jpg"
        cv2.imwrite(str(preview), frame)
        return {"index": index, "shape": shape, "preview": preview, "is_1080p": is_1080p}
    finally:
        cap.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts.find_camera", description=__doc__)
    parser.add_argument("--max-index", type=int, default=5, help="highest index to probe (0..N)")
    parser.add_argument("--warmup", type=int, default=5, help="reads per index before sampling")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(tempfile.gettempdir()) / "orc_camera_probe",
        help="folder for preview JPEGs",
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Probing camera indices 0..{args.max_index}  (previews -> {out_dir})\n")
    working_indices: list[int] = []
    candidate_indices: list[int] = []
    for index in range(args.max_index + 1):
        result = probe_index(index, args.warmup, out_dir)
        if result is None:
            print(f"  [{index}]  (did not open)")
            continue
        shape = result["shape"]
        if shape is None:
            print(f"  [{index}]  opened but no frame")
            continue
        tag = "  <-- 1080p (Camo-shaped)" if result["is_1080p"] else ""
        print(f"  [{index}]  {shape[1]}x{shape[0]}  preview={result['preview'].name}{tag}")
        working_indices.append(index)
        if result["is_1080p"]:
            candidate_indices.append(index)

    print()
    if not working_indices:
        print("No camera opened at all. Is the Camo app running with the phone attached?")
        return 1

    # 1080p is a hint, not a fingerprint (iPhone Continuity is 1080p too) — so the
    # operator MUST eyeball the previews to pick the phone feed. Suggest the first
    # 1080p candidate if any, else fall back to the first working index.
    if candidate_indices:
        print(f"1080p (Camo-shaped) camera(s) at index: {candidate_indices}")
    print("Open the preview JPEGs above and pick the index showing the phone's view "
          "of the instruments, then start the dashboard:")
    suggested = candidate_indices[0] if candidate_indices else working_indices[0]
    print(
        "  uv run python -m backend.main "
        "--weights ../../model/weights/checkpoint_best_regular.onnx "
        "--model-version rfdetr-2026-07-07 "
        f"--camera-index {suggested}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
