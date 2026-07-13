#!/usr/bin/env python3
"""Measure the delivered capture rate at requested 1080p60 (TR7 AC7).

RECORDING.md §FPS is explicit: virtual cameras routinely lie to
``CAP_PROP_FPS``, so the requested rate must be **measured**, not trusted,
before anyone flips ``capture_fps`` to 60. This spike opens the camera exactly
as the capture loop does (``CAP_PROP_FRAME_WIDTH/HEIGHT`` + ``BUFFERSIZE=1``),
additionally requests 60 fps, warms up so the pipeline settles, then times a
fixed number of blocking ``cap.read()`` calls and reports the **delivered** fps
alongside the negotiated frame shape and what the camera *claims* via
``CAP_PROP_FPS``.

The camera must be free — stop the dashboard first (it holds the device while
running). Camo only registers while the Camo app is running.

Usage:
    uv run python -m scripts.spike_fps --camera-index 3 [--frames 300] [--warmup 30] [--fps 60]

Prints a one-line summary line prefixed ``RESULT`` for easy copy into the
RECORDING.md §FPS Open items runbook. Exit 0 on a successful measurement, 1 if
the camera never yielded frames.
"""

from __future__ import annotations

import argparse
import time

import cv2

_TARGET_W, _TARGET_H = 1920, 1080


def measure(index: int, req_fps: float, frames: int, warmup: int) -> dict | None:
    """Open ``index`` at 1080p/``req_fps``, warm up, time ``frames`` reads.

    Returns a result dict (or ``None`` if the camera would not open / never
    yielded a frame). ``delivered_fps`` is ``n_ok / elapsed`` over the timed
    window, counting only successful reads.
    """
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, _TARGET_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _TARGET_H)
        cap.set(cv2.CAP_PROP_FPS, req_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Warm up: the first reads after a resolution/fps change return the
        # previous format before the pipeline settles — discard them.
        shape = None
        for _ in range(max(1, warmup)):
            ok, frame = cap.read()
            if ok and frame is not None:
                shape = tuple(int(v) for v in frame.shape)
        if shape is None:
            return None

        claimed_fps = float(cap.get(cv2.CAP_PROP_FPS))

        # Timed window: count only successful reads; a virtual camera that
        # stalls or drops still gets an honest delivered-fps (n_ok / wall time).
        n_ok = 0
        start = time.perf_counter()
        for _ in range(frames):
            ok, frame = cap.read()
            if ok and frame is not None:
                n_ok += 1
        elapsed = time.perf_counter() - start

        delivered_fps = n_ok / elapsed if elapsed > 0 else 0.0
        return {
            "index": index,
            "requested_fps": req_fps,
            "claimed_fps": claimed_fps,
            "negotiated_shape": shape,  # (H, W, C)
            "frames_requested": frames,
            "frames_read_ok": n_ok,
            "elapsed_s": elapsed,
            "delivered_fps": delivered_fps,
        }
    finally:
        cap.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scripts.spike_fps", description=__doc__)
    parser.add_argument("--camera-index", type=int, required=True, help="OpenCV index (find via scripts.find_camera)")
    parser.add_argument("--frames", type=int, default=300, help="reads to time (default 300)")
    parser.add_argument("--warmup", type=int, default=30, help="reads to discard before timing (default 30)")
    parser.add_argument("--fps", type=float, default=60.0, help="requested CAP_PROP_FPS (default 60)")
    args = parser.parse_args(argv)

    print(
        f"Spiking camera index {args.camera_index}: requesting "
        f"{_TARGET_W}x{_TARGET_H} @ {args.fps} fps, timing {args.frames} reads "
        f"(after {args.warmup} warmup reads)...\n"
    )
    result = measure(args.camera_index, args.fps, args.frames, args.warmup)
    if result is None:
        print("Camera did not open or yielded no frames. Is Camo running and the index correct?")
        return 1

    h, w, _ = result["negotiated_shape"]
    print(f"  negotiated frame shape : {w}x{h}  (H,W,C = {result['negotiated_shape']})")
    print(f"  CAP_PROP_FPS claims    : {result['claimed_fps']:.1f} fps")
    print(f"  frames read ok         : {result['frames_read_ok']} / {result['frames_requested']}")
    print(f"  elapsed                : {result['elapsed_s']:.3f} s")
    print(f"  DELIVERED fps          : {result['delivered_fps']:.1f} fps\n")
    print(
        f"RESULT index={result['index']} requested={args.fps}fps@{w}x{h} "
        f"claimed={result['claimed_fps']:.1f} delivered={result['delivered_fps']:.1f}fps"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
