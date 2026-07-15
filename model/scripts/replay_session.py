#!/usr/bin/env python
"""Replay a recorded take through the real ``load_tracker()`` pipeline.

Feeds a video's frames (subsampled) through ONE `load_tracker()` instance —
real RF-DETR ONNX -> real Deep OC-SORT -> real SessionLinker — and writes a
per-frame trace plus the linker's own INFO logs for inspection and scoring.

fps semantics
-------------
`--stride N` takes every Nth frame of a 30 fps recording, so the tracker's
effective feed rate is `30/N` fps. That value is passed to `load_tracker(fps=)`
so every `*_s` window (death, enrolment, evidence, OC-SORT max_age) maps onto
the same VIDEO wall-clock it would live-camera. A stride-6 feed declared as
12 fps would stretch a "1.0 s" coast to 2.5 s of video time.

Detection cache
---------------
The detector is ~0.33 s/frame (CoreML) and dominates the run, so every
detection it produces is collected and written to `<out>.dets.npz` when the
run completes. `--from-cache` then replays tracker+linker over those exact
detections, so the
matcher/linker can be re-run (e.g. after an open-set gate change) without
re-burning ~25 min of inference. The cached run is faithful: identical
detections in, real OC-SORT and real SessionLinker still doing the work.

Run:
    model/.venv/bin/python model/scripts/replay_session.py \
        --video matching/data/testing/15-07-26-001/videos/15-07-26-001.mp4 \
        --out /tmp/orc-take-a
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = REPO / "model/weights/checkpoint_best_regular.onnx"


# -- exact, compact mask storage ----------------------------------------
# Masks are sparse (~5-14k of 2.07M px) but NOT bbox-contained (up to ~411 px
# spill measured), so crop-to-bbox would silently lose pixels. RLE over the
# flat mask is exact and ~100x smaller than packbits of the full frame.
def rle_encode(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = mask.ravel().astype(np.int8)
    d = np.diff(flat)
    starts = (np.flatnonzero(d == 1) + 1).astype(np.uint32)
    ends = (np.flatnonzero(d == -1) + 1).astype(np.uint32)
    if flat[0]:
        starts = np.concatenate([[0], starts]).astype(np.uint32)
    if flat[-1]:
        ends = np.concatenate([ends, [flat.size]]).astype(np.uint32)
    return starts, (ends - starts).astype(np.uint32)


def rle_decode(starts: np.ndarray, lengths: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    flat = np.zeros(shape[0] * shape[1], dtype=bool)
    for s, ln in zip(starts, lengths):
        flat[s : s + ln] = True
    return flat.reshape(shape)


class RecordingDetector:
    """Wraps the real Detector, recording every prediction. Real detections in,
    real detections out — the tracker cannot tell the difference."""

    def __init__(self, inner, sink: dict):
        self._inner = inner
        self._sink = sink
        self.session = inner.session

    def predict(self, image, confidence_threshold=None):
        det = self._inner.predict(image, confidence_threshold=confidence_threshold)
        i = self._sink["n"]
        self._sink[f"xyxy_{i}"] = det.xyxy.astype(np.float32)
        self._sink[f"conf_{i}"] = det.confidence.astype(np.float32)
        self._sink[f"cls_{i}"] = det.class_id.astype(np.int16)
        starts, lengths, offsets = [], [], [0]
        for m in (det.mask if det.mask is not None else []):
            s, ln = rle_encode(m)
            starts.append(s)
            lengths.append(ln)
            offsets.append(offsets[-1] + len(s))
        self._sink[f"rs_{i}"] = np.concatenate(starts) if starts else np.zeros(0, np.uint32)
        self._sink[f"rl_{i}"] = np.concatenate(lengths) if lengths else np.zeros(0, np.uint32)
        self._sink[f"ro_{i}"] = np.array(offsets, np.uint32)
        self._sink["n"] = i + 1
        return det


class CachedDetector:
    """Replays recorded detections in feed order. `predict()` ignores the image."""

    def __init__(self, cache, shape):
        self._c = cache
        self._shape = shape
        self._i = 0
        self.session = None

    def predict(self, image, confidence_threshold=None):
        i = self._i
        self._i += 1
        xyxy = self._c[f"xyxy_{i}"]
        if len(xyxy) == 0:
            return sv.Detections.empty()
        off = self._c[f"ro_{i}"]
        rs, rl = self._c[f"rs_{i}"], self._c[f"rl_{i}"]
        masks = np.stack([
            rle_decode(rs[off[k] : off[k + 1]], rl[off[k] : off[k + 1]], self._shape)
            for k in range(len(xyxy))
        ])
        return sv.Detections(
            xyxy=xyxy, mask=masks,
            confidence=self._c[f"conf_{i}"], class_id=self._c[f"cls_{i}"].astype(int),
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--confidence", type=float, default=0.5)
    ap.add_argument("--from-cache", action="store_true")
    ap.add_argument("--cache", default=None,
                    help="explicit detection-cache path (default: <out>.dets.npz), so a re-run "
                         "can write a new trace while reading an existing take's detections")
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N sampled frames")
    ap.add_argument("--log-gate", action="store_true",
                    help="capture per-row linker scores and K=1 cosine-gate decisions")
    ap.add_argument("--cos-tau", type=float, default=None, help="override the K=1 gate threshold")
    ap.add_argument(
        "--instruments-dir",
        default=None,
        help="persistent specimen photos the linker binds to (default: the shipped "
             "model/data/instruments). Pass a directory holding a SUBSET of the "
             "instrument{N} folders to force a partial bind, or 'none' to disable "
             "binding entirely -- both are how the T08 atom-count asymmetry is measured.",
    )
    ap.add_argument(
        "--workspace-max-center-y-ratio",
        type=float,
        default=0.88,
        help="fixed-camera workspace boundary; pass 1.0 to disable spatial cropping",
    )
    args = ap.parse_args()

    if args.stride < 1:
        ap.error("--stride must be at least 1")

    video = Path(args.video)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    eff_fps = src_fps / args.stride

    print(f"{video.name}: {n_frames} frames @ {src_fps:.1f} fps ({n_frames/src_fps:.0f}s), {w}x{h}")
    print(f"stride={args.stride} -> effective feed {eff_fps:.2f} fps -> load_tracker(fps={eff_fps:.2f})")

    # -- linker log capture, stamped with the frame being processed --------
    logs: list[dict] = []
    state = {"frame": -1, "t": 0.0}

    class Capture(logging.Handler):
        def emit(self, record):
            logs.append({
                "video_frame": state["frame"],
                "t_video_s": round(state["t"], 2),
                "msg": record.getMessage(),
            })

    lg = logging.getLogger("orc_model.session_linker")
    lg.setLevel(logging.DEBUG if args.log_gate else logging.INFO)
    lg.addHandler(Capture())
    # T08: how many specimens actually embedded, and the per-specimen view counts.
    plg = logging.getLogger("orc_model.persistent_gallery")
    plg.setLevel(logging.INFO)
    plg.addHandler(Capture())
    if args.log_gate:
        mlg = logging.getLogger("orc_model.matching")
        mlg.setLevel(logging.DEBUG)
        mlg.addHandler(Capture())

    from orc_model.pipelines.tracking import load_tracker

    t0 = time.monotonic()
    load_kwargs = {}
    if args.instruments_dir is not None:
        load_kwargs["instruments_dir"] = (
            None if args.instruments_dir.lower() == "none" else Path(args.instruments_dir)
        )
    tracker = load_tracker(
        WEIGHTS,
        confidence=args.confidence,
        fps=eff_fps,
        workspace_max_center_y_ratio=args.workspace_max_center_y_ratio,
        **load_kwargs,
    )
    if args.cos_tau is not None:
        tracker._session_linker._matcher.cos_tau = args.cos_tau
        print(f"cos_tau overridden -> {args.cos_tau}")
    load_s = time.monotonic() - t0
    print(f"load_tracker: {load_s:.1f}s  providers={tracker._detector.session.get_providers()}")

    cache_path = Path(args.cache) if args.cache else out.with_suffix(".dets.npz")
    if args.from_cache:
        cache = np.load(cache_path)
        tracker._detector = CachedDetector(cache, (h, w))
        print(f"replaying cached detections from {cache_path.name}")
        sink = None
    else:
        sink = {"n": 0}
        tracker._detector = RecordingDetector(tracker._detector, sink)

    rows = []
    update_times = []
    idx = 0
    t_start = time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.stride:
            idx += 1
            continue
        state["frame"] = idx
        state["t"] = idx / src_fps

        t = time.monotonic()
        dets = tracker.update(frame)
        update_times.append(time.monotonic() - t)

        row = {"video_frame": idx, "t_video_s": round(idx / src_fps, 2), "dets": []}
        for k in range(len(dets)):
            x1, y1, x2, y2 = (float(v) for v in dets.xyxy[k])
            row["dets"].append({
                "id": int(dets.tracker_id[k]),
                "cx": round((x1 + x2) / 2, 1),
                "cy": round((y1 + y2) / 2, 1),
                "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "conf": round(float(dets.confidence[k]), 3),
                "area": int(dets.mask[k].sum()) if dets.mask is not None else None,
            })
        rows.append(row)

        n = len(rows)
        if n % 25 == 0:
            done = idx / n_frames
            el = time.monotonic() - t_start
            print(f"  {n} frames | t_video={idx/src_fps:6.1f}s | {done*100:4.1f}% | "
                  f"{np.mean(update_times):.3f}s/update | eta {el/max(done,1e-6)*(1-done)/60:.1f} min",
                  flush=True)
        idx += 1
        if args.limit and len(rows) >= args.limit:
            break
    cap.release()

    roster = sorted(tracker._session_linker.roster)
    meta = {
        "video": str(video), "stride": args.stride, "src_fps": src_fps,
        "effective_fps": round(eff_fps, 3), "confidence": args.confidence,
        "workspace_max_center_y_ratio": args.workspace_max_center_y_ratio,
        "frames_processed": len(rows), "roster": roster,
        "model_version": tracker.model_version,
        "update_s_mean": round(float(np.mean(update_times)), 3),
        "update_s_median": round(float(np.median(update_times)), 3),
        "end_to_end_fps": round(1.0 / float(np.mean(update_times)), 2),
        "wall_s": round(time.monotonic() - t_start, 1),
        "windows_frames": {
            "enrolment": tracker._session_linker._enrolment_window_frames,
            "evidence": tracker._session_linker._evidence_window_frames,
            "death": tracker._session_linker._death_threshold_frames,
        },
    }
    out.with_suffix(".json").write_text(json.dumps({"meta": meta, "frames": rows, "logs": logs}, indent=1))
    if sink is not None:
        np.savez_compressed(cache_path, **{k: v for k, v in sink.items() if k != "n"}, n=sink["n"])
        print(f"cached {sink['n']} frames of detections -> {cache_path.name} "
              f"({cache_path.stat().st_size/1e6:.0f} MB)")
    print(json.dumps(meta, indent=2))
    print(f"wrote {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
