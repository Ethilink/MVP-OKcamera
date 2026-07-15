#!/usr/bin/env python
"""T04 cache stage — detections + tracklets + DINOv2-B embeddings + montage.

Run from `model/` via `uv run`:
  cd model && uv run --quiet python <this>.py --session 09-07-26-004
  cd model && uv run --quiet python <this>.py --gallery-only

Per session it writes to the scratchpad T04-cache/<SESSION>/:
  detections.npz, crops/det_<idx>.jpg, tracklets.json, montage_tracklets.jpg
Globally it writes T04-cache/gallery_embeddings.npz.

Embedding method matches T02 bench.py EXACTLY (DINOv2-base, AutoImageProcessor +
AutoModel, last_hidden_state[:,0] CLS, batched) so numbers stay comparable. Here
masks are boolean HxW arrays (not RLE), so we mask+crop directly.
"""
import argparse
import ast
import json
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- paths / const
REPO = Path(__file__).resolve().parents[5]   # assets/T04/ -> ... -> repo root
DATA = REPO / "model/data/instruments"   # canonical gallery (bench.py DATA)
WEIGHTS = REPO / "model/weights/checkpoint_best_regular.onnx"
# Heavy cache (detections/embeddings/crops) is regenerable and NOT committed.
# Point T04_CACHE at wherever you want it; defaults to <repo>/outputs/t04-cache.
OUT_BASE = Path(os.environ.get("T04_CACHE", REPO / "outputs" / "t04-cache"))

MODEL_ID = "facebook/dinov2-base"
CONF_THRESHOLD = 0.30      # detector confidence floor (spec)
FRAME_STEP = 6             # sample every 6th frame (~5 fps from 30 fps)
NATIVE_FPS = 30.0
SAMPLE_FPS = NATIVE_FPS / FRAME_STEP  # 5.0

# ByteTrack params (see report): scaled to 5 fps sampling. minimum_consecutive_
# _frames=2 confirms real tracks fast (1 lost lead-in frame) while single-frame
# junk stays tracker_id=-1 and is dropped; lost_track_buffer=25 (=5 s) re-links
# instruments across brief hand occlusions.
BYTETRACK_KW = dict(
    frame_rate=int(round(SAMPLE_FPS)),
    track_activation_threshold=0.50,
    minimum_consecutive_frames=2,
    lost_track_buffer=25,
    minimum_iou_threshold=-0.30,
    high_conf_det_threshold=0.80,
)


# ---- helpers copied verbatim from T02 bench.py (embedding method must match) --
# bench.py itself pulls in sklearn/psutil at import time, which aren't in the
# model venv; the spec explicitly allows copying the small helpers instead.
def parse_field(v):
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except Exception:
        return ast.literal_eval(v)


def decode_rle(seg):
    from pycocotools import mask as coco_mask
    seg = parse_field(seg)
    counts = seg["counts"]
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    return coco_mask.decode({"size": seg["size"], "counts": counts})


def crop_masked(img_arr, seg, bbox):
    m = decode_rle(seg)
    masked = img_arr * m[:, :, None]
    x, y, w, h = [int(v) for v in parse_field(bbox)]
    return Image.fromarray(masked[y:y + h, x:x + w])


def load_instruments():
    """Returns crops (list[PIL]), labels (np[int]), names (list[str])."""
    dirs = sorted([d for d in DATA.iterdir() if d.is_dir() and d.name.startswith("instrument")],
                  key=lambda p: int("".join(c for c in p.name if c.isdigit())))
    crops, labels, names = [], [], []
    for li, d in enumerate(dirs):
        names.append(d.name)
        coco = json.loads((d / "annotations" / "annotations.json").read_text())
        fn_by_id = {im["id"]: im["file_name"] for im in coco["images"]}
        img_cache = {}
        for a in coco["annotations"]:
            fn = fn_by_id[a["image_id"]]
            if fn not in img_cache:
                img_cache[fn] = np.array(Image.open(d / "images" / fn).convert("RGB"))
            crops.append(crop_masked(img_cache[fn], a["segmentation"], a["bbox"]))
            labels.append(li)
    return crops, np.array(labels), names


def get_model(model_id, device):
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    return proc, model


def embed(images, proc, model, device, batch_size=8):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            inp = proc(images=images[i:i + batch_size], return_tensors="pt").to(device)
            out.append(model(**inp).last_hidden_state[:, 0].float().cpu().numpy())
    return np.vstack(out)


# ------------------------------------------------------------ mask-crop helper
def crop_masked_bool(img_rgb: np.ndarray, mask, xyxy) -> Image.Image:
    """Apply boolean mask, crop to bbox -> PIL RGB (mirrors bench.crop_masked).

    Falls back to a plain bbox crop when the mask is missing.
    """
    h, w = img_rgb.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in xyxy]
    x0, x1 = max(0, min(x0, w - 1)), max(0, min(x1, w))
    y0, y1 = max(0, min(y0, h - 1)), max(0, min(y1, h))
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    if mask is not None:
        m = mask.astype(np.uint8)
        masked = img_rgb * m[:, :, None]
    else:
        masked = img_rgb
    return Image.fromarray(masked[y0:y1, x0:x1])


def l2norm(E: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(E, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return E / n


# ------------------------------------------------------------ gallery embedding
def build_gallery(proc, model, device):
    print("[gallery] loading canonical instruments ...", flush=True)
    crops, y, names = load_instruments()
    print(f"  instruments={len(names)} crops={len(crops)}", flush=True)
    E = l2norm(embed(crops, proc, model, device))
    out = OUT_BASE / "gallery_embeddings.npz"
    np.savez_compressed(out, E=E.astype(np.float32), y=y, names=np.array(names))
    print(f"  WROTE {out}  E={E.shape}", flush=True)


# ------------------------------------------------------------ session pipeline
def process_session(session, proc, model, device):
    video = REPO / f"matching/data/testing/{session}/videos/{session}.mp4"
    if not video.exists():
        raise FileNotFoundError(video)
    out_dir = OUT_BASE / session
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # tracker (import here so imports are cheap for gallery-only runs)
    from trackers import ByteTrackTracker
    from trackers.utils.iou import GIoU
    from trackers.utils.state_representations import XCYCSRStateEstimator

    from orc_model.components.detector.detector import Detector

    det = Detector(str(WEIGHTS), confidence_threshold=CONF_THRESHOLD)
    tracker = ByteTrackTracker(
        state_estimator_class=XCYCSRStateEstimator, iou=GIoU(), **BYTETRACK_KW
    )

    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[{session}] frames={total} step={FRAME_STEP} -> ~{total // FRAME_STEP} sampled",
          flush=True)

    # per-detection accumulators (aligned by global detection index)
    frame_number, seconds, bbox_xyxy, conf, tracker_id = [], [], [], [], []
    crop_imgs = []           # PIL crops, embedded + flushed in chunks (bounded RAM)
    emb_chunks = []          # embedded chunks (order-preserving), vstacked at end
    EMBED_CHUNK = 1024       # flush crops to embeddings every N to bound peak memory
    masks_present = 0
    masks_missing = 0
    raw_det_total = 0        # detections from detector (pre-tracker)
    n_sampled = 0

    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % FRAME_STEP == 0:
            ok2, frame = cap.retrieve()
            if not ok2:
                break
            n_sampled += 1
            dets = det.predict(frame)              # BGR in, sv.Detections out
            raw_det_total += len(dets)
            tracked = tracker.update(dets)
            if len(tracked) > 0:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tids = tracked.tracker_id
                for j in range(len(tracked)):
                    tid = int(tids[j])
                    if tid < 0:                    # unconfirmed / junk -> drop
                        continue
                    mask_j = None
                    if tracked.mask is not None:
                        mask_j = tracked.mask[j]
                        masks_present += 1
                    else:
                        masks_missing += 1
                    xyxy = tracked.xyxy[j]
                    di = len(frame_number)
                    crop = crop_masked_bool(img_rgb, mask_j, xyxy)
                    crop.save(crops_dir / f"det_{di}.jpg", quality=90)
                    crop_imgs.append(crop)
                    frame_number.append(idx)
                    seconds.append(idx / NATIVE_FPS)
                    bbox_xyxy.append([float(v) for v in xyxy])
                    conf.append(float(tracked.confidence[j]))
                    tracker_id.append(tid)
            if len(crop_imgs) >= EMBED_CHUNK:     # flush -> embeddings, free PIL crops
                emb_chunks.append(embed(crop_imgs, proc, model, device))
                crop_imgs.clear()
            if n_sampled % 100 == 0:
                print(f"  frame {idx}/{total}  sampled={n_sampled}  dets={len(frame_number)}",
                      flush=True)
        idx += 1
    cap.release()
    if crop_imgs:                                  # flush remainder
        emb_chunks.append(embed(crop_imgs, proc, model, device))
        crop_imgs.clear()

    n = len(frame_number)
    print(f"[{session}] sampled={n_sampled} raw_dets={raw_det_total} "
          f"confirmed_dets={n} masks_present={masks_present} missing={masks_missing}",
          flush=True)

    # ---- embeddings were computed in chunks during the loop (same method as
    # gallery); concat in order and L2-normalize. Order matches the accumulators.
    if n > 0:
        E = l2norm(np.vstack(emb_chunks)).astype(np.float32)
        assert E.shape[0] == n, f"embedding/accumulator misalignment {E.shape[0]} vs {n}"
    else:
        E = np.zeros((0, 768), np.float32)

    frame_number = np.array(frame_number, np.int64)
    seconds = np.array(seconds, np.float64)
    bbox_xyxy = np.array(bbox_xyxy, np.float32).reshape(-1, 4)
    conf = np.array(conf, np.float32)
    tracker_id = np.array(tracker_id, np.int64)

    np.savez_compressed(
        out_dir / "detections.npz",
        frame_number=frame_number, seconds=seconds, bbox_xyxy=bbox_xyxy,
        conf=conf, tracker_id=tracker_id, embedding=E,
    )
    print(f"  WROTE {out_dir / 'detections.npz'}", flush=True)

    # ---- tracklets.json
    tracklets = []
    for tid in sorted(set(tracker_id.tolist())):
        sel = np.where(tracker_id == tid)[0]
        secs = seconds[sel]
        order = np.argsort(secs)
        sel = sel[order]
        secs = seconds[sel]
        centers = np.stack([
            (bbox_xyxy[sel, 0] + bbox_xyxy[sel, 2]) / 2,
            (bbox_xyxy[sel, 1] + bbox_xyxy[sel, 3]) / 2,
        ], axis=1)
        mid = (secs[0] + secs[-1]) / 2
        rep_local = int(np.argmin(np.abs(secs - mid)))
        rep_idx = int(sel[rep_local])
        mean_emb = l2norm(E[sel].mean(0, keepdims=True))[0] if len(sel) else np.zeros(768)
        tracklets.append({
            "tracker_id": int(tid),
            "n_dets": int(len(sel)),
            "start_sec": round(float(secs[0]), 3),
            "end_sec": round(float(secs[-1]), 3),
            "frames": [int(v) for v in frame_number[sel].tolist()],
            "mean_conf": round(float(conf[sel].mean()), 4),
            "centroid_start": [round(float(centers[0, 0]), 1), round(float(centers[0, 1]), 1)],
            "centroid_end": [round(float(centers[-1, 0]), 1), round(float(centers[-1, 1]), 1)],
            "rep_det_idx": rep_idx,
            "rep_crop": f"crops/det_{rep_idx}.jpg",
            "mean_embedding": [round(float(v), 6) for v in mean_emb.tolist()],
        })
    (out_dir / "tracklets.json").write_text(json.dumps(tracklets, indent=2))
    print(f"  WROTE {out_dir / 'tracklets.json'}  n_tracklets={len(tracklets)}", flush=True)

    # ---- montage
    build_montage(session, out_dir, crops_dir, tracklets)
    return {"n_sampled": n_sampled, "raw_dets": raw_det_total, "confirmed_dets": n,
            "n_tracklets": len(tracklets), "masks_present": masks_present,
            "masks_missing": masks_missing}


def build_montage(session, out_dir, crops_dir, tracklets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not tracklets:
        print("  (no tracklets, skipping montage)", flush=True)
        return
    # order tiles by tracklet length (longest first) so stable instruments lead
    order = sorted(range(len(tracklets)), key=lambda i: -tracklets[i]["n_dets"])
    n = len(order)
    ncols = min(8, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 2.3))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for k, ti in enumerate(order):
        t = tracklets[ti]
        ax = axes[k]
        img = Image.open(out_dir / t["rep_crop"])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"t{t['tracker_id']}  {t['start_sec']:.1f}-{t['end_sec']:.1f}s  n={t['n_dets']}",
            fontsize=7,
        )
    fig.suptitle(f"{session} — {n} tracklets (rep crop per tracker_id)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = out_dir / "montage_tracklets.jpg"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  WROTE {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None)
    ap.add_argument("--gallery-only", action="store_true")
    ap.add_argument("--no-gallery", action="store_true")
    args = ap.parse_args()

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[setup] device={device} model={MODEL_ID}", flush=True)
    proc, model = get_model(MODEL_ID, device)

    if not args.no_gallery and (args.gallery_only or
                                not (OUT_BASE / "gallery_embeddings.npz").exists()):
        build_gallery(proc, model, device)
    if args.gallery_only:
        return

    if not args.session:
        print("no --session given; done (gallery only).", flush=True)
        return
    stats = process_session(args.session, proc, model, device)
    print("STATS " + json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
