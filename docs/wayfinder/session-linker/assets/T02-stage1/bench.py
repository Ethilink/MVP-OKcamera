#!/usr/bin/env python
"""
T02 stage-1 embedding benchmark — session-linker (ORC MVP).

Method (from matching/notebooks/embeddings.ipynb):
  AutoImageProcessor + AutoModel -> last_hidden_state[:,0] (CLS) -> cosine 1-NN.

Runs on the REAL 8 demo instruments in model/data/instruments/instrumentN/:
15 frames each (1920x1080, captured through the data-collection tool, masks from
the rf-detr detector). Same camera + same detector as the live demo, so this is
the real session-re-identification problem, not a catalog proxy.

Weights must already be cached (offline load is forced). DINOv3 is gated on HF;
this runs DINOv2 (ungated) as a stand-in for structure + latency until the
DINOv3 licence is accepted.

Usage:
  python bench.py --model facebook/dinov2-base --tag dinov2_base --out OUTDIR
"""
import argparse, ast, json, time, os
from pathlib import Path
from itertools import combinations

import numpy as np
from PIL import Image
import psutil
from pycocotools import mask as coco_mask
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import torch
from transformers import AutoImageProcessor, AutoModel

REPO = Path(__file__).resolve().parents[5]
DATA = REPO / "model" / "data" / "instruments"


def parse_field(v):
    """bbox / segmentation are stored as python-repr strings in these files."""
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except Exception:
        return ast.literal_eval(v)


def decode_rle(seg):
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
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    return proc, model


@torch.no_grad()
def embed(images, proc, model, device, batch_size=8):
    out = []
    for i in range(0, len(images), batch_size):
        inp = proc(images=images[i:i + batch_size], return_tensors="pt").to(device)
        out.append(model(**inp).last_hidden_state[:, 0].float().cpu().numpy())
    return np.vstack(out)


@torch.no_grad()
def measure_latency(images, proc, model, device, batch, reps=5):
    sample = images[:batch]
    for _ in range(2):
        inp = proc(images=sample, return_tensors="pt").to(device)
        _ = model(**inp).last_hidden_state[:, 0]
        if device.type == "mps": torch.mps.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        inp = proc(images=sample, return_tensors="pt").to(device)
        _ = model(**inp).last_hidden_state[:, 0]
        if device.type == "mps": torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    tot = float(np.median(ts))
    return {"batch": batch, "total_ms": round(tot * 1e3, 1), "per_crop_ms": round(tot / batch * 1e3, 1)}


def reid_split(E, y, n_gallery, seeds=(0, 1, 2, 3, 4)):
    """Per-instrument split: n_gallery enrolment views, rest are queries.
    Returns mean top-1 8-way, mean margin, frac queries whose own gallery wins."""
    classes = sorted(set(y.tolist()))
    accs, margins, fracpos = [], [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        g_idx, q_idx = [], []
        for c in classes:
            idx = np.where(y == c)[0].copy(); rng.shuffle(idx)
            g_idx += idx[:n_gallery].tolist(); q_idx += idx[n_gallery:].tolist()
        g_idx, q_idx = np.array(g_idx), np.array(q_idx)
        knn = KNeighborsClassifier(1, metric="cosine", algorithm="brute").fit(E[g_idx], y[g_idx])
        accs.append(accuracy_score(y[q_idx], knn.predict(E[q_idx])))
        S = cosine_similarity(E[q_idx], E[g_idx])
        yg = y[g_idx]
        m = []
        for r, qc in enumerate(y[q_idx]):
            sb = S[r, yg == qc].max(); ob = S[r, yg != qc].max()
            m.append(sb - ob)
        m = np.array(m); margins.append(m.mean()); fracpos.append((m > 0).mean())
    return {"n_gallery": n_gallery,
            "top1_8way": round(float(np.mean(accs)), 3),
            "margin_self_minus_other_mean": round(float(np.mean(margins)), 3),
            "frac_query_own_gallery_wins": round(float(np.mean(fracpos)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rss0 = psutil.Process().memory_info().rss

    print(f"[{args.tag}] loading data ...", flush=True)
    crops, y, names = load_instruments()
    print(f"  instruments={len(names)} crops={len(crops)} device={device}", flush=True)

    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print(f"[{args.tag}] loading {args.model} (OFFLINE) ...", flush=True)
    t0 = time.perf_counter()
    try:
        proc, model = get_model(args.model, device)
    except Exception as e:
        (out / f"summary_{args.tag}.json").write_text(json.dumps(
            {"model": args.model, "offline_load_ok": False, "error": str(e)[:400]}, indent=2))
        print(f"OFFLINE LOAD FAILED: {e}"); raise SystemExit(2)
    load_s = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded {load_s:.1f}s params={n_params/1e6:.0f}M", flush=True)

    lat = [measure_latency(crops, proc, model, device, b) for b in (1, 4, 10)]
    E = embed(crops, proc, model, device)
    d = E.shape[1]
    np.savez_compressed(out / f"emb_{args.tag}.npz", E=E, y=y, names=np.array(names))
    rss_mb = (psutil.Process().memory_info().rss - rss0) / 1e6
    mps_mb = (torch.mps.current_allocated_memory() / 1e6) if device.type == "mps" else None

    # 8x8 mean-embedding cosine -> which of the real instruments are confusable
    classes = sorted(set(y.tolist()))
    mean_E = np.array([E[y == c].mean(0) for c in classes])
    sim8 = cosine_similarity(mean_E)
    pairs = sorted([(round(float(sim8[a, b]), 3), names[a], names[b])
                    for a, b in combinations(range(len(classes)), 2)], reverse=True)

    # re-ID as a function of enrolment gallery size
    reid = [reid_split(E, y, g) for g in (3, 5, 8)]

    # rotation / flip robustness (synthetic stand-in for the flipped-return face)
    rng = np.random.default_rng(0)
    sidx = rng.choice(len(crops), size=min(48, len(crops)), replace=False)
    base = E[sidx]; base_y = y[sidx]
    knn_all = KNeighborsClassifier(1, metric="cosine", algorithm="brute").fit(E, y)
    rot = {}
    for label, op in [("rot90", 90), ("rot180", 180), ("rot270", 270), ("mirror", "flip")]:
        if op == "flip":
            imgs = [crops[i].transpose(Image.FLIP_LEFT_RIGHT) for i in sidx]
        else:
            imgs = [crops[i].rotate(op, expand=True) for i in sidx]
        Ev = embed(imgs, proc, model, device)
        cos = np.array([float(cosine_similarity(base[[k]], Ev[[k]])[0, 0]) for k in range(len(sidx))])
        rot[label] = {"mean_cos_to_orig": round(float(cos.mean()), 3),
                      "min_cos_to_orig": round(float(cos.min()), 3),
                      "still_correct_1nn": round(float(np.mean(knn_all.predict(Ev) == base_y)), 3)}

    # gallery aggregation (5 enrolment views), top-1 8-way
    def agg_acc(kind, n_gallery=5, seeds=(0, 1, 2)):
        out_accs = []
        for s in seeds:
            r = np.random.default_rng(s); gi, qi = [], []
            for c in classes:
                idx = np.where(y == c)[0].copy(); r.shuffle(idx)
                gi += idx[:n_gallery].tolist(); qi += idx[n_gallery:].tolist()
            gi, qi = np.array(gi), np.array(qi); yg = y[gi]
            if kind == "nearest":
                k = KNeighborsClassifier(1, metric="cosine", algorithm="brute").fit(E[gi], yg)
                out_accs.append(accuracy_score(y[qi], k.predict(E[qi]))); continue
            sim = cosine_similarity(E[qi], E[gi]); preds = []
            for rr in range(len(qi)):
                if kind == "mean":
                    sc = {c: E[gi][yg == c].mean(0) for c in classes}
                    sc = {c: float(cosine_similarity(E[[qi[rr]]], v[None])[0, 0]) for c, v in sc.items()}
                else:  # top3mean
                    sc = {c: np.sort(sim[rr, yg == c])[-3:].mean() for c in classes}
                preds.append(max(sc, key=sc.get))
            out_accs.append(accuracy_score(y[qi], preds))
        return round(float(np.mean(out_accs)), 3)
    agg = {k: agg_acc(k) for k in ("nearest", "mean", "top3mean")}

    summary = {
        "model": args.model, "tag": args.tag, "embed_dim": int(d),
        "n_params_millions": round(n_params / 1e6, 1), "device": str(device),
        "offline_load_ok": True, "model_load_s": round(load_s, 1),
        "process_rss_mb": round(rss_mb, 0),
        "mps_allocated_mb": round(mps_mb, 0) if mps_mb else None,
        "n_instruments": len(names), "crops_per_instrument": int(len(crops) / len(names)),
        "latency": lat,
        "reid_by_gallery_size": reid,
        "most_similar_instrument_pairs": pairs[:5],
        "least_similar_pair": pairs[-1],
        "rotation_flip": rot,
        "gallery_aggregation_5views": agg,
        "sim8_matrix": [[round(float(v), 3) for v in row] for row in sim8],
        "sim8_labels": names,
    }
    (out / f"summary_{args.tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ("latency", "model_load_s", "process_rss_mb", "reid_by_gallery_size",
                       "most_similar_instrument_pairs", "rotation_flip", "gallery_aggregation_5views")}, indent=2))
    print(f"WROTE {out / f'summary_{args.tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
