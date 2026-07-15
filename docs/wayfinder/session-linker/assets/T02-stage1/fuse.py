#!/usr/bin/env python
"""Fuse the DINO embedding (shape) with mask size (scale) and re-measure re-ID.
Shows: embedding alone vs embedding+size, and the instrument1<->2 error count."""
import json, ast, sys
from pathlib import Path
import numpy as np
from pycocotools import mask as coco_mask
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

OUT = Path(sys.argv[1])
DATA = Path(__file__).resolve().parents[5] / "model" / "data" / "instruments"
# trusted self-produced npz (E, y, names) from bench.py this session
z = np.load(OUT / "emb_dinov2_base.npz", allow_pickle=True)
E, y = z["E"], z["y"]


def parse(v):
    if isinstance(v, (dict, list)): return v
    try: return json.loads(v)
    except Exception: return ast.literal_eval(v)


# rebuild per-crop size features in the SAME order bench.load_instruments used
dirs = sorted([p for p in DATA.iterdir() if p.is_dir() and p.name.startswith("instrument")],
              key=lambda p: int("".join(c for c in p.name if c.isdigit())))
area, length = [], []
for d in dirs:
    coco = json.loads((d / "annotations" / "annotations.json").read_text())
    for a in coco["annotations"]:
        seg = parse(a["segmentation"]); c = seg["counts"]
        m = coco_mask.decode({"size": seg["size"], "counts": c.encode() if isinstance(c, str) else c})
        ys, xs = np.nonzero(m)
        area.append(len(xs))
        pts = np.stack([xs, ys], 1).astype(float); pts -= pts.mean(0)
        _, _, vt = np.linalg.svd(pts, full_matrices=False)
        p = pts @ vt.T
        length.append(p[:, 0].max() - p[:, 0].min())
size = np.stack([np.log(area), np.log(length)], 1)
size = (size - size.mean(0)) / size.std(0)                 # z-scored log size
En = E / np.linalg.norm(E, axis=1, keepdims=True)          # L2-normalized embedding

classes = sorted(set(y.tolist()))
def reid(F, n_gallery=5, seeds=range(5)):
    accs, e12 = [], 0
    for s in seeds:
        r = np.random.default_rng(s); gi, qi = [], []
        for c in classes:
            idx = np.where(y == c)[0].copy(); r.shuffle(idx)
            gi += idx[:n_gallery].tolist(); qi += idx[n_gallery:].tolist()
        gi, qi = np.array(gi), np.array(qi)
        k = KNeighborsClassifier(1, metric="cosine", algorithm="brute").fit(F[gi], y[gi])
        pred = k.predict(F[qi]); accs.append(accuracy_score(y[qi], pred))
        C = confusion_matrix(y[qi], pred, labels=classes)
        i1, i2 = classes.index(0), classes.index(1)
        e12 += int(C[i1, i2] + C[i2, i1])
    return np.mean(accs), e12

print(f"{'descriptor':>26} {'top1_8way':>10} {'inst1<->2 errors (5 seeds)':>28}")
for alpha in (0.0, 0.5, 1.0, 1.5, 2.0):
    F = np.concatenate([En, alpha * size], 1)
    acc, e12 = reid(F)
    tag = "embedding only" if alpha == 0 else f"embedding + {alpha}*size"
    print(f"{tag:>26} {acc:>9.1%} {e12:>20}")
