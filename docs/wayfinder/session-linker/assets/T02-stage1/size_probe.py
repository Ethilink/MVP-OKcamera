#!/usr/bin/env python
"""Does mask SIZE separate the near-twin pair the embedding confuses?

Size is discarded by DINO's fixed-resolution preprocessing. The detector mask
still carries it. Test: per-instrument mask area + physical length (PCA major
axis of mask pixels, orientation-invariant), and whether a size threshold
separates instrument1 vs instrument2. Also report within-instrument scale
consistency (CV) — a caveat, since these frames may not be at demo-camera scale.
"""
import json, ast
from pathlib import Path
import numpy as np
from pycocotools import mask as coco_mask

DATA = Path(__file__).resolve().parents[5] / "model" / "data" / "instruments"


def parse(v):
    if isinstance(v, (dict, list)): return v
    try: return json.loads(v)
    except Exception: return ast.literal_eval(v)


def metrics_for(d):
    coco = json.loads((d / "annotations" / "annotations.json").read_text())
    areas, lengths, widths = [], [], []
    for a in coco["annotations"]:
        seg = parse(a["segmentation"]); c = seg["counts"]
        m = coco_mask.decode({"size": seg["size"], "counts": c.encode() if isinstance(c, str) else c})
        ys, xs = np.nonzero(m)
        areas.append(len(xs))
        pts = np.stack([xs, ys], 1).astype(float); pts -= pts.mean(0)
        # PCA: project onto principal axes -> major (length) and minor (width) extent
        _, _, vt = np.linalg.svd(pts, full_matrices=False)
        proj = pts @ vt.T
        lengths.append(proj[:, 0].max() - proj[:, 0].min())
        widths.append(proj[:, 1].max() - proj[:, 1].min())
    return np.array(areas), np.array(lengths), np.array(widths)


dirs = sorted([p for p in DATA.iterdir() if p.is_dir() and p.name.startswith("instrument")],
              key=lambda p: int("".join(c for c in p.name if c.isdigit())))
A, L, W = {}, {}, {}
print(f"{'instrument':>12} {'area_px(mean±cv)':>22} {'length_px(mean±cv)':>22} {'aspect L/W':>10}")
for d in dirs:
    a, l, w = metrics_for(d); A[d.name], L[d.name], W[d.name] = a, l, w
    cv = lambda x: x.std() / x.mean()
    print(f"{d.name:>12} {a.mean():>10.0f} ±{cv(a):>4.2f}        {l.mean():>10.0f} ±{cv(l):>4.2f}        {(l/w).mean():>6.2f}")

# instrument1 vs instrument2 separability on each 1-D size feature
print("\n--- instrument1 vs instrument2: can a single size threshold split them? ---")
for name, D in [("area", A), ("length", L), ("aspect(L/W)", {k: L[k] / W[k] for k in L})]:
    x1, x2 = D["instrument1"], D["instrument2"]
    # best threshold accuracy (1-D, balanced)
    lo, hi = min(x1.min(), x2.min()), max(x1.max(), x2.max())
    best = 0.0
    for t in np.linspace(lo, hi, 200):
        acc = max((x1 < t).mean() + (x2 >= t).mean(), (x1 >= t).mean() + (x2 < t).mean()) / 2
        best = max(best, acc)
    overlap = (max(x1.min(), x2.min()) < min(x1.max(), x2.max()))
    print(f"  {name:>12}: inst1 {x1.mean():.0f}±{x1.std():.0f}  inst2 {x2.mean():.0f}±{x2.std():.0f}"
          f"  | best 1-D split acc = {best:.0%}  | ranges overlap: {overlap}")
