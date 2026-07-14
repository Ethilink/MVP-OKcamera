#!/usr/bin/env python
"""Post-hoc analysis: confusion structure + a visual montage of the real 8."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import load_instruments  # noqa

OUT = Path(sys.argv[1])            # scratch out dir with emb_*.npz
ASSET = Path(sys.argv[2])          # repo asset dir for the montage
tag = "dinov2_base"
# allow_pickle: this .npz is written by our own bench.py this session (E, y, and a
# string array `names`); it is a trusted, self-produced file, not external input.
z = np.load(OUT / f"emb_{tag}.npz", allow_pickle=True)
E, y, names = z["E"], z["y"], list(z["names"])
classes = sorted(set(y.tolist()))

# confusion under a 5-view gallery, averaged over seeds
C = np.zeros((len(classes), len(classes)))
for s in range(5):
    r = np.random.default_rng(s); gi, qi = [], []
    for c in classes:
        idx = np.where(y == c)[0].copy(); r.shuffle(idx)
        gi += idx[:5].tolist(); qi += idx[5:].tolist()
    gi, qi = np.array(gi), np.array(qi)
    k = KNeighborsClassifier(1, metric="cosine", algorithm="brute").fit(E[gi], y[gi])
    C += confusion_matrix(y[qi], k.predict(E[qi]), labels=classes)
print("Confusion (rows=true, cols=pred), 5 seeds summed, gallery=5:")
print("        " + " ".join(f"{n[-2:]:>4}" for n in names))
for i, n in enumerate(names):
    print(f"{n:>12} " + " ".join(f"{int(C[i,j]):>4}" for j in range(len(classes))))
errs = [(int(C[i, j]), names[i], names[j]) for i in range(len(classes))
        for j in range(len(classes)) if i != j and C[i, j] > 0]
print("\nTop confusions:", sorted(errs, reverse=True)[:6])

# montage: 1 representative crop per instrument (+ 2 extra for the top pair)
crops, yy, nn = load_instruments()
by = {c: [i for i in range(len(yy)) if yy[i] == c] for c in classes}
cell = 220
grid = Image.new("RGB", (cell * len(classes), cell), (20, 20, 20))
for c in classes:
    im = crops[by[c][0]].convert("RGB").copy(); im.thumbnail((cell - 12, cell - 12))
    grid.paste(im, (c * cell + 6, 6))
grid.save(ASSET / "montage_one_per_instrument.jpg", quality=90)

# instrument1 vs instrument2 — 5 crops each, stacked
top = Image.new("RGB", (cell * 5, cell * 2), (20, 20, 20))
for row, c in enumerate([0, 1]):
    for col in range(5):
        im = crops[by[c][col]].convert("RGB").copy(); im.thumbnail((cell - 12, cell - 12))
        top.paste(im, (col * cell + 6, row * cell + 6))
top.save(ASSET / "pair_instrument1_vs_2.jpg", quality=90)
print("\nWROTE montage + pair image to", ASSET)
