#!/usr/bin/env python
"""Diagnose the open-set overlap: (1) what are the high-scoring foreign objects,
(2) does a best-vs-second-best MARGIN reject better than absolute cosine."""
import json, ast, sys, os
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from bench import get_model, embed  # noqa

OUT = Path(sys.argv[1]); ASSET = Path(sys.argv[2])
DATA = Path(__file__).resolve().parents[5] / "model" / "data" / "instruments"
FOREIGN = Path(__file__).resolve().parents[5] / "model" / "data" / "other_objects"


def parse(v):
    if isinstance(v, (dict, list)): return v
    try: return json.loads(v)
    except Exception: return ast.literal_eval(v)


def load_crops(root):
    coco = json.loads((root / "annotations" / "annotations.json").read_text())
    fn = {im["id"]: im["file_name"] for im in coco["images"]}
    cache, crops = {}, []
    for a in coco["annotations"]:
        f = fn[a["image_id"]]
        if f not in cache:
            cache[f] = np.array(Image.open(root / "images" / f).convert("RGB"))
        seg = parse(a["segmentation"]); c = seg["counts"]
        m = coco_mask.decode({"size": seg["size"], "counts": c.encode() if isinstance(c, str) else c})
        x, y, w, h = [int(v) for v in parse(a["bbox"])]
        if w < 4 or h < 4: continue
        crops.append(Image.fromarray((cache[f] * m[:, :, None])[y:y+h, x:x+w]))
    return crops


z = np.load(OUT / "emb_dinov2_base.npz", allow_pickle=True)  # trusted, self-produced
E, y = z["E"], z["y"]; names = list(z["names"]); classes = sorted(set(y.tolist()))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
proc, model = get_model("facebook/dinov2-base", device)
foreign = load_crops(FOREIGN); Ef = embed(foreign, proc, model, device)

# per-instrument score for a set of query embeddings = max cosine to that instrument's crops
def per_instrument_scores(Eq):
    S = cosine_similarity(Eq, E)
    out = np.zeros((len(Eq), len(classes)))
    for j, c in enumerate(classes):
        out[:, j] = S[:, y == c].max(1)
    return out

Sf = per_instrument_scores(Ef)
imp_top1 = Sf.max(1); imp_arg = Sf.argmax(1)
imp_sorted = np.sort(Sf, 1)
imp_margin = imp_sorted[:, -1] - imp_sorted[:, -2]

# genuine queries (gallery=5) top1/margin
self_top1, self_margin, self_ok = [], [], []
for s in range(5):
    r = np.random.default_rng(s); gi, qi = [], []
    for c in classes:
        idx = np.where(y == c)[0].copy(); r.shuffle(idx)
        gi += idx[:5].tolist(); qi += idx[5:].tolist()
    gi, qi = np.array(gi), np.array(qi)
    Sg = cosine_similarity(E[qi], E[gi]); yg = y[gi]
    for rr, qc in enumerate(y[qi]):
        per = np.array([Sg[rr, yg == c].max() for c in classes])
        srt = np.sort(per)
        self_top1.append(srt[-1]); self_margin.append(srt[-1] - srt[-2])
        self_ok.append(classes[per.argmax()] == qc)
self_top1 = np.array(self_top1); self_margin = np.array(self_margin); self_ok = np.array(self_ok)

print("MARGIN (best - 2nd-best instrument):")
print(f"  genuine correct-return margin: mean {self_margin[self_ok].mean():.3f}  p10 {np.percentile(self_margin[self_ok],10):.3f}")
print(f"  foreign impostor       margin: mean {imp_margin.mean():.3f}  p90 {np.percentile(imp_margin,90):.3f}")
print(f"\n{'rule':>34} {'true-accept':>12} {'foreign-reject':>15}")
for tau in (0.60, 0.65, 0.70):
    ta = np.mean(self_ok & (self_top1 >= tau)); fr = np.mean(imp_top1 < tau)
    print(f"{'abs cos >= %.2f'%tau:>34} {ta:>11.0%} {fr:>14.0%}")
for mg in (0.03, 0.05, 0.08):
    ta = np.mean(self_ok & (self_margin >= mg)); fr = np.mean(imp_margin < mg)
    print(f"{'margin >= %.2f'%mg:>34} {ta:>11.0%} {fr:>14.0%}")
for tau, mg in [(0.65, 0.03), (0.65, 0.05), (0.70, 0.05)]:
    ta = np.mean(self_ok & (self_top1 >= tau) & (self_margin >= mg))
    fr = np.mean((imp_top1 < tau) | (imp_margin < mg))
    print(f"{'abs>=%.2f AND margin>=%.2f'%(tau,mg):>34} {ta:>11.0%} {fr:>14.0%}")

# montage: 10 highest-scoring foreign objects, labelled with best-match instrument
order = np.argsort(-imp_top1)[:10]
cell = 200
grid = Image.new("RGB", (cell * 5, cell * 2), (20, 20, 20))
for k, oi in enumerate(order):
    im = foreign[oi].convert("RGB").copy(); im.thumbnail((cell - 10, cell - 30))
    grid.paste(im, ((k % 5) * cell + 5, (k // 5) * cell + 5))
grid.save(ASSET / "foreign_top10_by_score.jpg", quality=90)
print("\ntop-10 foreign impostor scores -> matched instrument:")
for oi in order:
    print(f"  cos {imp_top1[oi]:.3f} margin {imp_margin[oi]:.3f} -> {names[imp_arg[oi]]}")
print("montage:", ASSET / "foreign_top10_by_score.jpg")
