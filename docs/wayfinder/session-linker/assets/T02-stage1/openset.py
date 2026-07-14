#!/usr/bin/env python
"""Open-set check: do the foreign objects reject below the accept threshold?

genuine return  -> accepted as instrument k iff best-gallery-cosine >= tau AND argmax==k
foreign object  -> should be REJECTED (best-gallery-cosine < tau)
Reports a threshold sweep so T02/T03 can pin a starting tau, and the separation
between genuine self-match scores and foreign impostor scores.
"""
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

OUT = Path(sys.argv[1])
DATA = Path(__file__).resolve().parents[5] / "model" / "data" / "instruments"
FOREIGN = Path(__file__).resolve().parents[5] / "model" / "data" / "other_objects"


def parse(v):
    if isinstance(v, (dict, list)): return v
    try: return json.loads(v)
    except Exception: return ast.literal_eval(v)


def load_coco_crops(root):
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
        crops.append(Image.fromarray((np.array(Image.open(root / "images" / f).convert("RGB")) * m[:, :, None])[y:y+h, x:x+w]))
    return crops


# instrument embeddings from the saved npz (trusted, self-produced)
z = np.load(OUT / "emb_dinov2_base.npz", allow_pickle=True)
E, y = z["E"], z["y"]
classes = sorted(set(y.tolist()))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
proc, model = get_model("facebook/dinov2-base", device)
foreign = load_coco_crops(FOREIGN)
Ef = embed(foreign, proc, model, device)
print(f"foreign crops embedded: {len(foreign)}")

# genuine gallery/query split (gallery=5), collect self & other scores across seeds
self_s, other_s = [], []
for s in range(5):
    r = np.random.default_rng(s); gi, qi = [], []
    for c in classes:
        idx = np.where(y == c)[0].copy(); r.shuffle(idx)
        gi += idx[:5].tolist(); qi += idx[5:].tolist()
    gi, qi = np.array(gi), np.array(qi)
    S = cosine_similarity(E[qi], E[gi]); yg = y[gi]
    for rr, qc in enumerate(y[qi]):
        self_s.append(S[rr, yg == qc].max())
        other_s.append(S[rr, yg != qc].max())
self_s, other_s = np.array(self_s), np.array(other_s)
# foreign impostor score = best cosine to the full gallery (all instrument crops)
imp = cosine_similarity(Ef, E).max(1)

print(f"\ngenuine self-match cosine : mean {self_s.mean():.3f}  p10 {np.percentile(self_s,10):.3f}  min {self_s.min():.3f}")
print(f"foreign impostor  cosine : mean {imp.mean():.3f}  p90 {np.percentile(imp,90):.3f}  max {imp.max():.3f}")
print(f"\n{'tau':>5} {'true-accept(correct)':>20} {'foreign-REJECT':>15} {'wrong-link(cross)':>18}")
for tau in np.round(np.arange(0.45, 0.86, 0.05), 2):
    ta = np.mean((self_s >= tau) & (self_s >= other_s))
    fr = np.mean(imp < tau)
    wl = np.mean(other_s >= tau)         # a wrong instrument gallery clears the bar
    print(f"{tau:>5.2f} {ta:>19.0%} {fr:>14.0%} {wl:>17.0%}")
