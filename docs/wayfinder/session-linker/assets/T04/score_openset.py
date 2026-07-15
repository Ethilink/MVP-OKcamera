#!/usr/bin/env python
"""T04 open-set scoring — known-instrument vs foreign-object rejection, and
crucially whether MULTI-FRAME VOTING over a tracklet separates them where a
single crop cannot (the open question T02 left).

Per session it uses:
  T04-cache/<SESSION>/detections.npz   (per-det embedding, tracker_id, bbox, sec)
  T04-cache/<SESSION>/tracklets.json
  T04-cache/labels_<SESSION>.json      (tracker_id -> known|foreign|junk)
  T04-cache/gallery_embeddings.npz     (canonical 8 instruments x15 crops)

Score model: gallery -> 8 per-instrument prototype vectors (mean, renorm). For a
detection embedding e, score = max_i cos(e, proto_i)  (best match to any known
instrument). Then per tracklet we compare:
  - s_framemax   : max over its frames         (the most dangerous single frame -> false accepts)
  - s_framemed   : median over its frames       (typical single frame)
  - s_voted      : cos(mean_embedding, nearest proto)  (multi-frame averaging = voting)
A foreign object is a FALSE ACCEPT if its score exceeds the accept threshold.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get("T04_CACHE",
            Path(__file__).resolve().parents[5] / "outputs" / "t04-cache"))


def l2norm(E):
    n = np.linalg.norm(E, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return E / n


def load_gallery():
    g = np.load(BASE / "gallery_embeddings.npz")  # numeric + string arrays; no pickle needed
    E, y, names = g["E"], g["y"], g["names"]
    protos = l2norm(np.stack([E[y == c].mean(0) for c in sorted(set(y.tolist()))]))
    return protos, list(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    args = ap.parse_args()
    sess = args.session

    protos, names = load_gallery()                      # (8,768)
    d = np.load(BASE / sess / "detections.npz")
    E = l2norm(d["embedding"].astype(np.float64))        # (N,768)
    tid = d["tracker_id"]
    labels = json.loads((BASE / f"labels_{sess}.json").read_text())["labels"]
    tj = {t["tracker_id"]: t for t in json.loads((BASE / sess / "tracklets.json").read_text())}

    per_det_score = (E @ protos.T).max(1)                # best match to any instrument
    per_det_best = (E @ protos.T).argmax(1)

    rows = []
    for t, cls in labels.items():
        t = int(t)
        sel = np.where(tid == t)[0]
        if len(sel) == 0:
            continue
        s = per_det_score[sel]
        mean_emb = l2norm(E[sel].mean(0, keepdims=True))[0]
        s_voted = float((mean_emb @ protos.T).max())
        rows.append({
            "tracker_id": t, "cls": cls, "n": int(len(sel)),
            "start": round(float(tj[t]["start_sec"]), 1),
            "end": round(float(tj[t]["end_sec"]), 1),
            "s_framemax": round(float(s.max()), 3),
            "s_framemed": round(float(np.median(s)), 3),
            "s_voted": round(s_voted, 3),
            "best_instrument": names[int(np.bincount(per_det_best[sel]).argmax())],
        })

    def grp(cls, key):
        return np.array([r[key] for r in rows if r["cls"] == cls], float)

    print(f"\n===== {sess} open-set =====")
    print(f"tracklets: known={sum(r['cls']=='known' for r in rows)} "
          f"foreign={sum(r['cls']=='foreign' for r in rows)} "
          f"junk={sum(r['cls']=='junk' for r in rows)}")
    for cls in ("known", "foreign", "junk"):
        for key in ("s_framemax", "s_framemed", "s_voted"):
            v = grp(cls, key)
            if len(v):
                print(f"  {cls:8s} {key:11s} mean={v.mean():.3f} "
                      f"p10={np.percentile(v,10):.3f} p50={np.percentile(v,50):.3f} "
                      f"p90={np.percentile(v,90):.3f} max={v.max():.3f}")

    # per-foreign detail (the danger cases)
    print("\n  foreign tracklets (each is one intruder over its whole window):")
    fdesc = json.loads((BASE / f"labels_{sess}.json").read_text()).get("foreign_desc", {})
    for r in sorted([r for r in rows if r["cls"] == "foreign"], key=lambda z: -z["s_voted"]):
        print(f"    t{r['tracker_id']:<3d} {fdesc.get(str(r['tracker_id']),''):20s} "
              f"n={r['n']:<4d} framemax={r['s_framemax']:.3f} framemed={r['s_framemed']:.3f} "
              f"VOTED={r['s_voted']:.3f} -> looks like {r['best_instrument']}")

    # threshold sweep: reject if score < tau. Compare single-frame(framemax) vs voted.
    print("\n  threshold sweep (accept a tracklet if score >= tau):")
    print("    tau |  single-frame(framemax)        |  multi-frame(voted)")
    print("        |  known-acc  foreign-rej         |  known-acc  foreign-rej")
    kn_fm, fo_fm = grp("known", "s_framemax"), grp("foreign", "s_framemax")
    kn_v, fo_v = grp("known", "s_voted"), grp("foreign", "s_voted")
    for tau in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        ka_fm = (kn_fm >= tau).mean(); fr_fm = (fo_fm < tau).mean()
        ka_v = (kn_v >= tau).mean(); fr_v = (fo_v < tau).mean()
        print(f"   {tau:.2f} |   {ka_fm:5.0%}      {fr_fm:5.0%}             |"
              f"   {ka_v:5.0%}      {fr_v:5.0%}")

    out = BASE / sess / "openset_scores.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n  WROTE {out}")


if __name__ == "__main__":
    main()
