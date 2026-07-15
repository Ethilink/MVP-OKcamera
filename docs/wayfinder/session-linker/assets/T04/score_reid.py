#!/usr/bin/env python
"""T04 re-ID scoring — does a RETURNING instrument re-link to the right identity?

Per session, using per-instrument identity labels (labels_<S>.json 'identity'):
  * Enroll each instrument from its EARLIEST solid tracklet (Start-time enrollment,
    multi-view = that tracklet's per-frame embeddings).
  * Every LATER tracklet of a known instrument is a "return" query (the tracker
    broke its id across a move / occlusion / removal-and-replacement).
  * Predict each query's identity = argmax over instruments of nearest-enrolled-
    view cosine (multi-frame voted via the query's mean embedding). Correct if it
    equals the true identity.
Reports top-1 re-ID accuracy, the twin (instrument1 vs instrument2) confusion,
and the same with SIZE-FUSION (z-scored log bbox area+diagonal appended, à la
T02) swept over fusion weight.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get("T04_CACHE",
            Path(__file__).resolve().parents[5] / "outputs" / "t04-cache"))
MIN_ENROLL = 20   # a solid enrollment tracklet
MIN_QUERY = 5     # ignore tiny query fragments


def l2norm(E):
    n = np.linalg.norm(E, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return E / n


def size_feats(bbox_xyxy):
    w = np.clip(bbox_xyxy[:, 2] - bbox_xyxy[:, 0], 1, None)
    h = np.clip(bbox_xyxy[:, 3] - bbox_xyxy[:, 1], 1, None)
    area = w * h
    diag = np.sqrt(w * w + h * h)
    return np.stack([np.log(area), np.log(diag)], 1)


def fuse(E, z, w):
    """Append size-weighted z-features to the (already L2-normed) embedding."""
    if w == 0:
        return E
    return np.hstack([E, w * z])


def predict(query_vecs, gal_vecs, gal_ids):
    """For each query row, argmax over instruments of nearest gallery-view cosine."""
    sims = query_vecs @ gal_vecs.T                       # (Q, G)
    ids = np.array(gal_ids)
    out = []
    for r in range(sims.shape[0]):
        best, bid = -2, None
        for iid in set(gal_ids):
            m = sims[r, ids == iid].max()
            if m > best:
                best, bid = m, iid
        out.append(bid)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    args = ap.parse_args()
    sess = args.session

    d = np.load(BASE / sess / "detections.npz")
    E = l2norm(d["embedding"].astype(np.float64))
    tid = d["tracker_id"]
    bbox = d["bbox_xyxy"].astype(np.float64)
    lab = json.loads((BASE / f"labels_{sess}.json").read_text())
    identity = {int(k): v for k, v in lab.get("identity", {}).items()}
    tj = {t["tracker_id"]: t for t in json.loads((BASE / sess / "tracklets.json").read_text())}

    # size z-scored over known detections only
    known_mask = np.array([identity.get(int(t)) is not None for t in tid])
    Z = size_feats(bbox)
    mu, sd = Z[known_mask].mean(0), Z[known_mask].std(0) + 1e-9
    Zz = (Z - mu) / sd

    # group known tracklets by identity, ordered by start
    by_id = {}
    for t, name in identity.items():
        if t in tj:
            by_id.setdefault(name, []).append(t)
    for name in by_id:
        by_id[name].sort(key=lambda t: tj[t]["start_sec"])

    def rows_for(t):
        return np.where(tid == t)[0]

    results = {}
    for w in (0.0, 0.5, 1.0, 2.0):
        gal_vecs, gal_ids = [], []
        enroll_of = {}
        for name, tl in by_id.items():
            solid = [t for t in tl if tj[t]["n_dets"] >= MIN_ENROLL] or tl
            et = solid[0]
            enroll_of[name] = et
            sel = rows_for(et)
            V = fuse(E[sel], Zz[sel], w)
            gal_vecs.append(V)
            gal_ids += [name] * len(sel)
        gal_vecs = l2norm(np.vstack(gal_vecs))

        q_true, q_pred = [], []
        for name, tl in by_id.items():
            for t in tl:
                if t == enroll_of[name] or tj[t]["n_dets"] < MIN_QUERY:
                    continue
                sel = rows_for(t)
                qv = fuse(E[sel].mean(0, keepdims=True), Zz[sel].mean(0, keepdims=True), w)
                qv = l2norm(qv)
                pred = predict(qv, gal_vecs, gal_ids)[0]
                q_true.append(name)
                q_pred.append(pred)
        q_true, q_pred = np.array(q_true), np.array(q_pred)
        acc = float((q_true == q_pred).mean()) if len(q_true) else float("nan")
        # twin confusion
        twin = {"i1_as_i2": 0, "i2_as_i1": 0, "i1_n": 0, "i2_n": 0}
        for tr, pr in zip(q_true, q_pred):
            if tr == "instrument1":
                twin["i1_n"] += 1; twin["i1_as_i2"] += (pr == "instrument2")
            if tr == "instrument2":
                twin["i2_n"] += 1; twin["i2_as_i1"] += (pr == "instrument1")
        results[w] = {"n_query": int(len(q_true)), "top1": round(acc, 3), "twin": twin}

    print(f"\n===== {sess} re-ID (per-session enrollment) =====")
    print(f"instruments enrolled: {len(by_id)}   "
          f"query (return) tracklets: {results[0.0]['n_query']}")
    print("  size_w | top1  | twin i1->i2 | twin i2->i1")
    for w, r in results.items():
        tw = r["twin"]
        print(f"   {w:.1f}   | {r['top1'] if r['top1']==r['top1'] else 'n/a':<5} | "
              f"{tw['i1_as_i2']}/{tw['i1_n']:<9} | {tw['i2_as_i1']}/{tw['i2_n']}")
    # list the misses at w=0 for inspection
    print("\n  misclassified return tracklets (size_w=0):")
    gal_vecs, gal_ids, enroll_of = [], [], {}
    for name, tl in by_id.items():
        solid = [t for t in tl if tj[t]["n_dets"] >= MIN_ENROLL] or tl
        enroll_of[name] = solid[0]; sel = rows_for(solid[0])
        gal_vecs.append(E[sel]); gal_ids += [name] * len(sel)
    gal_vecs = l2norm(np.vstack(gal_vecs))
    for name, tl in by_id.items():
        for t in tl:
            if t == enroll_of[name] or tj[t]["n_dets"] < MIN_QUERY:
                continue
            sel = rows_for(t)
            pred = predict(l2norm(E[sel].mean(0, keepdims=True)), gal_vecs, gal_ids)[0]
            if pred != name:
                print(f"    t{t} ({name}, {tj[t]['start_sec']:.0f}-{tj[t]['end_sec']:.0f}s, "
                      f"n={tj[t]['n_dets']}) -> predicted {pred}")

    (BASE / sess / "reid_scores.json").write_text(json.dumps(
        {str(w): r for w, r in results.items()}, indent=2))
    print(f"\n  WROTE {BASE / sess / 'reid_scores.json'}")


if __name__ == "__main__":
    main()
