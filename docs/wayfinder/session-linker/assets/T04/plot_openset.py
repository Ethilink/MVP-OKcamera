#!/usr/bin/env python
"""Strip plot of per-tracklet open-set scores by class, for one or more sessions.
Shows single-frame-max (the dangerous lucky frame) vs multi-frame voted score,
and the known/foreign separation gap."""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(os.environ.get("T04_CACHE",
            Path(__file__).resolve().parents[5] / "outputs" / "t04-cache"))
COL = {"known": "#1a9850", "foreign": "#d73027", "junk": "#999999"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--out", default=str(BASE / "openset_plot.jpg"))
    args = ap.parse_args()

    fig, axes = plt.subplots(1, len(args.sessions), figsize=(4.6 * len(args.sessions), 5.2),
                             squeeze=False)
    for ax, sess in zip(axes[0], args.sessions):
        rows = json.loads((BASE / sess / "openset_scores.json").read_text())
        xmap = {"known": 0, "foreign": 1, "junk": 2}
        rng = np.random.default_rng(0)
        for r in rows:
            x = xmap[r["cls"]]
            jitter = (rng.random() - 0.5) * 0.28
            # voted (filled) with a faint line up to framemax (the lucky single frame)
            ax.plot([x + jitter, x + jitter], [r["s_voted"], r["s_framemax"]],
                    color=COL[r["cls"]], alpha=0.25, lw=1, zorder=1)
            ax.scatter(x + jitter, r["s_framemax"], s=18, facecolors="none",
                       edgecolors=COL[r["cls"]], alpha=0.6, zorder=2)
            ax.scatter(x + jitter, r["s_voted"], s=60, color=COL[r["cls"]],
                       edgecolors="k", linewidths=0.4, zorder=3)
        # separation band between max-foreign-voted and min-known-voted
        kv = [r["s_voted"] for r in rows if r["cls"] == "known"]
        fv = [r["s_voted"] for r in rows if r["cls"] == "foreign"]
        if kv and fv and max(fv) < min(kv):
            ax.axhspan(max(fv), min(kv), color="#4575b4", alpha=0.12, zorder=0)
            ax.axhline((max(fv) + min(kv)) / 2, color="#4575b4", ls="--", lw=1, zorder=0)
            ax.text(2.35, (max(fv) + min(kv)) / 2, f"gap\n{max(fv):.2f}–{min(kv):.2f}",
                    fontsize=7, color="#4575b4", va="center")
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["known", "foreign", "junk"])
        ax.set_ylim(0.1, 1.0); ax.set_ylabel("cosine to nearest known instrument")
        ax.set_title(sess, fontsize=10); ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Open-set: filled = multi-frame VOTED score · hollow = best single frame (framemax)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
