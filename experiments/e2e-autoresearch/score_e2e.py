#!/usr/bin/env python3
"""End-to-end session-linker scorer for the two July-15 demo takes.

Ground truth = model/docs/demo-validation.md (the scripted recording checklist
Bram filmed against) + the shipped-config baseline it reproduces bit-for-bit.

Because replay is `--from-cache` over fixed detections and Deep OC-SORT uses a
mobilenet appearance model (NOT the linker's DINOv2/SRC), the raw track ids and
their timings are STABLE across every matcher/embedder/gallery challenger. Only
the SessionLinker's per-track outcome (linked:K / unknown / deferred:K) changes.
So we key ground truth to raw ids and score any challenger as a diff against a
reference: retain the correct links, keep foreign objects rejected, never emit a
wrong link, and flag any change on the two known-hard fail-safe cases for review.

This scorer is a REGRESSION GATE + improvement flagger, not the primary optimizer
(the crop-level frozen eval in experiments/matcher-autoresearch is the trustworthy
graded signal). Its job: prove a challenger does not break the demo, and surface
legitimate end-to-end wins (e.g. the 368s flip re-linking after flip-augmentation).

Usage:
    python3 score_e2e.py --trace <replay.json> --gt <gt_events.json> --take A|B
    python3 score_e2e.py --build-gt --trace-a <base_a.json> --trace-b <base_b.json> --out gt_events.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

OUTCOME_RE = re.compile(r"(\d+): '([^']+)'")
RAWBINDS_RE = re.compile(r"raw_binds=(\{[^}]*\})")


def extract_outcomes(trace_path: Path) -> dict:
    """Return {'roster', 'events', 'final_t', 'raw_binds'}.

    raw_binds ({raw_id: specimen}) is read off the enrolment-freeze log. It is the
    PHYSICAL anchor: with binding on, session id == specimen, so a `linked:S`
    outcome means the same physical tool across configs iff their raw_binds agree.
    A challenger whose raw_binds differs from the baseline's has re-bound the roster
    (its numbers mean different tools) and its per-link diff must be flagged, not
    compared number-for-number.
    """
    d = json.loads(Path(trace_path).read_text())
    roster = sorted(d["meta"]["roster"])
    events = []
    final_t = 0.0
    raw_binds = {}
    for l in d["logs"]:
        msg = l["msg"]
        t = float(l["t_video_s"])
        final_t = max(final_t, t)
        if "enrolment freeze" in msg:
            m = RAWBINDS_RE.search(msg)
            if m:
                raw_binds = {int(k): int(v) for k, v in
                             re.findall(r"(\d+): (\d+)", m.group(1))}
        if "batch decision" not in msg:
            continue
        oc = msg.split("outcomes=")[-1]
        for raw, out in OUTCOME_RE.findall(oc):
            events.append({"t": t, "raw": int(raw), "outcome": out})
    return {"roster": roster, "events": events, "final_t": final_t, "raw_binds": raw_binds}


def build_gt(trace_a: Path, trace_b: Path) -> dict:
    """Derive the ground-truth spec from the shipped-config baselines + demo-validation.md.

    For each take we record, per raw id, the baseline outcome and a label:
      - 'foreign'   : must stay REJECT (unknown). Hard floor. (demo-validation.md foreign windows)
      - 'hardcase'  : documented fail-safe (302s twin swap raw72, 368s flip raw88);
                      currently 'unknown'; a change is a candidate WIN but flagged for
                      human review (correct target id is genuinely ambiguous).
      - 'link'      : baseline linked to a session id -> must not regress / must not
                      change to a WRONG id. This is the "don't break the demo" set.
      - 'noop'      : baseline unknown with no Missing slot to fill (eligible_missing=0
                      at that time) -> benign; not scored.
    Foreign / hardcase raw ids are hand-verified from demo-validation.md 'Latest results'.
    """
    FOREIGN = {"A": set(), "B": {12, 25, 26, 28, 30, 36}}      # Take B phone/pen/keys/random
    HARDCASE = {"A": set(), "B": {72, 88}}                     # 302s twin swap, 368s flip
    gt = {}
    for take, tr in (("A", trace_a), ("B", trace_b)):
        o = extract_outcomes(tr)
        # last outcome per raw id = its settled decision
        settled = {}
        for e in o["events"]:
            settled[e["raw"]] = e["outcome"]
        labels = {}
        for raw, out in settled.items():
            if raw in FOREIGN[take]:
                labels[str(raw)] = {"label": "foreign", "baseline": out}
            elif raw in HARDCASE[take]:
                labels[str(raw)] = {"label": "hardcase", "baseline": out}
            elif out.startswith("linked:"):
                labels[str(raw)] = {"label": "link", "baseline": out}
            else:
                labels[str(raw)] = {"label": "noop", "baseline": out}
        gt[take] = {
            "expected_roster": [1, 2, 3, 4, 5, 6, 7, 8],
            "baseline_raw_binds": {str(k): v for k, v in o["raw_binds"].items()},
            "raw_labels": labels,
        }
    return gt


def score(trace_path: Path, gt: dict, take: str) -> dict:
    o = extract_outcomes(trace_path)
    spec = gt[take]
    settled = {}
    for e in o["events"]:
        settled[e["raw"]] = e["outcome"]

    roster_ok = o["roster"] == spec["expected_roster"]
    base_binds = {int(k): v for k, v in spec.get("baseline_raw_binds", {}).items()}
    now_binds = o["raw_binds"]
    # physical roster identical iff the freeze bound the same raw ids to the same
    # specimens. If not, the per-link number-comparison below is NOT physical and the
    # result must be read as "rebound — review", never as a silent pass/fail.
    binding_matches_baseline = bool(base_binds) and now_binds == base_binds
    rebound = bool(base_binds) and not binding_matches_baseline

    foreign_total = foreign_reject = 0
    correct_link_total = correct_link_retained = 0
    regressions = []       # correct link -> lost or changed to wrong id (DEMO-BREAKING)
    wrong_links = []        # foreign or hardcase -> linked (DEMO-BREAKING)
    hardcase_changes = []   # fail-safe -> something else (FLAG for human review)

    for raw, info in spec["raw_labels"].items():
        raw_i = int(raw)
        now = settled.get(raw_i, "absent")
        base = info["baseline"]
        lab = info["label"]
        if lab == "foreign":
            foreign_total += 1
            if now == "unknown" or now == "absent":
                foreign_reject += 1
            else:
                wrong_links.append({"raw": raw_i, "was": base, "now": now})
        elif lab == "hardcase":
            if now != base:
                hardcase_changes.append({"raw": raw_i, "was": base, "now": now})
        elif lab == "link":
            correct_link_total += 1
            if now == base:
                correct_link_retained += 1
            else:
                regressions.append({"raw": raw_i, "was": base, "now": now})

    foreign_reject_rate = foreign_reject / foreign_total if foreign_total else 1.0
    link_retention = correct_link_retained / correct_link_total if correct_link_total else 1.0
    # a regression that turned a correct link into a link to a DIFFERENT id is a wrong link
    wrong_link_regressions = [r for r in regressions if r["now"].startswith("linked:")]

    demo_safe = (
        roster_ok
        and foreign_reject_rate == 1.0
        and len(wrong_links) == 0
        and len(wrong_link_regressions) == 0
        and binding_matches_baseline
    )
    return {
        "take": take,
        "roster": o["roster"],
        "roster_ok": roster_ok,
        "binding_matches_baseline": binding_matches_baseline,
        "rebound": rebound,
        "foreign_reject_rate": round(foreign_reject_rate, 4),
        "foreign_total": foreign_total,
        "link_retention": round(link_retention, 4),
        "correct_link_total": correct_link_total,
        "correct_link_retained": correct_link_retained,
        "regressions": regressions,
        "wrong_links": wrong_links,
        "hardcase_changes": hardcase_changes,
        "demo_safe": demo_safe,
        "review_required": rebound or bool(hardcase_changes),
        "n_events": len(o["events"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace")
    ap.add_argument("--gt")
    ap.add_argument("--take", choices=["A", "B"])
    ap.add_argument("--build-gt", action="store_true")
    ap.add_argument("--trace-a")
    ap.add_argument("--trace-b")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.build_gt:
        gt = build_gt(Path(args.trace_a), Path(args.trace_b))
        Path(args.out).write_text(json.dumps(gt, indent=2))
        for take in ("A", "B"):
            labs = gt[take]["raw_labels"]
            from collections import Counter
            c = Counter(v["label"] for v in labs.values())
            print(f"Take {take}: {dict(c)}")
        print(f"wrote {args.out}")
        return 0

    gt = json.loads(Path(args.gt).read_text())
    m = score(Path(args.trace), gt, args.take)
    print(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
