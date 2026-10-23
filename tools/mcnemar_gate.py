"""Exact McNemar preview of the PR gate on a probe run (dev or test).

The gate agreed in PR #10 (maintainer, 2026-10-22), to be applied to the test split:
either a real paired gain in hit@5 (>= +5 points and exact McNemar p < 0.05) or fewer
irrelevant facts without losing the right ones. This tool computes both branches for
every mode against `off` on whatever provenance_probe output files you point it at:

    python tools/mcnemar_gate.py                      # all dev files
    python tools/mcnemar_gate.py results/provenance_household_dev_on.jsonl

Raw rows are paired per (lang, qid); p is the two-sided exact binomial on the
discordant pair counts (b = rescued, c = lost).
"""
from __future__ import annotations

import collections
import glob
import json
import math
import os
import sys


def mcnemar(b: int, c: int) -> float:
    """Two-sided exact binomial p-value on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def report(path: str) -> None:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    real = [r for r in rows if r["kind"] == "real"]
    chit = [r for r in rows if r["kind"] == "chit"]
    modes = sorted({r["mode"] for r in real})
    if "off" not in modes:
        print(f"{path}: no off mode, skipped")
        return
    hits: dict[str, dict[tuple, bool]] = collections.defaultdict(dict)
    confused = collections.Counter()
    lost = collections.Counter()
    neg = collections.Counter()
    for r in real:
        key = (r["lang"], r["qid"])
        hits[r["mode"]][key] = bool(r["hit5"])
        if r.get("confused"):
            confused[r["mode"]] += 1
        lost[r["mode"]] += len(set(r.get("targets") or []) - set(r.get("keys") or []))
    for r in chit:
        if (r.get("out_n") or 0) > 0:
            neg[r["mode"]] += 1

    base = hits["off"]
    n = len(base)
    base_hits = sum(base.values())
    print(f"\n== {os.path.basename(path)} | real={n} negatives={len(chit)} "
          f"| off hit5 {base_hits}/{n} confused={confused['off']} lost={lost['off']}")
    print(f"{'mode':15s} {'hit5':>8} {'pts':>6} {'b':>3} {'c':>3} {'p':>6} "
          f"{'conf':>5} {'lost':>5} {'neg':>4}  gate branches")
    for m in modes:
        cur = hits[m]
        b = sum(1 for k in base if not base[k] and cur.get(k))
        c = sum(1 for k in base if base[k] and not cur.get(k))
        h = sum(cur.values())
        pts = 100.0 * (h - base_hits) / n
        p = mcnemar(b, c)
        branches = ""
        if m != "off":
            if pts >= 5.0 and p < 0.05:
                branches += "A"
            if c == 0 and confused[m] < confused["off"] and neg[m] <= neg["off"]:
                branches += "B"
            branches = branches or "-"
        print(f"{m:15s} {h:4d}/{n} {pts:+5.1f} {b:3d} {c:3d} {p:6.3f} "
              f"{confused[m]:5d} {lost[m]:5d} {neg[m]:4d}  {branches}")


def main() -> None:
    paths = sys.argv[1:] or sorted(
        glob.glob(os.path.join(os.path.dirname(__file__), "..",
                               "results", "provenance_household_dev*.jsonl")))
    for path in paths:
        report(path)


if __name__ == "__main__":
    main()