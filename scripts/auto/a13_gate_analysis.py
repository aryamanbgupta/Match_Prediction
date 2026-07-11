#!/usr/bin/env python3
"""A13 gate analysis — dispersion (fan-out) vs single-vector baseline.

Reads the aligned baseline (k=1) and dispersion (k fitted) detail JSONs emitted
by a13_dispersion_eval.py score, and evaluates the A13 sim gate pair:

  METRIC 1 (calibration): P10-P90 coverage of the continuous families moves
    TOWARD 80% (report mean |coverage - 80%| base vs disp; lower = better).
  METRIC 2 (tail O/U Brier): pooled Brier over pp_total + first_wicket +
    highest_over O/U families IMPROVES paired vs baseline (want dBrier < 0,
    CI < 0), cluster-bootstrapped by match.
  GUARD: batter_runs_mae MAE must not regress (fan-out is mean-preserving so
    dMAE ~ 0 by construction); top_bowler unchanged (not a fanned family).

Both metrics improve -> LANDED; one -> TABLED; neither -> FAILED.
"""
import argparse
import json
from collections import defaultdict

import numpy as np

OU_GATE = ["pp_total_ou_45_5", "pp_total_ou_50_5", "pp_total_ou_55_5",
           "first_wicket_runs_ou_30_5",
           "highest_over_runs_ou_18_5", "highest_over_runs_ou_24_5"]
COVERAGE_FAMS = ["batter_runs_mae", "team_total_fours_mae", "team_total_sixes_mae",
                 "team_first_over_mae", "highest_individual_mae", "batter_fours_mae"]
GUARD_BINARY = ["top_bowler", "bowler_wkts_1plus", "bowler_wkts_2plus", "bowler_wkts_3plus"]


def load(p):
    return json.load(open(p))


def paired_rows(base, disp, fam):
    idx = {r["match_id"]: r["obs"] for r in disp}
    out = []
    for rb in base:
        mid = rb["match_id"]
        ob = rb["obs"].get(fam)
        od = idx.get(mid, {}).get(fam)
        if not ob or not od or len(ob) != len(od):
            continue
        for xb, xd in zip(ob, od):
            if xb.get("y") != xd.get("y"):
                continue
            out.append((mid, float(xb["y"]), float(xb["p"]), float(xd["p"])))
    return out


def cluster_boot(rows, fn, n_boot=2000, seed=29):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for r in rows:
        by[r[0]].append(r)
    mids = list(by)
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(len(mids), size=len(mids), replace=True)
        acc = []
        for i in samp:
            acc.extend(fn(r) for r in by[mids[i]])
        vals.append(np.mean(acc))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def brier_line(base, disp, fam):
    rows = paired_rows(base, disp, fam)
    if not rows:
        return None
    bb = float(np.mean([(p - y) ** 2 for _, y, p, _ in rows]))
    bd = float(np.mean([(p - y) ** 2 for _, y, _, p in rows]))
    lo, hi = cluster_boot(rows, lambda r: (r[3] - r[1]) ** 2 - (r[2] - r[1]) ** 2)
    flag = "DOWN(better)" if hi < 0 else "UP(worse)" if lo > 0 else "~noise"
    return fam, len(rows), bb, bd, bd - bb, lo, hi, flag


def coverage(det, fam):
    h = n = 0
    for m in det:
        for r in m["obs"].get(fam, []):
            if "sim_p10" not in r:
                continue
            h += 1 if (r["sim_p10"] <= r["actual"] <= r["sim_p90"]) else 0
            n += 1
    return (h / n if n else float("nan")), n


def mae(det, fam):
    errs = []
    for m in det:
        for r in m["obs"].get(fam, []):
            errs.append(abs(r["sim_mean"] - r["actual"]))
    return float(np.mean(errs)) if errs else float("nan"), len(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--disp", required=True)
    args = ap.parse_args()
    base, disp = load(args.base), load(args.disp)
    print(f"base: {len(base)} matches | disp: {len(disp)} matches\n")

    # METRIC 1: coverage toward 80%
    print("=== METRIC 1: P10-P90 coverage toward 80% (continuous families) ===")
    print(f"{'family':24s}{'n':>7}{'cov_base':>10}{'cov_disp':>10}{'|dev80|_b':>11}{'|dev80|_d':>11}")
    devb, devd = [], []
    for fam in COVERAGE_FAMS:
        cb, n = coverage(base, fam)
        cd, _ = coverage(disp, fam)
        db, dd = abs(cb - 0.80), abs(cd - 0.80)
        devb.append(db); devd.append(dd)
        print(f"{fam:24s}{n:>7}{cb*100:>9.1f}%{cd*100:>9.1f}%{db*100:>10.1f}%{dd*100:>10.1f}%")
    mdb, mdd = float(np.mean(devb)), float(np.mean(devd))
    print(f"  --- mean |coverage-80%|: base {mdb*100:.1f}%  disp {mdd*100:.1f}%  "
          f"({'IMPROVED' if mdd < mdb - 0.005 else 'not improved'})")

    # METRIC 2: pooled tail O/U Brier
    print("\n=== METRIC 2: tail O/U Brier (disp - base), cluster-boot by match ===")
    print(f"{'family':24s}{'n':>7}{'B_base':>9}{'B_disp':>9}{'dBrier':>10}   95% CI      flag")
    for fam in OU_GATE:
        r = brier_line(base, disp, fam)
        if r:
            print(f"{r[0]:24s}{r[1]:>7}{r[2]:>9.4f}{r[3]:>9.4f}{r[4]:>+10.4f}   "
                  f"[{r[5]:+.4f},{r[6]:+.4f}]  {r[7]}")
    pooled = []
    for fam in OU_GATE:
        pooled += paired_rows(base, disp, fam)
    if pooled:
        d = float(np.mean([(r[3]-r[1])**2 - (r[2]-r[1])**2 for r in pooled]))
        lo, hi = cluster_boot(pooled, lambda r: (r[3]-r[1])**2 - (r[2]-r[1])**2)
        print(f"  --- POOLED tail dBrier = {d:+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]  "
              f"({'IMPROVED' if hi < 0 else 'worse' if lo > 0 else 'noise'})")

    # GUARD
    print("\n=== GUARD: batter_runs_mae (mean-preserving -> ~0) + binary guards ===")
    mb, n = mae(base, "batter_runs_mae")
    md, _ = mae(disp, "batter_runs_mae")
    print(f"batter_runs_mae  n={n}  MAE base {mb:.4f}  disp {md:.4f}  dMAE {md-mb:+.4f}")
    for fam in GUARD_BINARY:
        r = brier_line(base, disp, fam)
        if r:
            print(f"{r[0]:24s}{r[1]:>7}{r[2]:>9.4f}{r[3]:>9.4f}{r[4]:>+10.4f}   "
                  f"[{r[5]:+.4f},{r[6]:+.4f}]  {r[7]}")


if __name__ == "__main__":
    main()
