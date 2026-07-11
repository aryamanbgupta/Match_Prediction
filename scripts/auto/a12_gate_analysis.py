"""A12 gate analysis — dew-conditional 2nd-innings calibrator vs single-vector.

Reads two prop_backtest detail JSONs (same matches / same seed; only the ball
calibrator differs: dew vs vec). The dew tilt touches ONLY innings-2 balls, so
within any per-innings / per-player family the observations that DIFFER between
the two runs are exactly the second-innings (chasing side, or innings-2 bowler)
observations; identical obs are innings-1 and contribute 0 to any paired delta.
We therefore self-identify the innings-2 subset as the *changed* obs (no
cricsheet join needed) and report the changed count as a sanity check
(per-innings families should change ~1 obs/match = the chasing team).

Gate pair (A12):
  GATE 1 (must IMPROVE): 2nd-innings scoring + wicket calibration, i.e. the
     innings-2 obs of {innings_runs_ou, pp_total_ou} (scoring) and
     {bowler_wkts, bowler_economy} (wicket/economy) — paired Brier_sim DOWN,
     CI excluding 0 (the A14 bar for a real, above-noise gain).
  GATE 2 (must NOT REGRESS): an established sim skill — top_bowler (Brier) and
     team_total_fours_mae (MAE) — want the 95% CI to include 0.

Verdict (sim-pair dual-metric): GATE1 improves AND GATE2 holds -> LANDED;
GATE1 improves but GATE2 regresses -> TABLED; GATE1 does not improve -> FAILED.
"""
import argparse
import json
from collections import defaultdict

import numpy as np

SCORE2 = ["innings_runs_ou_160_5", "innings_runs_ou_170_5", "innings_runs_ou_180_5",
          "pp_total_ou_45_5", "pp_total_ou_50_5", "pp_total_ou_55_5"]
WKT2 = ["bowler_wkts_1plus", "bowler_wkts_2plus", "bowler_wkts_3plus",
        "bowler_economy_ou_8_5", "bowler_economy_ou_10_5"]
GUARD_BIN = ["top_bowler"]
GUARD_MAE = ["team_total_fours_mae"]
EPS = 1e-9


def load(p):
    return json.load(open(p))


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


def paired_bin(det_a, det_b, fams, only_changed):
    """rows = (mid, y, p_vec, p_dew). Guards identity by (family, y). Returns
    rows, n_all_obs, n_changed_obs."""
    idx_b = {r["match_id"]: r["obs"] for r in det_b}
    rows, n_all, n_chg = [], 0, 0
    for ra in det_a:
        mid = ra["match_id"]
        ob = idx_b.get(mid, {})
        for fam in fams:
            oa, obb = ra["obs"].get(fam), ob.get(fam)
            if not oa or not obb or len(oa) != len(obb):
                continue
            for xa, xb in zip(oa, obb):
                if xa.get("y") != xb.get("y"):
                    continue
                n_all += 1
                changed = abs(float(xa["p"]) - float(xb["p"])) > EPS
                if changed:
                    n_chg += 1
                if only_changed and not changed:
                    continue
                rows.append((mid, float(xa["y"]), float(xa["p"]), float(xb["p"])))
    return rows, n_all, n_chg


def brier(rows):
    ba = float(np.mean([(pv - y) ** 2 for _, y, pv, _ in rows]))
    bd = float(np.mean([(pd_ - y) ** 2 for _, y, _, pd_ in rows]))
    lo, hi = cluster_boot(rows, lambda r: (r[3] - r[1]) ** 2 - (r[2] - r[1]) ** 2)
    return ba, bd, bd - ba, lo, hi


def mae_pair(det_a, det_b, fam, only_changed):
    idx_b = {r["match_id"]: r["obs"] for r in det_b}
    rows, n_all, n_chg = [], 0, 0
    for ra in det_a:
        mid = ra["match_id"]
        la, lb = ra["obs"].get(fam), idx_b.get(mid, {}).get(fam)
        if not la or not lb or len(la) != len(lb):
            continue
        for xa, xb in zip(la, lb):
            if xa.get("actual") != xb.get("actual"):
                continue
            n_all += 1
            changed = abs(float(xa["sim_mean"]) - float(xb["sim_mean"])) > EPS
            if changed:
                n_chg += 1
            if only_changed and not changed:
                continue
            rows.append((mid, float(xa["actual"]),
                         float(xa["sim_mean"]), float(xb["sim_mean"])))
    if not rows:
        return None
    ma = float(np.mean([abs(pv - y) for _, y, pv, _ in rows]))
    md = float(np.mean([abs(pd_ - y) for _, y, _, pd_ in rows]))
    lo, hi = cluster_boot(rows, lambda r: abs(r[3] - r[1]) - abs(r[2] - r[1]))
    return ma, md, md - ma, lo, hi, len(rows), n_all, n_chg


def flag(lo, hi):
    return "DOWN(better)" if hi < 0 else "UP(worse)" if lo > 0 else "~noise"


def bin_line(det_a, det_b, fam, only_changed):
    rows, n_all, n_chg = paired_bin(det_a, det_b, [fam], only_changed)
    if not rows:
        return None
    ba, bd, d, lo, hi = brier(rows)
    return (fam, len(rows), n_chg, n_all, ba, bd, d, lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec", required=True, help="single-vector detail JSON (baseline)")
    ap.add_argument("--dew", required=True, help="dew-conditional detail JSON")
    args = ap.parse_args()

    dv, dd = load(args.vec), load(args.dew)
    print(f"vec: {len(dv)} matches | dew: {len(dd)} matches\n")

    hdr = (f"{'family':<24}{'n_used':>8}{'n_chg':>7}{'n_all':>7}"
           f"{'Brier_vec':>11}{'Brier_dew':>11}{'dBrier':>10}"
           f"{'  95% CI (dew-vec)':>21}  flag")

    print("=== GATE 1a — 2nd-innings SCORING (changed=innings-2 obs only) ===")
    print(hdr)
    for fam in SCORE2:
        r = bin_line(dv, dd, fam, only_changed=True)
        if r:
            print(f"{r[0]:<24}{r[1]:>8}{r[2]:>7}{r[3]:>7}{r[4]:>11.4f}"
                  f"{r[5]:>11.4f}{r[6]:>+10.4f}   [{r[7]:+.4f},{r[8]:+.4f}]  {flag(r[7],r[8])}")
    rows, na, nc = paired_bin(dv, dd, SCORE2, only_changed=True)
    ba, bd, d, lo, hi = brier(rows)
    print(f"  >>> POOLED 2nd-inns scoring: n={len(rows)} dBrier={d:+.4f} "
          f"CI [{lo:+.4f},{hi:+.4f}]  {flag(lo,hi)}")

    print("\n=== GATE 1b — 2nd-innings WICKET/economy (changed=innings-2 obs only) ===")
    print(hdr)
    for fam in WKT2:
        r = bin_line(dv, dd, fam, only_changed=True)
        if r:
            print(f"{r[0]:<24}{r[1]:>8}{r[2]:>7}{r[3]:>7}{r[4]:>11.4f}"
                  f"{r[5]:>11.4f}{r[6]:>+10.4f}   [{r[7]:+.4f},{r[8]:+.4f}]  {flag(r[7],r[8])}")
    rows, na, nc = paired_bin(dv, dd, WKT2, only_changed=True)
    ba, bd, d, lo, hi = brier(rows)
    print(f"  >>> POOLED 2nd-inns wicket/econ: n={len(rows)} dBrier={d:+.4f} "
          f"CI [{lo:+.4f},{hi:+.4f}]  {flag(lo,hi)}")

    rows, na, nc = paired_bin(dv, dd, SCORE2 + WKT2, only_changed=True)
    ba, bd, d, lo, hi = brier(rows)
    print(f"\n  ===> GATE 1 (scoring+wicket, innings-2 pooled): n={len(rows)} "
          f"dBrier={d:+.4f}  CI [{lo:+.4f},{hi:+.4f}]  {flag(lo,hi)}  "
          f"(IMPROVED iff CI<0)")

    print("\n=== GATE 2 — established-skill guard (WHOLE family, must not regress) ===")
    print(hdr)
    for fam in GUARD_BIN:
        r = bin_line(dv, dd, fam, only_changed=False)
        if r:
            print(f"{r[0]:<24}{r[1]:>8}{r[2]:>7}{r[3]:>7}{r[4]:>11.4f}"
                  f"{r[5]:>11.4f}{r[6]:>+10.4f}   [{r[7]:+.4f},{r[8]:+.4f}]  {flag(r[7],r[8])}")
    for fam in GUARD_MAE:
        res = mae_pair(dv, dd, fam, only_changed=False)
        if res:
            ma, md, d, lo, hi, n, na, nc = res
            print(f"{fam:<24}{n:>8}{nc:>7}{na:>7}  MAE_vec={ma:.3f} MAE_dew={md:.3f} "
                  f"dMAE={d:+.3f}  [{lo:+.3f},{hi:+.3f}]  {flag(lo,hi)}")
    print("  >>> GATE 2 holds iff every guard CI includes 0 (no significant regression)")


if __name__ == "__main__":
    main()
