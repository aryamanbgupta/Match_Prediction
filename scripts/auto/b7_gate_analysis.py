"""B7 gate analysis — refit (venue-ON) ball calibrators vs the stale ones.

Two prop_backtest detail JSONs, both venue-ON at seed 43 (same 261 matches,
same model + venue encoder; ONLY difference = the ball calibrator):

  baseline   models/auto/b6/detail_venue_s43_n261.json
             stale global vector v1 (E5, fit venue-BLIND) — the canonical
             venue-ON baseline B6 established.
  challenger models/auto/b7/detail_b7cal_s43_n261.json
             refit stack (global + over-0 vectors, both refit on venue-ON
             val predictions; scripts/auto/b7_fit_calibrators.py).

Paired run-vs-run change in Brier/MAE, cluster-bootstrapped by match,
reusing the A8 tooling (this file lives in scripts/auto/, NOT the frozen
eval framework).

PRE-COMMITTED GATE (written before any B7 sim result existed; per IDEAS.md
B7, appended by B6 before any of tonight's work):

  GATE 1 (PRIMARY, decisive) — BOTH sub-conditions required:
    (a) no-regress: batter_runs_mae AND team_first_over_mae must NOT
        regress (no dMAE > 0 with 95% CI excluding 0) vs the venue-ON
        baseline;
    (b) improvement: at least ONE of
          - pooled tail dBrier over the 6 binary lines
              pp_total_ou_{45_5,50_5,55_5},
              first_wicket_runs_ou_30_5,
              highest_over_runs_ou_{18_5,24_5}
            < 0 with 95% CI excluding 0,
          - bowler_wkts_1plus dBrier < 0 with 95% CI excluding 0.

  GATE 2 (guards, must NOT regress = no CI-excludes-0 increase):
    top_bowler, team_total_fours_mae, team_total_sixes_mae.

  Verdict mapping: GATE1 met + GATE2 held -> LANDED (ship: refit global
  replaces models/xgb_v3/vector_scaling_calibrator_v1.pkl after backing up
  the stale one to models/auto/b7/; refit over-0 supersedes A15's artifact
  as the recommended first-over calibrator); exactly one -> TABLED;
  neither -> FAILED.

Everything else is reported for context only and cannot flip the verdict.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "auto"))

from a8_gate_analysis import (brier_pair, cluster_boot, load, mae_pair,  # noqa
                              paired_rows)

TAIL_POOL = [
    "pp_total_ou_45_5", "pp_total_ou_50_5", "pp_total_ou_55_5",
    "first_wicket_runs_ou_30_5",
    "highest_over_runs_ou_18_5", "highest_over_runs_ou_24_5",
]
IMPROVE_BINARY = "bowler_wkts_1plus"
NOREGRESS_MAE = ["batter_runs_mae", "team_first_over_mae"]

GUARD_BINARY = ["top_bowler"]
GUARD_MAE = ["team_total_fours_mae", "team_total_sixes_mae"]

CONTEXT_BINARY = [
    "bowler_wkts_2plus", "bowler_wkts_3plus",
    "batter_50plus", "batter_6plus_six",
    "batter_fours_1plus", "batter_fours_2plus", "batter_fours_3plus",
    "team_highest_individual_ou_29_5", "team_highest_individual_ou_34_5",
    "team_highest_individual_ou_39_5",
    "innings_runs_ou_160_5", "innings_runs_ou_170_5", "innings_runs_ou_180_5",
    "match_total_sixes_ou_15_5", "match_total_sixes_ou_20_5",
    "bowler_economy_ou_8_5", "bowler_economy_ou_10_5",
    "top_batter", "p_tie",
]
CONTEXT_MAE = ["highest_individual_mae", "batter_fours_mae"]


def report_binary(da, db, fams, la, lb):
    hdr = (f"{'family':<32}{'n':>7}{'Brier_' + la:>12}{'Brier_' + lb:>12}"
           f"{'dBrier':>10}   95% CI ({lb}-{la})   flag")
    print(hdr)
    flags = {}
    for fam in fams:
        rows = paired_rows(da, db, fam)
        if not rows:
            continue
        ba, bb, d, (lo, hi) = brier_pair(rows)
        flag = "DOWN(better)" if hi < 0 else "UP(worse)" if lo > 0 else "~noise"
        flags[fam] = (d, lo, hi, flag)
        print(f"{fam:<32}{len(rows):>7}{ba:>12.4f}{bb:>12.4f}{d:>+10.4f}"
              f"   [{lo:+.4f},{hi:+.4f}]  {flag}")
    return flags


def report_mae(da, db, fams, la, lb):
    flags = {}
    for fam in fams:
        res = mae_pair(da, db, fam)
        if res:
            ma, mb, d, (lo, hi), n = res
            flag = "DOWN(better)" if hi < 0 else "UP(worse)" if lo > 0 else "~noise"
            flags[fam] = (d, lo, hi, flag)
            print(f"{fam:<32}{n:>7}  MAE_{la}={ma:.3f}  MAE_{lb}={mb:.3f}  "
                  f"dMAE={d:+.3f}  [{lo:+.3f},{hi:+.3f}]  {flag}")
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stale",
                    default=str(REPO / "models/auto/b6/detail_venue_s43_n261.json"),
                    help="venue-ON baseline with STALE v1 global calibrator (s43)")
    ap.add_argument("--refit",
                    default=str(REPO / "models/auto/b7/detail_b7cal_s43_n261.json"),
                    help="venue-ON run with REFIT global+over-0 calibrator (s43)")
    args = ap.parse_args()

    da, db = load(args.stale), load(args.refit)
    print(f"stale-cal (v1, venue-ON s43): {len(da)} matches | "
          f"refit-cal (B7, venue-ON s43): {len(db)} matches\n")

    print("=" * 84)
    print("GATE 1a — no-regress primaries (dMAE must NOT be >0 with CI excluding 0)")
    print("=" * 84)
    f1a = report_mae(da, db, NOREGRESS_MAE, "stale", "refit")
    g1a = all(not (v[1] > 0) for v in f1a.values()) and len(f1a) == len(NOREGRESS_MAE)

    print("\n" + "=" * 84)
    print("GATE 1b — improvement (>=1 CI-clean DOWN required)")
    print("=" * 84)
    print("tail-pool lines:")
    report_binary(da, db, TAIL_POOL, "stale", "refit")
    pooled = []
    for fam in TAIL_POOL:
        pooled += paired_rows(da, db, fam)
    pool_ok = False
    if pooled:
        d = float(np.mean([(r[3] - r[1]) ** 2 - (r[2] - r[1]) ** 2 for r in pooled]))
        lo, hi = cluster_boot(pooled, lambda r: (r[3] - r[1]) ** 2 - (r[2] - r[1]) ** 2)
        pool_ok = hi < 0
        verdict = ("IMPROVED (CI<0)" if hi < 0
                   else "WORSE (CI>0)" if lo > 0 else "NOISE (CI straddles 0)")
        print(f"\n  >>> pooled tail dBrier over {len(TAIL_POOL)} lines "
              f"({len(pooled)} obs) = {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"  ->  {verdict}")
    print("\nbowler_wkts_1plus:")
    f1b = report_binary(da, db, [IMPROVE_BINARY], "stale", "refit")
    bw_ok = IMPROVE_BINARY in f1b and f1b[IMPROVE_BINARY][2] < 0
    g1b = pool_ok or bw_ok
    gate1 = g1a and g1b

    print("\n" + "=" * 84)
    print("GATE 2 — guards (must NOT regress: no CI-excludes-0 increase)")
    print("=" * 84)
    fg_b = report_binary(da, db, GUARD_BINARY, "stale", "refit")
    fg_m = report_mae(da, db, GUARD_MAE, "stale", "refit")
    gate2 = all(not (v[1] > 0) for v in {**fg_b, **fg_m}.values())

    print("\n" + "=" * 84)
    print("CONTEXT (reported for completeness; cannot flip the verdict)")
    print("=" * 84)
    report_binary(da, db, CONTEXT_BINARY, "stale", "refit")
    report_mae(da, db, CONTEXT_MAE, "stale", "refit")

    print("\n" + "=" * 84)
    print(f"GATE 1a (no-regress primaries): {'MET' if g1a else 'NOT MET'}")
    print(f"GATE 1b (>=1 CI-clean improvement): {'MET' if g1b else 'NOT MET'}"
          f"  [pooled tail: {'yes' if pool_ok else 'no'};"
          f" bowler_wkts_1plus: {'yes' if bw_ok else 'no'}]")
    print(f"GATE 1 = 1a AND 1b: {'MET' if gate1 else 'NOT MET'}")
    print(f"GATE 2 (guards): {'HELD' if gate2 else 'REGRESSED'}")
    v = ("LANDED" if gate1 and gate2
         else "TABLED" if gate1 or gate2 else "FAILED")
    print(f"VERDICT per pre-committed mapping: {v}")


if __name__ == "__main__":
    main()
