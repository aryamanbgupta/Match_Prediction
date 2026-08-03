"""B8 gate analysis — hybrid calibrator (stale v1 global + B7 venue-ON over-0).

Two prop_backtest detail JSONs, both venue-ON at seed 43 (same 261 matches,
same model + venue encoder + empirical bowler selector; ONLY difference =
the ball calibrator):

  baseline   models/auto/b10/detail_blind_s43_n261.json
             bare stale global vector v1 (E5) — the canonical seed-43
             venue-ON baseline established by B10 on the current engine
             (D1/D15 landed after B6, so B6's detail is no longer the
             comparable twin).
  challenger models/auto/b8/detail_b8_s43_n261.json
             hybrid = v1 global (bit-exact) + B7's venue-ON refit over-0
             vector (scripts/auto/b8_compose_hybrid.py). Identical to the
             baseline on every ball outside over 0.

Paired run-vs-run change in Brier/MAE, cluster-bootstrapped by match,
reusing the A8 tooling (this file lives in scripts/auto/, NOT the frozen
eval framework).

PRE-COMMITTED GATE (written and committed BEFORE the B8 sim was run; per
the B8 plan at research/handoff/B8/plan.md):

  GATE 1 (PRIMARY, decisive) — BOTH sub-conditions required:
    (a) improvement: team_first_over_mae dMAE (b8 - blind) < 0 with 95% CI
        excluding 0;
    (b) no-regress: NO CI-excludes-0 *increase* on ANY of
          - pooled tail dBrier over the 6 binary lines
              pp_total_ou_{45_5,50_5,55_5},
              first_wicket_runs_ou_30_5,
              highest_over_runs_ou_{18_5,24_5}
          - bowler_wkts_1plus dBrier
          - batter_runs_mae dMAE

  GATE 2 (guards, must NOT regress = no CI-excludes-0 increase):
    top_bowler, team_total_fours_mae, team_total_sixes_mae.

  Verdict mapping: GATE1 met + GATE2 held -> LANDED; exactly one -> TABLED;
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
IMPROVE_MAE = "team_first_over_mae"
NOREGRESS_BINARY = "bowler_wkts_1plus"
NOREGRESS_MAE = "batter_runs_mae"

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
    ap.add_argument("--blind",
                    default=str(REPO / "models/auto/b10/detail_blind_s43_n261.json"),
                    help="venue-ON baseline with bare STALE v1 global calibrator (s43)")
    ap.add_argument("--hybrid",
                    default=str(REPO / "models/auto/b8/detail_b8_s43_n261.json"),
                    help="venue-ON run with the B8 HYBRID calibrator (s43)")
    args = ap.parse_args()

    da, db = load(args.blind), load(args.hybrid)
    print(f"blind (bare v1, venue-ON s43): {len(da)} matches | "
          f"hybrid (B8, venue-ON s43): {len(db)} matches\n")

    print("=" * 84)
    print("GATE 1a — improvement: team_first_over_mae dMAE < 0 with CI excluding 0")
    print("=" * 84)
    f1a = report_mae(da, db, [IMPROVE_MAE], "blind", "hybrid")
    g1a = IMPROVE_MAE in f1a and f1a[IMPROVE_MAE][2] < 0  # hi < 0

    print("\n" + "=" * 84)
    print("GATE 1b — no-regress: NO CI-excludes-0 INCREASE on pooled tail,")
    print("          bowler_wkts_1plus, batter_runs_mae")
    print("=" * 84)
    print("tail-pool lines:")
    report_binary(da, db, TAIL_POOL, "blind", "hybrid")
    pooled = []
    for fam in TAIL_POOL:
        pooled += paired_rows(da, db, fam)
    pool_regress = False
    if pooled:
        d = float(np.mean([(r[3] - r[1]) ** 2 - (r[2] - r[1]) ** 2 for r in pooled]))
        lo, hi = cluster_boot(pooled, lambda r: (r[3] - r[1]) ** 2 - (r[2] - r[1]) ** 2)
        pool_regress = lo > 0
        verdict = ("IMPROVED (CI<0)" if hi < 0
                   else "WORSE (CI>0)" if lo > 0 else "NOISE (CI straddles 0)")
        print(f"\n  >>> pooled tail dBrier over {len(TAIL_POOL)} lines "
              f"({len(pooled)} obs) = {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"  ->  {verdict}")
    print(f"\n{NOREGRESS_BINARY}:")
    f1b_b = report_binary(da, db, [NOREGRESS_BINARY], "blind", "hybrid")
    bw_regress = NOREGRESS_BINARY in f1b_b and f1b_b[NOREGRESS_BINARY][1] > 0  # lo > 0
    print(f"\n{NOREGRESS_MAE}:")
    f1b_m = report_mae(da, db, [NOREGRESS_MAE], "blind", "hybrid")
    br_regress = NOREGRESS_MAE in f1b_m and f1b_m[NOREGRESS_MAE][1] > 0  # lo > 0
    g1b = not (pool_regress or bw_regress or br_regress)
    gate1 = g1a and g1b

    print("\n" + "=" * 84)
    print("GATE 2 — guards (must NOT regress: no CI-excludes-0 increase)")
    print("=" * 84)
    fg_b = report_binary(da, db, GUARD_BINARY, "blind", "hybrid")
    fg_m = report_mae(da, db, GUARD_MAE, "blind", "hybrid")
    gate2 = all(not (v[1] > 0) for v in {**fg_b, **fg_m}.values())

    print("\n" + "=" * 84)
    print("CONTEXT (reported for completeness; cannot flip the verdict)")
    print("=" * 84)
    report_binary(da, db, CONTEXT_BINARY, "blind", "hybrid")
    report_mae(da, db, CONTEXT_MAE, "blind", "hybrid")

    print("\n" + "=" * 84)
    print(f"GATE 1a (team_first_over_mae CI-clean improvement): "
          f"{'MET' if g1a else 'NOT MET'}")
    print(f"GATE 1b (no CI-clean regression on the 3 no-regress families): "
          f"{'MET' if g1b else 'NOT MET'}"
          f"  [pooled tail worse: {'yes' if pool_regress else 'no'};"
          f" {NOREGRESS_BINARY} worse: {'yes' if bw_regress else 'no'};"
          f" {NOREGRESS_MAE} worse: {'yes' if br_regress else 'no'}]")
    print(f"GATE 1 = 1a AND 1b: {'MET' if gate1 else 'NOT MET'}")
    print(f"GATE 2 (guards): {'HELD' if gate2 else 'REGRESSED'}")
    v = ("LANDED" if gate1 and gate2
         else "TABLED" if gate1 or gate2 else "FAILED")
    print(f"VERDICT per pre-committed mapping: {v}")


if __name__ == "__main__":
    main()
