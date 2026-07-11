"""A16 gate analysis — sparse regime-change-over calibrator {0,6,15}.

A16 adds over-6 (start of middle overs) and over-15 (start of death) vectors on
top of A15's over-0-only calibrator. The decisive question is NOT "does A16 beat
the single global vector" (over 0 alone already does, per A15) but "do overs 6
and 15 add anything OBSERVABLE beyond over 0?".

So we run TWO paired comparisons, reusing A8's paired cluster-bootstrap tooling
(this file lives in scripts/auto/, NOT the frozen eval framework):

  (1) A16 (regime) vs single-vector  -> total effect (compare to A15's total)
  (2) A16 (regime) vs A15 (over-0)   -> INCREMENTAL effect of overs 6 & 15 alone

Gate (per IDEAS.md A16):
  GATE 1: a phase-boundary prop in the ADDED overs (death: highest_over_runs;
          middle/full-innings: innings_runs, match_total_sixes) must improve
          paired vs single-vector BEYOND A15's over-0-only result -> i.e. the
          A16-vs-A15 incremental delta must show a real (CI-excludes-0) gain.
  GATE 2: top_bowler + bowler_wkts + team_first_over_mae must NOT regress.
  If {0,6,15} only reproduces A15's over-0 gain (incremental all noise) -> FAILED.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "auto"))

from a8_gate_analysis import (brier_pair, cluster_boot, load, mae_pair,  # noqa
                              paired_rows)

VEC = REPO / "models/auto/a8/detail_vec_n261.json"      # single global vector (E5 v1)
A15 = REPO / "models/auto/a15/detail_over0_n261.json"   # over-0-only
A16 = REPO / "models/auto/a16/detail_regime_n261.json"  # {0,6,15}

# Families that the ADDED overs (6=middle start, 15=death start) can move.
DEATH_MIDDLE = [
    "highest_over_runs_ou_18_5", "highest_over_runs_ou_24_5",  # single top over (death-heavy)
    "innings_runs_ou_160_5", "innings_runs_ou_170_5", "innings_runs_ou_180_5",  # full total
    "match_total_sixes_ou_15_5", "match_total_sixes_ou_20_5",  # sixes (death-heavy)
]
# GATE-2 guards (must not regress).
GUARD_BINARY = ["top_bowler", "bowler_wkts_1plus", "bowler_wkts_2plus", "bowler_wkts_3plus"]
GUARD_MAE = ["team_first_over_mae"]
# A15's established win (to confirm it is retained by the total-effect run).
FIRSTOVER_MAE = "team_first_over_mae"


def bin_line(dv, dp, fam):
    rows = paired_rows(dv, dp, fam)
    if not rows:
        return None
    ba, bb, d, (lo, hi) = brier_pair(rows)
    flag = "DOWN(better)" if hi < 0 else "UP(worse)" if lo > 0 else "~noise"
    return fam, len(rows), ba, bb, d, lo, hi, flag


def report_binary(dv, dp, fams, label_a, label_b):
    hdr = (f"{'family':<28}{'n':>7}{'Brier_'+label_a:>13}{'Brier_'+label_b:>13}"
           f"{'dBrier':>10}   95% CI ({label_b}-{label_a})   flag")
    print(hdr)
    for fam in fams:
        r = bin_line(dv, dp, fam)
        if r:
            print(f"{r[0]:<28}{r[1]:>7}{r[2]:>13.4f}{r[3]:>13.4f}{r[4]:>+10.4f}"
                  f"   [{r[5]:+.4f},{r[6]:+.4f}]  {r[7]}")


def report_mae(dv, dp, fams, label_a, label_b):
    for fam in fams:
        res = mae_pair(dv, dp, fam)
        if res:
            ma, mb, d, (lo, hi), n = res
            flag = "DOWN(better)" if hi < 0 else "UP(worse)" if lo > 0 else "~noise"
            print(f"{fam:<28}{n:>7}  MAE_{label_a}={ma:.3f}  MAE_{label_b}={mb:.3f}  "
                  f"dMAE={d:+.3f}  [{lo:+.3f},{hi:+.3f}]  {flag}")


def main():
    dv = load(str(VEC))
    d15 = load(str(A15))
    d16 = load(str(A16))
    print(f"vec: {len(dv)} | a15(over0): {len(d15)} | a16(regime): {len(d16)} matches\n")

    print("=" * 78)
    print("(1) TOTAL EFFECT: A16 regime {0,6,15}  vs  single global vector")
    print("    (should reproduce A15's over-0 gains: team_first_over_mae, pp_total_45_5)")
    print("=" * 78)
    report_binary(dv, d16, ["pp_total_ou_45_5", "pp_total_ou_50_5", "pp_total_ou_55_5"]
                  + DEATH_MIDDLE, "vec", "a16")
    print()
    report_mae(dv, d16, GUARD_MAE, "vec", "a16")

    print("\n" + "=" * 78)
    print("(2) INCREMENTAL EFFECT: A16 regime {0,6,15}  vs  A15 over-0-only")
    print("    >>> GATE 1: overs 6 & 15 must add a real gain on a death/middle prop <<<")
    print("=" * 78)
    report_binary(d15, d16, DEATH_MIDDLE + ["pp_total_ou_45_5", "pp_total_ou_50_5",
                  "pp_total_ou_55_5"], "a15", "a16")

    print("\n" + "-" * 78)
    print("GATE 2 guards (A16 vs single-vector; must NOT regress):")
    print("-" * 78)
    report_binary(dv, d16, GUARD_BINARY, "vec", "a16")
    report_mae(dv, d16, GUARD_MAE, "vec", "a16")

    print("\n" + "-" * 78)
    print("GATE 2 guards (A16 vs A15 incremental; overs 6&15 must not regress guards):")
    print("-" * 78)
    report_binary(d15, d16, GUARD_BINARY, "a15", "a16")
    report_mae(d15, d16, GUARD_MAE, "a15", "a16")


if __name__ == "__main__":
    main()
