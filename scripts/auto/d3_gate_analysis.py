"""D3 gate analysis — empirical extras graft (sim-side half) vs the D15
canonical baseline.

Two prop_backtest detail JSONs, both venue-ON at seed 43, training-aligned
run_rate (post-D1), D2+D14+D15 attribution unit (post-D15 default path),
stale v1 global vector calibrator on BOTH sides:

  baseline   models/auto/d15/detail_d15_s43_n261.json
             canonical post-D15 baseline; flat 1%+1% extras graft
             renormalized after the calibrator (p_wide = p_no_ball =
             0.009804 per delivery).
  challenger models/auto/d3/detail_d3_s43_n261.json
             empirical extras graft: p_wide=0.037702 / p_no_ball=0.004409
             (val-split rates, builder d3_build_extras_rates.py), composed
             as a (1 - p_extras) scale on the calibrated 6-class block so
             the relative marginals are preserved exactly.

ONLY delta between the runs = the extras graft (rates + composition).

MECHANISM NOTE (pre-run): total extras mass rises 1.96% -> 4.21% per
delivery; wides quadruple, no-balls halve. Per LEGAL ball the 6-class
conditional distribution is unchanged, so off-the-bat scoring per innings
is untouched; extras runs rise ~+2.9/innings (120 x p/(1-p): 2.4 -> 5.3).
Since training labels already fold wide runs into the 6 classes (I5),
some double-count risk exists on total lines — exactly what the guards
watch.

Paired run-vs-run change in Brier/MAE, cluster-bootstrapped by match,
reusing the A8 tooling (this file lives in scripts/auto/, NOT the frozen
eval framework).

PRE-COMMITTED GATE (written before any D3 sim result existed; families
map the IDEAS.md D3 entry: "PRIMARY = simulated wide/no-ball rates match
the empirical val rates (report before/after) AND no CI-clean guard
regression (batter_runs_mae, pp_total/team-total families, top_bowler,
bowler_wkts)"):

  GATE 1 (rates): scripts/auto/d3_unit_check.py must pass, established
  BEFORE the sim run — Part 3 draws ~300k deliveries through the REAL
  T20Rules.simulate_ball and requires the simulated wide/no-ball
  frequencies to match the empirical val rates within 3-sigma
  (before: 0.009804/0.009804 flat; after: empirical val).

  GATE 2 (guards — no CI-excludes-0 regression on ANY of):
    MAE:    batter_runs_mae
    binary: pp_total_ou_{45_5,50_5,55_5},
            innings_runs_ou_{160_5,170_5,180_5},
            top_bowler, bowler_wkts_{1,2,3}plus
  ("team-total families" = the innings-runs total lines; boundary-count
  MAEs are context.)

  Verdict mapping: GATE 1 + GATE 2 -> LANDED (correctness fix; total-line
  improvements a bonus, re-baseline warning applies); exactly one ->
  TABLED; none -> FAILED.

Everything else is reported for context only and cannot flip the verdict.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "auto"))

from a8_gate_analysis import (brier_pair, cluster_boot, load, mae_pair,  # noqa
                              paired_rows)

GUARD_MAE = ["batter_runs_mae"]
GUARD_BINARY = [
    "pp_total_ou_45_5", "pp_total_ou_50_5", "pp_total_ou_55_5",
    "innings_runs_ou_160_5", "innings_runs_ou_170_5", "innings_runs_ou_180_5",
    "top_bowler", "bowler_wkts_1plus", "bowler_wkts_2plus", "bowler_wkts_3plus",
]

CONTEXT_BINARY = [
    "batter_50plus", "batter_6plus_six",
    "batter_fours_1plus", "batter_fours_2plus", "batter_fours_3plus",
    "team_highest_individual_ou_29_5", "team_highest_individual_ou_34_5",
    "team_highest_individual_ou_39_5",
    "first_wicket_runs_ou_30_5",
    "highest_over_runs_ou_18_5", "highest_over_runs_ou_24_5",
    "match_total_sixes_ou_15_5", "match_total_sixes_ou_20_5",
    "bowler_economy_ou_8_5", "bowler_economy_ou_10_5",
    "top_batter", "p_tie",
]
CONTEXT_MAE = ["team_first_over_mae", "highest_individual_mae",
               "batter_fours_mae", "team_total_fours_mae",
               "team_total_sixes_mae"]


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
    ap.add_argument("--baseline",
                    default=str(REPO / "models/auto/d15/detail_d15_s43_n261.json"),
                    help="canonical venue-ON post-D15 baseline (s43)")
    ap.add_argument("--fixed",
                    default=str(REPO / "models/auto/d3/detail_d3_s43_n261.json"),
                    help="run with the D3 empirical extras graft (s43)")
    ap.add_argument("--unit-check", choices=["met", "notmet"], required=True,
                    help="result of scripts/auto/d3_unit_check.py "
                         "(GATE 1: simulated rates == empirical val rates), "
                         "established before the sim run")
    args = ap.parse_args()

    da, db = load(args.baseline), load(args.fixed)
    print(f"baseline (flat graft, venue-ON s43): {len(da)} matches | "
          f"fixed (D3 empirical graft, venue-ON s43): {len(db)} matches\n")

    gate1 = args.unit_check == "met"
    print(f"GATE 1 (simulated wide/no-ball rates match empirical val rates; "
          f"unit check): {'MET' if gate1 else 'NOT MET'} (established pre-run)\n")

    print("=" * 84)
    print("GATE 2 — GUARDS: no CI-excludes-0 regression on any")
    print("=" * 84)
    fg_m = report_mae(da, db, GUARD_MAE, "base", "d3")
    fg_b = report_binary(da, db, GUARD_BINARY, "base", "d3")
    allg = {**fg_m, **fg_b}
    regressed = [f for f, v in allg.items() if v[1] > 0]
    gate2 = (not regressed
             and len(allg) == len(GUARD_MAE) + len(GUARD_BINARY))
    print(f"\n  GATE 2 (guards held): {'MET' if gate2 else 'NOT MET'}"
          + (f" — CI-clean regressions: {regressed}" if regressed else ""))

    print("\n" + "=" * 84)
    print("CONTEXT (reported for completeness; cannot flip the verdict)")
    print("=" * 84)
    report_binary(da, db, CONTEXT_BINARY, "base", "d3")
    report_mae(da, db, CONTEXT_MAE, "base", "d3")

    print("\n" + "=" * 84)
    print(f"GATE 1 (rates match, unit check): {'MET' if gate1 else 'NOT MET'}")
    print(f"GATE 2 (guards held): {'MET' if gate2 else 'NOT MET'}")
    n_met = int(gate1) + int(gate2)
    verdict = {2: "LANDED", 1: "TABLED", 0: "FAILED"}[n_met]
    print(f"VERDICT: {verdict}")
    print("=" * 84)


if __name__ == "__main__":
    main()
