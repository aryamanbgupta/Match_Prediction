"""D17 gate analysis — val-fit vector calibrator ON TOP OF the no-weights arm (i7).

WRITTEN AND COMMITTED BEFORE THE D17 EVAL RESULT EXISTED.

D16 LANDED: the no-class-weights i7 ball retrain (`models/auto/d16/noweights`)
passes the teacher-forced marginal audit RAW and beats the deployed-stack twin
(control booster + fresh vector calibrator) CI-clean on the pooled tail and on
`batter_runs_mae`. But its raw marginal residuals are nonzero. D17 asks the one
remaining cheap marginal question: does a `VectorScalingCalibrator` fit on the
NO-WEIGHTS arm's OWN val predictions buy anything further on top of raw?

A null here is decision-grade: it closes the marginal-calibration chain
(E5 -> A8 -> A14/A15 -> A16 -> B7 -> B8) for the structural arm and certifies
RAW as the final i7 ball stack for the I17 bundle.

Two prop_backtest detail JSONs. SAME seed (46), SAME booster, SAME encoders,
SAME engine, SAME B12-shipped B10 usage selector, n=261 x 100 sims, i7 stats
identity. The ONLY delta is the calibrator -> clean pairing (B7 precedent):

  baseline  models/auto/d16/detail_noweights_raw_s46_n261.json
            the EXISTING D16 Arm N detail, `--ball-calibrator none`.
            Byte-frozen; D17 never rewrites anything under models/auto/d16/.
  d17       models/auto/d17/detail_noweights_vec_s46_n261.json
            same arm + `models/auto/d17/vector_scaling_calibrator_d17.pkl`,
            fit on the no-weights arm's own i7 validation predictions.

The D16 calibrator (`models/auto/d16/vector_scaling_calibrator_d16.pkl`) is fit
on the CONTROL arm and is the WRONG object for this idea; it is never used here.

The D16 GATE 1(a) marginal-audit section is deliberately DROPPED — that audit
is D16's. D17's equivalent pre-run check is the fitted-vector divergence
recorded in `research/handoff/D17/raw/expectation_check.txt` BEFORE the eval
was launched.

PRE-COMMITTED GATE (per research/handoff/D17/plan.md):

  GATE 1 (improvement), BOTH conditions:
    (i)  pooled tail dBrier (d17_vec - noweights_raw) over the ROW POOL
         {pp_total_ou_45_5, pp_total_ou_50_5, pp_total_ou_55_5,
          bowler_wkts_1plus} is CI-clean negative (95% CI hi < 0);
    (ii) `batter_runs_mae` delta is NOT CI-clean positive (no regression;
         this is the exact trade the E5-era calibrator historically lost).
  GATE 2 (guards): no CI-clean positive delta on `top_bowler` dBrier OR
    `team_first_over_mae` dMAE (CI lo <= 0 on each).

  This script PRINTS the mapping; the ORCHESTRATOR issues the verdict.

Pairing machinery is imported verbatim from `b12_gate_analysis` (paired
per-row delta, cluster bootstrap BY MATCH, 2000 resamples, seed 29, rows
matched by (team, name) identity with a positional cross-check), identical to
`d16_gate_analysis`, so D17 is measured with exactly the tooling that produced
the b12 and d16 gates.

Run:
  uv run python scripts/auto/d17_gate_analysis.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "auto"))

from b12_gate_analysis import (  # noqa: E402
    N_BOOT,
    BOOT_SEED,
    flag_of,
    keyed_rows,
    load,
    paired_stat,
    positional_rows,
    report,
)
from a8_gate_analysis import cluster_boot  # noqa: E402

TAIL_POOL = ["pp_total_ou_45_5", "pp_total_ou_50_5", "pp_total_ou_55_5",
             "bowler_wkts_1plus"]
GATE1_MAE = "batter_runs_mae"
GUARD_BINARY = ["top_bowler"]
GUARD_MAE = ["team_first_over_mae"]

LABEL_A = "noweights_raw"
LABEL_B = "noweights_vec"

VERDICT_MAPPING = """\
Pre-committed verdict mapping (orchestrator applies it, you don't):
GATE1 fully met + GATE2 held -> LANDED. GATE1's pooled-tail conjunct met but
batter_runs_mae or a guard regresses CI-clean -> TABLED. Pooled tail NOT
CI-clean negative -> FAILED regardless of guards (null = chain closed --
per the idea text this negative is decision-grade, not a disappointment)."""


def pooled_tail(det_a, det_b, fams):
    """Concatenate the tail families' paired rows and score one Brier delta.

    Rows carry the match id in slot 0, so `cluster_boot` resamples whole
    matches across families jointly — the correct clustering for a pooled
    statistic. Equal weight PER ROW (the plan's 'row-pool'); an
    equal-weight-per-family variant is printed separately as context.
    """
    rows, dropped, per_fam = [], 0, {}
    for fam in fams:
        frows, fdrop = keyed_rows(det_a, det_b, fam)
        per_fam[fam] = frows
        rows.extend(frows)
        dropped += fdrop
    if not rows:
        return None
    ba = float(np.mean([(a - y) ** 2 for _, y, a, _ in rows]))
    bb = float(np.mean([(b - y) ** 2 for _, y, _, b in rows]))
    fn = lambda r: (r[3] - r[1]) ** 2 - (r[2] - r[1]) ** 2  # noqa: E731
    lo, hi = cluster_boot(rows, fn, n_boot=N_BOOT, seed=BOOT_SEED)
    return {"brier_a": ba, "brier_b": bb, "delta": bb - ba,
            "lo": lo, "hi": hi, "n": len(rows), "dropped": dropped,
            "per_fam": per_fam}


def equal_weight_family_delta(per_fam):
    """Context only: mean of the per-family dBrier, one vote per family."""
    deltas = []
    for fam, rows in per_fam.items():
        if not rows:
            continue
        da = float(np.mean([(a - y) ** 2 for _, y, a, _ in rows]))
        db = float(np.mean([(b - y) ** 2 for _, y, _, b in rows]))
        deltas.append(db - da)
    return float(np.mean(deltas)) if deltas else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline",
        default=str(REPO / "models/auto/d16/detail_noweights_raw_s46_n261.json"))
    ap.add_argument(
        "--d17",
        default=str(REPO / "models/auto/d17/detail_noweights_vec_s46_n261.json"))
    args = ap.parse_args()

    da, db = load(args.baseline), load(args.d17)
    print(f"baseline ({LABEL_A}): {len(da)} matches "
          f"({Path(args.baseline).name})")
    print(f"d17      ({LABEL_B}): {len(db)} matches "
          f"({Path(args.d17).name})")
    print(f"pairing: cluster bootstrap by match, n_boot={N_BOOT}, "
          f"seed={BOOT_SEED}; delta = {LABEL_B} - {LABEL_A} "
          f"(negative = calibrated better)\n")

    # ---------------------------------------------------------- GATE 1 (i)
    print("=" * 96)
    print("GATE 1(i) — POOLED TAIL dBrier over the row-pool "
          + ", ".join(TAIL_POOL) + " must be CI-clean negative")
    print("=" * 96)
    pool = pooled_tail(da, db, TAIL_POOL)
    gate1i = False
    if pool is None:
        print("  no rows — GATE 1(i): FAIL")
    else:
        gate1i = pool["hi"] < 0
        print(f"{'pooled tail':<34}{pool['n']:>6}{pool['dropped']:>6}"
              f"{pool['brier_a']:>12.4f}{pool['brier_b']:>12.4f}"
              f"{pool['delta']:>+10.4f}   "
              f"[{pool['lo']:+.4f},{pool['hi']:+.4f}]  "
              f"{flag_of(pool['lo'], pool['hi'])}")
        print(f"  (context, equal weight per family: "
              f"{equal_weight_family_delta(pool['per_fam']):+.4f})")
        print("\n  per-family breakdown (context; the pool is the gate):")
        report(da, db, TAIL_POOL, LABEL_A, LABEL_B, show_positional=True)
        print(f"\n  GATE 1(i): {'PASS' if gate1i else 'FAIL'}")

    # --------------------------------------------------------- GATE 1 (ii)
    print("\n" + "=" * 96)
    print(f"GATE 1(ii) — {GATE1_MAE} must NOT regress CI-clean (CI lo <= 0)")
    print("=" * 96)
    g1ii = report(da, db, [GATE1_MAE], LABEL_A, LABEL_B, show_positional=True)
    v = g1ii.get(GATE1_MAE)
    gate1ii = bool(v and not (v[1] > 0))
    print(f"  GATE 1(ii): {'PASS' if gate1ii else 'FAIL'}")

    gate1 = gate1i and gate1ii
    print(f"\n  GATE 1 (i AND ii): {'MET' if gate1 else 'NOT MET'}   "
          f"[i={gate1i} ii={gate1ii}]")

    # ------------------------------------------------------------- GATE 2
    print("\n" + "=" * 96)
    print("GATE 2 — guards: no CI-clean regression on "
          + ", ".join(GUARD_BINARY + GUARD_MAE))
    print("=" * 96)
    g2 = report(da, db, GUARD_BINARY + GUARD_MAE, LABEL_A, LABEL_B,
                show_positional=True)
    gate2 = (len(g2) == len(GUARD_BINARY) + len(GUARD_MAE)
             and all(not (val[1] > 0) for val in g2.values()))
    for fam, val in g2.items():
        print(f"  {fam:<28} {'REGRESSED CI-clean' if val[1] > 0 else 'ok'}")
    print(f"  GATE 2: {'MET' if gate2 else 'NOT MET'}")

    # ------------------------------------------------------------ CONTEXT
    print("\n" + "=" * 96)
    print("CONTEXT — full family scan (cannot flip the verdict)")
    print("=" * 96)
    all_fams = sorted(set(da[0]["obs"]) - {"cricsheet_id", "display_match_id",
                                           "match_identity_version"})
    scan = report(da, db, all_fams, LABEL_A, LABEL_B)
    clean = [(f, val) for f, val in scan.items()
             if val[1] > 0 or val[2] < 0]
    print("\n  CI-excludes-0 families (either direction):")
    if not clean:
        print("    (none)")
    for f, val in sorted(clean, key=lambda x: x[1][0]):
        print(f"    {f:<34}{val[0]:>+10.4f}  "
              f"[{val[1]:+.4f},{val[2]:+.4f}]  {val[3]}")

    print("\n" + "=" * 96)
    print(f"GATE 1: {'MET' if gate1 else 'NOT MET'} | "
          f"GATE 2: {'MET' if gate2 else 'NOT MET'}")
    if gate1 and gate2:
        mapping = "LANDED"
    elif gate1i:
        mapping = "TABLED"
    else:
        mapping = "FAILED"
    print(VERDICT_MAPPING)
    print(f"\nPre-committed verdict MAPPING (orchestrator decides): {mapping}")


if __name__ == "__main__":
    main()
