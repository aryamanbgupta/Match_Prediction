"""D18 gate analysis — no-weights-adapted hyperparameters vs the D16 arm (i7).

WRITTEN AND COMMITTED BEFORE THE D18 EVAL RESULT EXISTED.

D16 certified no-weights RAW (lr 0.2404, best_iteration 24 of 444) as the
i7 ball stack; D17 closed the calibrator question. D18 asks the last cheap
question about that arm: the i7 config was swept under balanced weights, and
early stopping's cut to 24 trees says it is over-aggressive for the
uniform-weight loss surface. Arms lr {0.025, 0.05, 0.10} were trained with
`--no-class-weights` (scripts/auto/d18_train_arms.py); selection is
VAL-LL-ONLY (D8/E4 discipline) and the winner had to beat D16's no-weights
val mlogloss 1.4334 before any sim eval was allowed.

Two prop_backtest detail JSONs. SAME seed (46), SAME engine, SAME
B12-shipped B10 usage selector, n=261 x 100 sims, i7 stats identity. The
ONLY delta is the booster's hyperparameters (frame-derived encoder sidecars
are asserted byte-identical before the eval):

  baseline  models/auto/d16/detail_noweights_raw_s46_n261.json
            the EXISTING D16 Arm N detail, `--ball-calibrator none`.
            Byte-frozen; D18 never rewrites anything under models/auto/d16/.
  d18       models/auto/d18/detail_<winner>_raw_s46_n261.json
            the val-LL winner arm, also raw (D17 closed the calibrator
            chain — no calibrator belongs on either side).

PRE-COMMITTED GATE (per research/IDEAS.md D18):

  GATE 1 (improvement), BOTH conditions:
    (i)  pooled tail dBrier (d18 - noweights_raw) over the ROW POOL
         {pp_total_ou_45_5, pp_total_ou_50_5, pp_total_ou_55_5,
          bowler_wkts_1plus} is CI-clean negative (95% CI hi < 0);
    (ii) `batter_runs_mae` delta is NOT CI-clean positive (no regression).
  GATE 2 (guards): no CI-clean positive delta on `top_bowler` dBrier OR
    `team_first_over_mae` dMAE (CI lo <= 0 on each).

  Mapping: GATE 1 fully met + GATE 2 held -> LANDED (winner supersedes the
  D16 arm as the I17-bundle ball stack). GATE 1(i) met but (ii)/a guard
  regresses CI-clean -> TABLED. Pooled tail NOT CI-clean negative -> FAILED
  (the D16 lr 0.2404 arm stands; better val LL did not transfer to props).

  This script PRINTS the mapping; the ORCHESTRATOR issues the verdict.

Pairing machinery is imported verbatim from `b12_gate_analysis` (paired
per-row delta, cluster bootstrap BY MATCH, 2000 resamples, seed 29), the
exact tooling that produced the b12/d16/d17 gates.

Run:
  uv run python scripts/auto/d18_gate_analysis.py --d18 <detail json>
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

LABEL_A = "d16_noweights_raw"
LABEL_B = "d18_winner_raw"

VERDICT_MAPPING = """\
Pre-committed verdict mapping (orchestrator applies it, you don't):
GATE1 fully met + GATE2 held -> LANDED (winner supersedes the D16 arm in the
I17 bundle). GATE1's pooled-tail conjunct met but batter_runs_mae or a guard
regresses CI-clean -> TABLED. Pooled tail NOT CI-clean negative -> FAILED
(D16's lr 0.2404 no-weights arm stands)."""


def pooled_tail(det_a, det_b, fams):
    """Concatenate the tail families' paired rows and score one Brier delta.

    Rows carry the match id in slot 0, so `cluster_boot` resamples whole
    matches across families jointly. Equal weight PER ROW (the row-pool the
    d16/d17 gates used); the equal-weight-per-family reading is printed as
    context per D17's honest-note precedent.
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
        "--d18",
        required=True,
        help="detail JSON of the val-LL winner arm (raw, seed 46, n=261x100)")
    args = ap.parse_args()

    da, db = load(args.baseline), load(args.d18)
    print(f"baseline ({LABEL_A}): {len(da)} matches "
          f"({Path(args.baseline).name})")
    print(f"d18      ({LABEL_B}): {len(db)} matches "
          f"({Path(args.d18).name})")
    print(f"pairing: cluster bootstrap by match, n_boot={N_BOOT}, "
          f"seed={BOOT_SEED}; delta = {LABEL_B} - {LABEL_A} "
          f"(negative = d18 better)\n")

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
