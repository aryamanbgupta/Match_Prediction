"""A11 — A7 boundary sweep (betting-layer, A7 follow-up).

A7 landed the slice-conditional edge threshold with the mismatch/close boundary
fixed at |top6_batting_elo_diff| = 5 (inherited from the M8 write-up) and
edge > 10%. The edge threshold was swept (A7: 0.05/0.10/0.15 all improved) but
the *boundary* was never varied. A11 sweeps the boundary at {3, 5, 8, 12} with
edge > 10% held fixed, on BOTH the >=$50k and >=$100k slices, and applies a
pre-committed decision rule so we do not slice-shop.

PRE-COMMITTED DECISION RULE (fixed a priori, matches research/IDEAS.md A11):
  A7's landed boundary = 5 stays production UNLESS a challenger boundary beats
  it on BOTH >=$50k and >=$100k ROI by more than the 2pp floor AND keeps the
  >=$50k CI lower bound excluding 0. Otherwise boundary 5 stands -> A11 FAILED
  (no change), reported as the betting-layer null.

Pure eval composition: reuses `scripts/auto/a7_conditional_threshold.py::run`
(which itself reuses the eval framework's read-only helpers) on the FROZEN
production sliced eval JSONs. No retraining, no probability changes, no
eval-framework edit; log-loss is unchanged by construction (0.6299).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "auto"))

from a7_conditional_threshold import run  # noqa: E402

FEATURE_PARQUET = PROJECT_ROOT / "data/xgb_match_data_v3_m6_unfrozen/test.parquet"
EVAL_50K = (PROJECT_ROOT /
            "eval_out_m7prod_sliced/hier_all_20260425_165622_w0p00_min_volume_50000.json")
EVAL_100K = (PROJECT_ROOT /
             "eval_out_m7prod_sliced/hier_all_20260425_165622_w0p00_min_volume_100000.json")

BOUNDARIES = [3.0, 5.0, 8.0, 12.0]
EDGE_THR = 0.10          # held fixed (A7's landed threshold)
INCUMBENT = 5.0          # A7's production boundary
ROI_FLOOR_PP = 2.0       # betting-layer noise floor (program.md)


def sweep(eval_json: Path) -> Dict[float, dict]:
    """boundary -> variant summary dict at EDGE_THR."""
    out: Dict[float, dict] = {}
    for b in BOUNDARIES:
        res = run(str(eval_json), FEATURE_PARQUET, b, [EDGE_THR])
        out[b] = res["variants"][0]
        out["_baseline"] = res["baseline"]  # identical across boundaries
    return out


def _fmt(v: dict) -> str:
    return (f"n={v['n_bets']:>3d} close={v['n_close']:>3d} "
            f"mis_keep={v['n_mismatch_kept']:>3d} mis_drop={v['n_mismatch_dropped']:>3d} "
            f"ROI={v['roi_pct']:>+7.2f}% CI=[{v['roi_ci_lo']:>+6.2f},{v['roi_ci_hi']:>+6.2f}] "
            f"win={v['win_rate']:>5.1%} maxDD={v['max_drawdown']:>5.2f}")


def main() -> int:
    s50 = sweep(EVAL_50K)
    s100 = sweep(EVAL_100K)

    b50 = s50["_baseline"]
    b100 = s100["_baseline"]
    print("=" * 100)
    print("A11 — boundary sweep (edge>10% fixed). Betting-layer; LL unchanged (0.6299).")
    print("=" * 100)
    print(f"\nBaseline (flat thr 0 = production):")
    print(f"  >=$50k : n={b50['n_bets']} ROI={b50['roi_pct']:+.2f}% "
          f"CI=[{b50['roi_ci_lo']:+.2f},{b50['roi_ci_hi']:+.2f}]")
    print(f"  >=$100k: n={b100['n_bets']} ROI={b100['roi_pct']:+.2f}% "
          f"CI=[{b100['roi_ci_lo']:+.2f},{b100['roi_ci_hi']:+.2f}]")

    for name, s in [(">=$50k", s50), (">=$100k", s100)]:
        print(f"\n--- {name} (conditional: mismatch=|elo_diff|>b requires edge>10%) ---")
        for b in BOUNDARIES:
            marker = " <- incumbent(A7)" if b == INCUMBENT else ""
            print(f"  b={b:>4.0f}: {_fmt(s[b])}{marker}")

    # --- pre-committed decision ---
    inc50 = s50[INCUMBENT]
    inc100 = s100[INCUMBENT]
    print("\n" + "=" * 100)
    print("PRE-COMMITTED DECISION (boundary 5 stands unless a challenger beats it on")
    print("BOTH slices' ROI by >2pp AND keeps >=$50k CI lower bound > 0):")
    print("=" * 100)
    winner = None
    for b in BOUNDARIES:
        if b == INCUMBENT:
            continue
        d50 = s50[b]["roi_pct"] - inc50["roi_pct"]
        d100 = s100[b]["roi_pct"] - inc100["roi_pct"]
        ci_ok = s50[b]["roi_ci_lo"] > 0.0
        beats = (d50 > ROI_FLOOR_PP) and (d100 > ROI_FLOOR_PP) and ci_ok
        print(f"  b={b:>4.0f}: d50k={d50:>+7.2f}pp  d100k={d100:>+7.2f}pp  "
              f"50k_CI_lo={s50[b]['roi_ci_lo']:>+6.2f}  "
              f"beats_bar={'YES' if beats else 'no'}")
        if beats:
            winner = b
    print("-" * 100)
    if winner is None:
        print(f"VERDICT: No challenger clears the dual-slice bar. Boundary {INCUMBENT:.0f} "
              f"stands (A7 rule unchanged). A11 = FAILED (betting-layer null).")
    else:
        print(f"VERDICT: Boundary {winner:.0f} clears the pre-committed bar and replaces 5.")

    out = {
        "edge_thr": EDGE_THR, "incumbent": INCUMBENT, "roi_floor_pp": ROI_FLOOR_PP,
        "baseline_50k": b50, "baseline_100k": b100,
        "sweep_50k": {str(b): s50[b] for b in BOUNDARIES},
        "sweep_100k": {str(b): s100[b] for b in BOUNDARIES},
        "winner": winner,
    }
    out_path = PROJECT_ROOT / "models/auto/a11/a11_boundary_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
