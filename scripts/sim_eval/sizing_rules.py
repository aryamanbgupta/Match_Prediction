"""Post-hoc sizing rules on an existing eval JSON.

M8 (2026-05-10) — layer betting decisions on top of model probabilities
without retraining. Supports:
- Edge-threshold filter: skip bets where max(model_edge) <= threshold.
- Per-bet outlier cap: cap stake at `cap` fraction of bank (default 0.02)
  before applying the Kelly multiplier.
- Sizing modes: flat (1 unit per bet) and fractional Kelly (default 0.25
  of full Kelly) with the per-bet cap.

Iteration-only by design — golden is held for production-launch
confirmation (per `feedback_iteration_only_decisions.md`).

Usage:
    uv run python scripts/sim_eval/sizing_rules.py \\
        --in eval_out_m7prod/hier_all_20260425_165622_w0p00.json \\
        --odds betting_odds_polymarket.json \\
        --feature-parquet data/xgb_match_data_v3_m6_unfrozen/test.parquet \\
        --min-volume 50000 \\
        --thresholds 0,0.01,0.02,0.03,0.05,0.07,0.10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "sim_eval"))

from reslice_eval_json import (  # noqa: E402
    _bootstrap_ci,
    _load_feature_lookup,
    _slice_predicate,
    SLICE_NAMES,
)


def _bet_team_and_edge(match: dict) -> Tuple[Optional[str], float]:
    edges = match.get("edge", {})
    if not edges:
        return None, 0.0
    best_team = max(edges, key=edges.get)
    return best_team, float(edges[best_team])


def _compute_pnl(match: dict, sizing: str, kelly_mult: float, cap: float
                  ) -> Optional[float]:
    """Compute pnl for this match under the requested sizing rule.

    sizing='flat': 1-unit bet on the model's preferred team.
    sizing='kelly': capped fractional Kelly: stake = min(full_kelly, cap) * kelly_mult.
    Returns None if no bet placed (insufficient odds/edge data).
    """
    best_team, edge = _bet_team_and_edge(match)
    if best_team is None:
        return None
    odds = match.get("market_odds", {}).get(best_team)
    if odds is None or odds <= 1.0:
        return None
    actual_winner = match.get("actual_winner")
    won = (actual_winner == best_team)

    if sizing == "flat":
        return float(odds - 1.0) if won else -1.0

    if sizing == "kelly":
        full_k = float(match.get("full_kelly_fraction", 0.0))
        if full_k <= 0:
            return 0.0  # no bet (no kelly edge)
        capped = min(full_k, cap)
        stake = capped * kelly_mult
        return stake * float(odds - 1.0) if won else -stake

    raise ValueError(f"Unknown sizing: {sizing}")


def evaluate(eval_json: str, odds_json: str,
              threshold: float, sizing: str,
              kelly_mult: float = 0.25, cap: float = 0.02,
              feature_parquet: Optional[Path] = None,
              slice_name: str = "all",
              min_volume: Optional[float] = None,
              n_resamples: int = 1000) -> Dict:
    """Apply sizing rules to the bets passing slice/volume filters."""
    with open(eval_json) as f:
        eval_data = json.load(f)
    with open(odds_json) as f:
        odds_data = json.load(f)
    vol_by_id = {m["match_id"]: m.get("polymarket_volume_usd")
                 for m in odds_data.get("matches", [])}
    feat_lookup = _load_feature_lookup(feature_parquet) if feature_parquet else {}
    predicate = _slice_predicate(slice_name, 15.0, 5.0)

    pnls: List[float] = []
    n_eligible = 0
    skipped_under_threshold = 0
    for match in eval_data.get("matches", []):
        if min_volume is not None:
            vol = vol_by_id.get(match["match_id"])
            if vol is None or vol < min_volume:
                continue
        feat = feat_lookup.get(match["match_id"], {})
        if not predicate(match, feat):
            continue
        n_eligible += 1

        _, edge = _bet_team_and_edge(match)
        if edge <= threshold:
            skipped_under_threshold += 1
            continue
        pnl = _compute_pnl(match, sizing, kelly_mult, cap)
        if pnl is None:
            continue
        pnls.append(pnl)

    n_bets = len(pnls)
    total_pnl = float(np.sum(pnls)) if pnls else 0.0
    roi_pct = (total_pnl / n_bets * 100.0) if n_bets else 0.0
    win_rate = sum(1 for p in pnls if p > 0) / n_bets if n_bets else 0.0
    roi_lo, roi_hi = _bootstrap_ci(pnls, n=n_resamples)

    # Drawdown (sequential by match order)
    cum = np.cumsum(pnls) if pnls else np.array([0.0])
    peak = np.maximum.accumulate(cum)
    drawdown = peak - cum
    max_dd = float(drawdown.max()) if pnls else 0.0
    max_dd_pct_of_total = (max_dd / abs(total_pnl) * 100.0
                           if total_pnl != 0 else float("nan"))

    return {
        "threshold": threshold, "sizing": sizing,
        "kelly_mult": kelly_mult if sizing == "kelly" else None,
        "cap": cap if sizing == "kelly" else None,
        "slice": slice_name, "min_volume": min_volume,
        "n_eligible": n_eligible,
        "n_bets": n_bets,
        "skipped_under_threshold": skipped_under_threshold,
        "total_pnl": total_pnl,
        "roi_pct": roi_pct,
        "roi_ci_lo": roi_lo * 100 if not np.isnan(roi_lo) else float("nan"),
        "roi_ci_hi": roi_hi * 100 if not np.isnan(roi_hi) else float("nan"),
        "win_rate": win_rate,
        "max_drawdown": max_dd,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--odds", required=True)
    ap.add_argument("--feature-parquet", type=Path, default=None)
    ap.add_argument("--slice", choices=SLICE_NAMES, default="all")
    ap.add_argument("--min-volume", type=int, default=None)
    ap.add_argument("--thresholds", default="0,0.01,0.02,0.03,0.05,0.07,0.10",
                    help="Comma-separated edge thresholds (decimals).")
    ap.add_argument("--sizing", choices=["flat", "kelly", "both"], default="both")
    ap.add_argument("--kelly-mult", type=float, default=0.25,
                    help="Kelly multiplier (default 0.25 = quarter Kelly)")
    ap.add_argument("--cap", type=float, default=0.02,
                    help="Per-bet outlier cap (default 0.02 = 2%% of bank)")
    ap.add_argument("--bootstrap-resamples", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional CSV output path.")
    args = ap.parse_args()

    thresholds = [float(s) for s in args.thresholds.split(",")]
    sizings = ["flat", "kelly"] if args.sizing == "both" else [args.sizing]

    rows = []
    for sz in sizings:
        for t in thresholds:
            r = evaluate(
                args.in_path, args.odds, threshold=t, sizing=sz,
                kelly_mult=args.kelly_mult, cap=args.cap,
                feature_parquet=args.feature_parquet,
                slice_name=args.slice, min_volume=args.min_volume,
                n_resamples=args.bootstrap_resamples,
            )
            rows.append(r)
            print(f"  sizing={sz:5s}  thr={t:.3f}  "
                  f"n_bets={r['n_bets']:3d}/{r['n_eligible']:3d}  "
                  f"ROI={r['roi_pct']:+6.2f}%  "
                  f"CI=[{r['roi_ci_lo']:+6.2f},{r['roi_ci_hi']:+6.2f}]  "
                  f"win={r['win_rate']:.1%}  "
                  f"maxDD={r['max_drawdown']:.2f}")

    if args.out:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"\n  → {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
