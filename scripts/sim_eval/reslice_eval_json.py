"""
Post-hoc re-slicer for previously saved match-level eval JSONs.

Loads an existing eval result file (with per-match `log_loss` /
`realized_pnl` etc.) and a polymarket-style odds file (with
`polymarket_volume_usd`), and produces 3 sliced summaries (all / >=$50k
/ >=$100k) using the same bootstrap-CI helper that powers
run_sim_eval.py. This avoids ~30 min of redundant re-eval compute when
the only change is the liquidity filter — useful for back-filling
sliced metrics on already-trained models (e.g., the v4 post-fix
baseline at eval_out_postfix/xgboost_20260421_220541.json).

Usage:
    uv run python scripts/sim_eval/reslice_eval_json.py \\
        --in  eval_out_postfix/xgboost_20260421_220541.json \\
        --odds betting_odds_polymarket.json \\
        --out-dir eval_out_phase1_sliced_v4
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


def _bootstrap_ci(values: List[float], n: int = 1000, ci: float = 0.95,
                  seed: int = 42) -> tuple:
    if not values:
        return (float('nan'), float('nan'))
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n, len(arr)))
    means = arr[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    return (float(np.quantile(means, alpha)),
            float(np.quantile(means, 1 - alpha)))


def reslice(eval_json_path: str, odds_json_path: str,
            min_volume: Optional[float], n_resamples: int = 1000) -> Dict:
    """Recompute summary stats over the slice {match: vol >= min_volume}."""
    with open(eval_json_path) as f:
        eval_data = json.load(f)
    with open(odds_json_path) as f:
        odds_data = json.load(f)

    # Build match_id → volume lookup (None if field absent).
    vol_by_id = {}
    for m in odds_data.get('matches', []):
        vol_by_id[m['match_id']] = m.get('polymarket_volume_usd')

    matches = eval_data.get('matches', [])
    kept_matches = []
    for match in matches:
        if min_volume is None:
            kept_matches.append(match)
            continue
        vol = vol_by_id.get(match['match_id'])
        if vol is None or vol < min_volume:
            continue
        kept_matches.append(match)

    log_losses = [m['log_loss'] for m in kept_matches
                  if m.get('log_loss') is not None and not (
                      isinstance(m['log_loss'], float) and np.isnan(m['log_loss']))]
    brier_scores = [m['brier_score'] for m in kept_matches
                    if m.get('brier_score') is not None and not (
                        isinstance(m['brier_score'], float) and np.isnan(m['brier_score']))]

    # Flat-betting P&L: only matches where a bet was placed (realized_pnl != 0).
    flat_returns = [m['realized_pnl'] for m in kept_matches
                    if m.get('realized_pnl') not in (None, 0, 0.0)]

    # CI on per-match log loss; CI on flat P&L (then ×100 for ROI).
    ll_lo, ll_hi = _bootstrap_ci(log_losses, n=n_resamples)
    pl_lo, pl_hi = _bootstrap_ci(flat_returns, n=n_resamples)

    avg_log_loss = float(np.mean(log_losses)) if log_losses else float('nan')
    avg_brier = float(np.mean(brier_scores)) if brier_scores else float('nan')

    total_pnl = float(np.sum(flat_returns)) if flat_returns else 0.0
    bets_placed = len(flat_returns)
    flat_roi_pct = (total_pnl / bets_placed * 100.0) if bets_placed else 0.0
    flat_roi_ci_low = pl_lo * 100 if not np.isnan(pl_lo) else float('nan')
    flat_roi_ci_high = pl_hi * 100 if not np.isnan(pl_hi) else float('nan')

    win_rate = (sum(1 for r in flat_returns if r > 0) / bets_placed) if bets_placed else 0.0

    if min_volume is None:
        slice_tag = "all"
    else:
        slice_tag = f"min_volume_{int(min_volume)}"

    summary = {
        'reslice_source': str(Path(eval_json_path).resolve()),
        'reslice_odds':   str(Path(odds_json_path).resolve()),
        'slice':          slice_tag,
        'min_volume':     min_volume,
        'n_matches_in_source': len(matches),
        'n_matches_evaluated': len(kept_matches),
        'avg_log_loss':            avg_log_loss,
        'avg_log_loss_ci_low':     ll_lo,
        'avg_log_loss_ci_high':    ll_hi,
        'avg_brier_score':         avg_brier,
        'flat_betting_total_pnl':  total_pnl,
        'flat_betting_roi_pct':    flat_roi_pct,
        'flat_betting_roi_ci_low': flat_roi_ci_low,
        'flat_betting_roi_ci_high': flat_roi_ci_high,
        'flat_betting_win_rate':   win_rate,
        'flat_betting_bets_placed': bets_placed,
    }

    return {'summary': summary, 'matches': kept_matches}


def main():
    parser = argparse.ArgumentParser(description='Re-slice an existing match-eval JSON by polymarket liquidity.')
    parser.add_argument('--in', dest='in_path', required=True, help='Path to eval results JSON to re-slice.')
    parser.add_argument('--odds', required=True, help='Polymarket-style odds JSON with polymarket_volume_usd.')
    parser.add_argument('--out-dir', required=True, help='Output directory for sliced JSONs.')
    parser.add_argument('--bootstrap-resamples', type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_stem = Path(args.in_path).stem

    for min_vol in (None, 50_000, 100_000):
        result = reslice(args.in_path, args.odds, min_vol,
                         n_resamples=args.bootstrap_resamples)
        slice_tag = result['summary']['slice']
        out_path = out_dir / f"{src_stem}_{slice_tag}.json"
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)

        s = result['summary']
        print(f"\n--- Slice: {slice_tag} ---")
        print(f"  Matches: {s['n_matches_evaluated']} / {s['n_matches_in_source']}")
        print(f"  Avg Log Loss: {s['avg_log_loss']:.4f}  "
              f"[95% CI: {s['avg_log_loss_ci_low']:.4f}, {s['avg_log_loss_ci_high']:.4f}]")
        print(f"  Flat ROI: {s['flat_betting_roi_pct']:+.2f}%  "
              f"[95% CI: {s['flat_betting_roi_ci_low']:+.2f}%, {s['flat_betting_roi_ci_high']:+.2f}%]")
        print(f"  Bets placed: {s['flat_betting_bets_placed']}, win rate: {s['flat_betting_win_rate']:.1%}")
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()
