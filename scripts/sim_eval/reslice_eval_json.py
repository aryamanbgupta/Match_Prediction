"""
Post-hoc re-slicer for previously saved match-level eval JSONs.

Loads an existing eval result file (with per-match `log_loss` /
`realized_pnl` etc.) and a polymarket-style odds file (with
`polymarket_volume_usd`), and produces sliced summaries using the shared I3
competition-block bootstrap. Output rows are upgraded with explicit
`bet_placed`, `bet_team`, and `competition_cluster_id` fields. Default loop
emits all / ≥$50k / ≥$100k volume slices; pass --slice to apply an additional
predicate (IPL-only, international-only, mismatch, close).

Adversarial slices (M1, 2026-05-10) require feature-row joining for
is_international / top6_batting_elo_diff / competition_tier — pass
--feature-parquet pointing at the materialized parquet that produced the
predictions JSON. --stratify-by tier_x_half enables stratified bootstrap
(stratum = competition_tier × early/late half of the match-date range).

Usage:
    uv run python scripts/sim_eval/reslice_eval_json.py \\
        --in  eval_out/postfix/xgboost_20260421_220541.json \\
        --odds betting_odds_polymarket.json \\
        --out-dir eval_out/phase1_sliced_v4
    uv run python scripts/sim_eval/reslice_eval_json.py \\
        --in  ... --odds ... --out-dir ... \\
        --slice ipl --feature-parquet data/xgb_match_data_v2_clean/test.parquet \\
        --stratify-by tier_x_half
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from sim_eval.eval_statistics import (  # noqa: E402
    BOOTSTRAP_CONTRACT_VERSION,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    MIN_RECOMMENDED_CLUSTERS,
    bootstrap_mean_ci,
    cluster_id_for_record,
    count_unique_clusters,
    flat_bet_team,
    flat_bet_won,
    load_competition_clusters,
)


SLICE_NAMES = ("all", "ipl", "international", "mismatch", "close")

# IPL franchises 2025–2026 era. A match counts as IPL iff BOTH teams are in
# this set. competition_tier alone won't do it — tier 3 also covers MLC, CPL,
# BBL, Hundred, SA20, T20 Blast.
_IPL_TEAMS = frozenset({
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Royal Challengers Bangalore",  # legacy name pre-2024 rebrand
    "Sunrisers Hyderabad",
})


def _bootstrap_ci(values: List[float], n: int = 1000, ci: float = 0.95,
                  seed: int = DEFAULT_BOOTSTRAP_SEED,
                  strata: Optional[List] = None,
                  clusters: Optional[List] = None) -> tuple:
    """Compatibility wrapper around the shared I3 bootstrap."""
    return bootstrap_mean_ci(
        values,
        n_resamples=n,
        ci=ci,
        seed=seed,
        clusters=clusters,
        strata=strata,
    )


def _load_feature_lookup(feature_parquet: Optional[Path]) -> Dict[str, Dict]:
    """Build match_id → {team1, team2, is_international, top6_batting_elo_diff,
    competition_tier, match_date} lookup from the parquet that produced
    the predictions JSON. Empty dict if no parquet given.
    """
    if feature_parquet is None:
        return {}
    import pandas as pd
    df = pd.read_parquet(feature_parquet)
    cols = ["match_id", "team1", "team2", "is_international",
            "top6_batting_elo_diff", "competition_tier", "match_date"]
    have = [c for c in cols if c in df.columns]
    return {row["match_id"]: {c: row[c] for c in have if c != "match_id"}
            for _, row in df[have].iterrows()}


def _slice_predicate(slice_name: str, mismatch_thresh: float,
                     close_thresh: float):
    """Return a function (match, feat_row) -> bool. feat_row may be {}.
    Team names live on the match object (eval JSON preserves the
    materializer fields); ELO/tier/intl come from the joined feature row.
    """
    if slice_name == "all":
        return lambda m, f: True
    if slice_name == "ipl":
        # Eval JSON match objects use `teams` (list); parquet feat_row
        # provides `team1`/`team2`. Prefer parquet when joined.
        def _is_ipl(m, f):
            t1 = f.get("team1")
            t2 = f.get("team2")
            if t1 is None or t2 is None:
                teams = m.get("teams") or []
                if len(teams) == 2:
                    t1, t2 = teams[0], teams[1]
            return t1 in _IPL_TEAMS and t2 in _IPL_TEAMS
        return _is_ipl
    if slice_name == "international":
        return lambda m, f: bool(f.get("is_international", 0))
    if slice_name == "mismatch":
        return lambda m, f: abs(f.get("top6_batting_elo_diff", 0.0)) >= mismatch_thresh
    if slice_name == "close":
        return lambda m, f: abs(f.get("top6_batting_elo_diff", 0.0)) <= close_thresh
    raise ValueError(f"Unknown slice: {slice_name}")


def _build_strata(matches: List[dict], feat_lookup: Dict[str, Dict],
                  mode: str) -> Optional[List]:
    """Build a per-match stratum label list. Mode 'tier_x_half' splits
    on (competition_tier, early/late half of date range). Returns None
    if mode is None or unknown.
    """
    if mode is None or mode == "none":
        return None
    if mode != "tier_x_half":
        raise ValueError(f"Unknown stratify mode: {mode}")
    dates = sorted(set(
        feat_lookup.get(m["match_id"], {}).get("match_date")
        for m in matches
        if feat_lookup.get(m["match_id"], {}).get("match_date") is not None
    ))
    if not dates:
        return None
    median_date = dates[len(dates) // 2]
    strata = []
    for m in matches:
        f = feat_lookup.get(m["match_id"], {})
        tier = f.get("competition_tier", "unknown")
        d = f.get("match_date")
        half = "early" if d is not None and d <= median_date else "late"
        strata.append((tier, half))
    return strata


def reslice(eval_json_path: str, odds_json_path: str,
            min_volume: Optional[float],
            n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
            slice_name: str = "all",
            feature_parquet: Optional[Path] = None,
            mismatch_thresh: float = 15.0,
            close_thresh: float = 5.0,
            stratify_by: Optional[str] = None,
            cluster_source_dir: Optional[Path] = None) -> Dict:
    """Recompute summary stats over the slice {match: vol >= min_volume
    AND predicate(slice_name)}.
    """
    with open(eval_json_path) as f:
        eval_data = json.load(f)
    with open(odds_json_path) as f:
        odds_data = json.load(f)

    vol_by_id = {m['match_id']: m.get('polymarket_volume_usd')
                 for m in odds_data.get('matches', [])}
    feat_lookup = _load_feature_lookup(feature_parquet)
    predicate = _slice_predicate(slice_name, mismatch_thresh, close_thresh)
    if cluster_source_dir is None:
        default_cluster_source = PROJECT_ROOT / "data" / "polymarket_test"
        cluster_source_dir = (
            default_cluster_source if default_cluster_source.is_dir() else None
        )
    cluster_lookup = (
        load_competition_clusters(cluster_source_dir)
        if cluster_source_dir is not None
        else {}
    )

    matches = eval_data.get('matches', [])
    kept_matches = []
    for match in matches:
        if min_volume is not None:
            vol = vol_by_id.get(match['match_id'])
            if vol is None or vol < min_volume:
                continue
        feat = feat_lookup.get(match['match_id'], {})
        if not predicate(match, feat):
            continue
        bet_team = flat_bet_team(match)
        enriched = dict(match)
        enriched["bet_placed"] = bet_team is not None
        enriched["bet_team"] = bet_team
        enriched["competition_cluster_id"] = cluster_id_for_record(
            match,
            cluster_lookup,
        )
        kept_matches.append(enriched)

    def _is_valid_ll(m):
        ll = m.get('log_loss')
        return ll is not None and not (isinstance(ll, float) and np.isnan(ll))

    def _has_bet(m):
        return flat_bet_team(m) is not None

    ll_matches = [m for m in kept_matches if _is_valid_ll(m)]
    log_losses = [m['log_loss'] for m in ll_matches]
    brier_scores = [m['brier_score'] for m in kept_matches
                    if m.get('brier_score') is not None and not (
                        isinstance(m['brier_score'], float) and np.isnan(m['brier_score']))]

    flat_betting_matches = [m for m in kept_matches if _has_bet(m)]
    flat_returns = [m['realized_pnl'] for m in flat_betting_matches]

    # Strata are filtered to the same subset they're scoring against.
    ll_strata = _build_strata(ll_matches, feat_lookup, stratify_by) \
        if stratify_by else None
    roi_strata = _build_strata(flat_betting_matches, feat_lookup, stratify_by) \
        if stratify_by else None
    ll_clusters = [
        cluster_id_for_record(match, cluster_lookup)
        for match in ll_matches
    ]
    roi_clusters = [
        cluster_id_for_record(match, cluster_lookup)
        for match in flat_betting_matches
    ]

    ll_lo, ll_hi = _bootstrap_ci(
        log_losses,
        n=n_resamples,
        strata=ll_strata,
        clusters=ll_clusters,
    )
    pl_lo, pl_hi = _bootstrap_ci(
        flat_returns,
        n=n_resamples,
        strata=roi_strata,
        clusters=roi_clusters,
    )

    avg_log_loss = float(np.mean(log_losses)) if log_losses else float('nan')
    avg_brier = float(np.mean(brier_scores)) if brier_scores else float('nan')

    total_pnl = float(np.sum(flat_returns)) if flat_returns else 0.0
    bets_placed = len(flat_returns)
    flat_roi_pct = (total_pnl / bets_placed * 100.0) if bets_placed else 0.0
    flat_roi_ci_low = pl_lo * 100 if not np.isnan(pl_lo) else float('nan')
    flat_roi_ci_high = pl_hi * 100 if not np.isnan(pl_hi) else float('nan')

    win_rate = (
        sum(1 for match in flat_betting_matches if flat_bet_won(match))
        / bets_placed
        if bets_placed else 0.0
    )

    vol_tag = "all" if min_volume is None else f"min_volume_{int(min_volume)}"
    slice_tag = vol_tag if slice_name == "all" else f"{slice_name}_{vol_tag}"

    summary = {
        'reslice_source': str(Path(eval_json_path).resolve()),
        'reslice_odds':   str(Path(odds_json_path).resolve()),
        'slice':          slice_tag,
        'slice_name':     slice_name,
        'min_volume':     min_volume,
        'mismatch_threshold': mismatch_thresh if slice_name == "mismatch" else None,
        'close_threshold':    close_thresh if slice_name == "close" else None,
        'stratify_by':    stratify_by,
        'bootstrap_contract': BOOTSTRAP_CONTRACT_VERSION,
        'bootstrap_seed': DEFAULT_BOOTSTRAP_SEED,
        'bootstrap_resamples': n_resamples,
        'n_bootstrap_clusters': count_unique_clusters(roi_clusters),
        'bootstrap_reliable': (
            count_unique_clusters(roi_clusters)
            >= MIN_RECOMMENDED_CLUSTERS
        ),
        'cluster_source_dir': (
            str(Path(cluster_source_dir).resolve())
            if cluster_source_dir is not None else None
        ),
        'cluster_metadata_coverage': sum(
            match['match_id'] in cluster_lookup for match in kept_matches
        ),
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
    parser = argparse.ArgumentParser(description='Re-slice an existing match-eval JSON by polymarket liquidity and adversarial slice predicate.')
    parser.add_argument('--in', dest='in_path', required=True, help='Path to eval results JSON to re-slice.')
    parser.add_argument('--odds', required=True, help='Polymarket-style odds JSON with polymarket_volume_usd.')
    parser.add_argument('--out-dir', required=True, help='Output directory for sliced JSONs.')
    parser.add_argument(
        '--bootstrap-resamples',
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    parser.add_argument('--slice', choices=SLICE_NAMES, default="all",
                        help='Adversarial slice predicate. Composes with '
                        '--min-volume (intersection). Default: all.')
    parser.add_argument('--mismatch-threshold', type=float, default=15.0,
                        help='|top6_batting_elo_diff| >= this is "mismatch". '
                        'Default 15.0 ≈ q90 of |diff| on the iteration test set; '
                        'top6 ELO is averaged over 6 batters so absolute diffs '
                        'are ~10x smaller than per-player ELO.')
    parser.add_argument('--close-threshold', type=float, default=5.0,
                        help='|top6_batting_elo_diff| <= this is "close". '
                        'Default 5.0 ≈ median of |diff| on iteration test.')
    parser.add_argument('--feature-parquet', type=Path, default=None,
                        help='Parquet path with match_id + is_international + '
                        'top6_batting_elo_diff + competition_tier + match_date '
                        'for slice predicates and stratification. '
                        'E.g. data/xgb_match_data_v2_clean/test.parquet.')
    parser.add_argument('--stratify-by', choices=("none", "tier_x_half"),
                        default="none",
                        help='Stratify the bootstrap by '
                        '(competition_tier, early/late half). Default: none.')
    parser.add_argument(
        '--cluster-source-dir',
        type=Path,
        default=None,
        help=(
            'Cricsheet JSON directory used for tournament/tour-season block '
            'labels. Defaults to data/polymarket_test when present; unmatched '
            'rows use team-pair-season blocks.'
        ),
    )
    parser.add_argument('--min-volume', type=int, action="append", default=None,
                        help='Volume threshold(s). Repeat for multiple. If '
                        'omitted, defaults to (None, 50000, 100000) — three '
                        'slices in one call (back-compat).')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_stem = Path(args.in_path).stem
    stratify = None if args.stratify_by == "none" else args.stratify_by

    if args.min_volume is None:
        min_vol_list = [None, 50_000, 100_000]
    else:
        min_vol_list = list(args.min_volume)

    for min_vol in min_vol_list:
        result = reslice(args.in_path, args.odds, min_vol,
                         n_resamples=args.bootstrap_resamples,
                         slice_name=args.slice,
                         feature_parquet=args.feature_parquet,
                         mismatch_thresh=args.mismatch_threshold,
                         close_thresh=args.close_threshold,
                         stratify_by=stratify,
                         cluster_source_dir=args.cluster_source_dir)
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
        print(
            f"  Bootstrap: {s['bootstrap_contract']} "
            f"({s['n_bootstrap_clusters']} bet clusters)"
        )
        if not s["bootstrap_reliable"]:
            print("  WARNING: fewer than 10 clusters; CI is descriptive only")
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()
