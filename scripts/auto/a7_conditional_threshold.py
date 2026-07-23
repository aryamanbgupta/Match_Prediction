"""A7 — slice-conditional edge threshold (betting-layer, M8 follow-up).

Pure betting-layer rule on top of the FROZEN production predictions. Does NOT
retrain and does NOT touch model probabilities, so log-loss is unchanged by
construction. Only the *set of bets placed* changes.

Rule (A7 as stated in research/IDEAS.md):
  - close fixtures   (|top6_batting_elo_diff| <= `boundary`): bet flat, threshold 0.
  - mismatch fixtures (|top6_batting_elo_diff|  > `boundary`): bet only when the
    model's edge on its preferred team exceeds `edge_thr` (~10%).

Everything reuses the eval framework's own helpers (read-only import) so the
methodology is identical to production:
  - per-bet pnl        = the framework's `realized_pnl` field (flat 1-unit,
                         threshold-0 production betting decision).
  - bet/win decision   = explicit framework helpers, never P&L sentinels.
  - CI                 = I3 whole-competition bootstrap (seed 42, n=10,000).
  - slice membership   = `top6_batting_elo_diff` via `_load_feature_lookup`;
                         missing feature row -> diff 0.0 (framework default) ->
                         treated as close (keep the bet).

Baseline = keep every bet (== flat threshold 0 == production).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "sim_eval"))

from reslice_eval_json import _bootstrap_ci, _load_feature_lookup  # noqa: E402
from eval_statistics import (  # noqa: E402
    BOOTSTRAP_CONTRACT_VERSION,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    MIN_RECOMMENDED_CLUSTERS,
    cluster_id_for_record,
    flat_bet_team,
    flat_bet_won,
    load_competition_clusters,
)


def _max_edge(match: dict) -> Optional[float]:
    """Model edge on its preferred (max-edge) team; None if no edge data."""
    edges = match.get("edge", {})
    if not edges:
        return None
    numeric = []
    for value in edges.values():
        try:
            edge = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(edge):
            numeric.append(edge)
    return max(numeric) if numeric else None


def _summarize(
    pnls: List[float],
    clusters: List[str],
    wins: List[bool],
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> Dict:
    if not (len(pnls) == len(clusters) == len(wins)):
        raise ValueError("pnls, clusters, and wins must have equal length")
    n = len(pnls)
    total = float(np.sum(pnls)) if pnls else 0.0
    roi = (total / n * 100.0) if n else 0.0
    win = (sum(wins) / n) if n else 0.0
    lo, hi = _bootstrap_ci(
        pnls,
        n=n_resamples,
        clusters=clusters,
    )
    if pnls:
        cum = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum)
        max_dd = float((peak - cum).max())
    else:
        max_dd = 0.0
    return {
        "n_bets": n, "total_pnl": total, "roi_pct": roi,
        "roi_ci_lo": lo * 100 if not np.isnan(lo) else float("nan"),
        "roi_ci_hi": hi * 100 if not np.isnan(hi) else float("nan"),
        "win_rate": win, "max_drawdown": max_dd,
        "bootstrap_contract": BOOTSTRAP_CONTRACT_VERSION,
        "n_bootstrap_clusters": len(set(clusters)),
        "bootstrap_reliable": (
            len(set(clusters)) >= MIN_RECOMMENDED_CLUSTERS
        ),
    }


def run(eval_json: str, feature_parquet: Path, boundary: float,
        edge_thrs: List[float],
        n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
        cluster_source_dir: Optional[Path] = None) -> Dict:
    with open(eval_json) as f:
        matches = json.load(f)["matches"]
    feat = _load_feature_lookup(feature_parquet)
    if cluster_source_dir is None:
        cluster_source_dir = PROJECT_ROOT / "data" / "polymarket_test"
    cluster_lookup = (
        load_competition_clusters(cluster_source_dir)
        if cluster_source_dir.is_dir()
        else {}
    )

    # Canonical bet set: matches where the framework placed a flat bet.
    bet_matches = [m for m in matches
                   if flat_bet_team(m) is not None]

    baseline_pnls = [float(m["realized_pnl"]) for m in bet_matches]
    baseline_clusters = [
        cluster_id_for_record(match, cluster_lookup)
        for match in bet_matches
    ]
    baseline_wins = [flat_bet_won(match) for match in bet_matches]
    baseline = _summarize(
        baseline_pnls,
        baseline_clusters,
        baseline_wins,
        n_resamples,
    )

    variants = []
    for thr in edge_thrs:
        kept: List[float] = []
        kept_clusters: List[str] = []
        kept_wins: List[bool] = []
        n_close = n_mis_kept = n_mis_dropped = 0
        for m in bet_matches:
            f = feat.get(m["match_id"], {})
            diff = abs(float(f.get("top6_batting_elo_diff", 0.0)))
            pnl = float(m["realized_pnl"])
            if diff > boundary:  # mismatch
                me = _max_edge(m)
                if me is not None and me > thr:
                    kept.append(pnl)
                    kept_clusters.append(
                        cluster_id_for_record(m, cluster_lookup)
                    )
                    kept_wins.append(flat_bet_won(m))
                    n_mis_kept += 1
                else:
                    n_mis_dropped += 1
            else:  # close
                kept.append(pnl)
                kept_clusters.append(cluster_id_for_record(m, cluster_lookup))
                kept_wins.append(flat_bet_won(m))
                n_close += 1
        s = _summarize(kept, kept_clusters, kept_wins, n_resamples)
        s.update({"edge_thr": thr, "boundary": boundary,
                  "n_close": n_close, "n_mismatch_kept": n_mis_kept,
                  "n_mismatch_dropped": n_mis_dropped})
        variants.append(s)

    return {
        "baseline": baseline,
        "variants": variants,
        "bootstrap_contract": BOOTSTRAP_CONTRACT_VERSION,
        "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
        "bootstrap_resamples": n_resamples,
        "boundary": boundary,
        "n_bet_matches": len(bet_matches),
        "cluster_source_dir": str(cluster_source_dir.resolve()),
        "cluster_metadata_coverage": sum(
            match["match_id"] in cluster_lookup for match in matches
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--feature-parquet", type=Path, required=True)
    ap.add_argument("--boundary", type=float, default=5.0,
                    help="|top6_batting_elo_diff| boundary: >boundary = mismatch.")
    ap.add_argument("--edge-thresholds", default="0.05,0.10,0.15",
                    help="Mismatch-slice edge thresholds to sweep (decimals).")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument(
        "--cluster-source-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "polymarket_test",
    )
    args = ap.parse_args()

    edge_thrs = [float(x) for x in args.edge_thresholds.split(",")]
    res = run(
        args.in_path,
        args.feature_parquet,
        args.boundary,
        edge_thrs,
        cluster_source_dir=args.cluster_source_dir,
    )

    b = res["baseline"]
    print(f"\nBaseline (flat threshold 0, production): "
          f"n={b['n_bets']}  ROI={b['roi_pct']:+.2f}%  "
          f"CI=[{b['roi_ci_lo']:+.2f},{b['roi_ci_hi']:+.2f}]  "
          f"win={b['win_rate']:.1%}  maxDD={b['max_drawdown']:.2f}  "
          f"clusters={b['n_bootstrap_clusters']}")
    print(f"\nConditional rule (mismatch = |elo_diff|>{res['boundary']}, "
          f"require edge>thr there; close bets flat):")
    print(f"{'edge_thr':>9} {'n_bets':>7} {'close':>6} {'mis_keep':>9} "
          f"{'mis_drop':>9} {'ROI%':>8} {'CI_lo':>8} {'CI_hi':>8} "
          f"{'win%':>6} {'maxDD':>7}")
    for v in res["variants"]:
        print(f"{v['edge_thr']:>9.2f} {v['n_bets']:>7d} {v['n_close']:>6d} "
              f"{v['n_mismatch_kept']:>9d} {v['n_mismatch_dropped']:>9d} "
              f"{v['roi_pct']:>+8.2f} {v['roi_ci_lo']:>+8.2f} "
              f"{v['roi_ci_hi']:>+8.2f} {v['win_rate']:>6.1%} "
              f"{v['max_drawdown']:>7.2f}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n  -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
