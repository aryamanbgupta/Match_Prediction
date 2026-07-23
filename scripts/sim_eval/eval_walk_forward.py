"""Walk-forward / monthly partition harness for match-level eval JSONs.

M1 (2026-05-10) — partitions an eval JSON's matches by year-month
(parsed from match_id prefix or from a feature parquet's match_date),
recomputes summary stats per partition, emits a markdown table.

Catches whether model edge is gaining or losing across the test period
— a one-shot eval hides per-month drift. Composes with --slice and
--min-volume identically to reslice_eval_json.py (which it borrows
helpers from), so e.g. you can do walk-forward on the IPL ≥$50k slice.

Usage:
    uv run python scripts/sim_eval/eval_walk_forward.py \\
        --in eval_out_m1_baseline/blend_w0p00.json \\
        --odds betting_odds_polymarket.json \\
        --out reports/walk_forward_m1.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "sim_eval"))

from reslice_eval_json import (  # noqa: E402
    SLICE_NAMES,
    _bootstrap_ci,
    _load_feature_lookup,
    _slice_predicate,
)
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


_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})_")


def _match_year_month(match: dict, feat_row: dict) -> Optional[str]:
    """Resolve YYYY-MM. Prefer parquet's match_date if joined; else
    parse from match_id prefix.
    """
    d = feat_row.get("match_date")
    if d is not None:
        return str(d)[:7]
    mid = match.get("match_id", "")
    m = _DATE_RE.match(mid)
    return f"{m.group(1)}-{m.group(2)}" if m else None


def walk_forward(eval_json_path: str, odds_json_path: str,
                 slice_name: str = "all",
                 feature_parquet: Optional[Path] = None,
                 mismatch_thresh: float = 15.0,
                 close_thresh: float = 5.0,
                 min_volume: Optional[float] = None,
                 n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
                 cluster_source_dir: Optional[Path] = None) -> List[dict]:
    """Group matches by YYYY-MM and recompute per-month stats."""
    eval_data = json.load(open(eval_json_path))
    odds_data = json.load(open(odds_json_path))
    vol_by_id = {m["match_id"]: m.get("polymarket_volume_usd")
                 for m in odds_data.get("matches", [])}
    feat_lookup = _load_feature_lookup(feature_parquet)
    predicate = _slice_predicate(slice_name, mismatch_thresh, close_thresh)
    if cluster_source_dir is None:
        default_source = PROJECT_ROOT / "data" / "polymarket_test"
        cluster_source_dir = (
            default_source if default_source.is_dir() else None
        )
    cluster_lookup = (
        load_competition_clusters(cluster_source_dir)
        if cluster_source_dir is not None
        else {}
    )

    by_month: Dict[str, List[dict]] = {}
    for match in eval_data.get("matches", []):
        if min_volume is not None:
            vol = vol_by_id.get(match["match_id"])
            if vol is None or vol < min_volume:
                continue
        feat = feat_lookup.get(match["match_id"], {})
        if not predicate(match, feat):
            continue
        ym = _match_year_month(match, feat)
        if ym is None:
            continue
        by_month.setdefault(ym, []).append(match)

    rows = []
    for ym in sorted(by_month):
        matches = by_month[ym]
        log_losses = [m["log_loss"] for m in matches
                      if m.get("log_loss") is not None and not (
                          isinstance(m["log_loss"], float)
                          and np.isnan(m["log_loss"]))]
        ll_matches = [
            match for match in matches
            if match.get("log_loss") is not None
            and not (
                isinstance(match["log_loss"], float)
                and np.isnan(match["log_loss"])
            )
        ]
        bet_matches = [
            match for match in matches if flat_bet_team(match) is not None
        ]
        flat_returns = [match["realized_pnl"] for match in bet_matches]
        avg_ll = float(np.mean(log_losses)) if log_losses else float("nan")
        ll_lo, ll_hi = _bootstrap_ci(
            log_losses,
            n=n_resamples,
            clusters=[
                cluster_id_for_record(match, cluster_lookup)
                for match in ll_matches
            ],
        )
        bets = len(flat_returns)
        total_pnl = float(np.sum(flat_returns)) if flat_returns else 0.0
        roi = (total_pnl / bets * 100) if bets else 0.0
        roi_clusters = [
            cluster_id_for_record(match, cluster_lookup)
            for match in bet_matches
        ]
        roi_lo, roi_hi = _bootstrap_ci(
            flat_returns,
            n=n_resamples,
            clusters=roi_clusters,
        )
        win_rate = (
            sum(1 for match in bet_matches if flat_bet_won(match)) / bets
            if bets else 0.0
        )
        rows.append({
            "month": ym,
            "n_matches": len(matches),
            "avg_log_loss": avg_ll,
            "ll_ci_low": ll_lo,
            "ll_ci_high": ll_hi,
            "bets": bets,
            "flat_roi_pct": roi,
            "roi_ci_low": roi_lo * 100 if not np.isnan(roi_lo) else float("nan"),
            "roi_ci_high": roi_hi * 100 if not np.isnan(roi_hi) else float("nan"),
            "win_rate": win_rate,
            "bootstrap_contract": BOOTSTRAP_CONTRACT_VERSION,
            "bootstrap_seed": DEFAULT_BOOTSTRAP_SEED,
            "bootstrap_resamples": n_resamples,
            "n_bootstrap_clusters": len(set(roi_clusters)),
            "bootstrap_reliable": (
                len(set(roi_clusters)) >= MIN_RECOMMENDED_CLUSTERS
            ),
            "cluster_metadata_coverage": sum(
                match["match_id"] in cluster_lookup for match in matches
            ),
        })
    return rows


def to_markdown(rows: List[dict], slice_name: str,
                min_volume: Optional[float]) -> str:
    """Render rows to a markdown summary."""
    vol_tag = "all volumes" if min_volume is None else f"≥${int(min_volume):,}"
    title = f"# Walk-forward eval — slice={slice_name}, {vol_tag}\n\n"
    if not rows:
        return title + "_(no matches in slice)_\n"
    contract = rows[0]["bootstrap_contract"]
    provenance = (
        f"I3 bootstrap: `{contract}`, seed "
        f"{rows[0]['bootstrap_seed']}, "
        f"{rows[0]['bootstrap_resamples']:,} resamples. "
        "Rows with fewer than 10 betting blocks are descriptive.\n\n"
    )
    header = "| Month | n | LL | LL 95% CI | Bets | Blocks | Flat ROI | ROI 95% CI | Win % |\n"
    sep = "|---|---|---|---|---|---|---|---|---|\n"
    body = []
    for r in rows:
        body.append(
            f"| {r['month']} | {r['n_matches']} | "
            f"{r['avg_log_loss']:.4f} | "
            f"[{r['ll_ci_low']:.4f}, {r['ll_ci_high']:.4f}] | "
            f"{r['bets']} | "
            f"{r['n_bootstrap_clusters']} | "
            f"{r['flat_roi_pct']:+.2f}% | "
            f"[{r['roi_ci_low']:+.2f}%, {r['roi_ci_high']:+.2f}%] | "
            f"{r['win_rate']:.1%} |"
        )
    return title + provenance + header + sep + "\n".join(body) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--odds", required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="Markdown output path.")
    ap.add_argument("--slice", choices=SLICE_NAMES, default="all")
    ap.add_argument("--mismatch-threshold", type=float, default=15.0)
    ap.add_argument("--close-threshold", type=float, default=5.0)
    ap.add_argument("--feature-parquet", type=Path, default=None)
    ap.add_argument("--min-volume", type=int, default=None)
    ap.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
    )
    ap.add_argument("--cluster-source-dir", type=Path, default=None)
    args = ap.parse_args()

    rows = walk_forward(
        args.in_path, args.odds,
        slice_name=args.slice,
        feature_parquet=args.feature_parquet,
        mismatch_thresh=args.mismatch_threshold,
        close_thresh=args.close_threshold,
        min_volume=args.min_volume,
        n_resamples=args.bootstrap_resamples,
        cluster_source_dir=args.cluster_source_dir,
    )
    md = to_markdown(rows, args.slice, args.min_volume)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(md)
    print(f"\n  → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
