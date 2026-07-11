"""
Side-by-side comparison of multiple model evaluations across liquidity slices.

Reads N eval JSON files (whether produced by run_sim_eval.py with
--min-volume or by reslice_eval_json.py post-hoc) and renders a flat
table: one row per model, one column group per slice, showing log loss
+ flat ROI with bootstrap 95% CIs.

Usage:
    uv run python scripts/sim_eval/compare_slices.py \\
        --label "v4 baseline"  --files eval_out/phase1_sliced_v4/*.json \\
        --label "v6 outcome-dist" --files eval_out/phase1_sliced/*.json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple


SLICE_ORDER = ["all", "min_volume_50000", "min_volume_100000"]
SLICE_LABELS = {
    "all":                "all (261)",
    "min_volume_50000":   "≥ $50k (170)",
    "min_volume_100000":  "≥ $100k (110)",
}


def _load_summary(path: Path) -> Tuple[str, dict]:
    """Load a Phase-1 summary JSON, return (slice_tag, summary_dict)."""
    with open(path) as f:
        data = json.load(f)
    summary = data.get("summary", {})
    slice_tag = summary.get("slice")
    if slice_tag is None:
        # legacy / non-sliced eval — treat as 'all'
        slice_tag = "all"
    return slice_tag, summary


def _fmt_ll(s: dict) -> str:
    ll = s.get("avg_log_loss")
    if ll is None or (isinstance(ll, float) and math.isnan(ll)):
        return "—"
    lo = s.get("avg_log_loss_ci_low")
    hi = s.get("avg_log_loss_ci_high")
    if lo is None or hi is None or math.isnan(lo) or math.isnan(hi):
        return f"{ll:.4f}"
    return f"{ll:.4f} [{lo:.4f}, {hi:.4f}]"


def _fmt_roi(s: dict) -> str:
    roi = s.get("flat_betting_roi_pct")
    if roi is None or (isinstance(roi, float) and math.isnan(roi)):
        return "—"
    lo = s.get("flat_betting_roi_ci_low")
    hi = s.get("flat_betting_roi_ci_high")
    if lo is None or hi is None or math.isnan(lo) or math.isnan(hi):
        return f"{roi:+.2f}%"
    return f"{roi:+.2f}% [{lo:+.2f}%, {hi:+.2f}%]"


def _fmt_n(s: dict) -> str:
    n = s.get("n_matches_evaluated") or s.get("n_matches")
    bets = s.get("flat_betting_bets_placed")
    if n is None and bets is None:
        return "—"
    if bets is None:
        return f"n={n}"
    return f"n={n} bets={bets}"


def render(groups: List[Tuple[str, List[Path]]]) -> str:
    """`groups` is a list of (label, [json_paths]).  Each model contributes
    one or more slice JSONs (typically 3: all/50k/100k)."""
    # Build {label: {slice_tag: summary_dict}}
    by_model: Dict[str, Dict[str, dict]] = {}
    for label, paths in groups:
        by_model.setdefault(label, {})
        for p in paths:
            slice_tag, summary = _load_summary(p)
            by_model[label][slice_tag] = summary

    lines: List[str] = []
    lines.append("=" * 110)
    lines.append("Sliced eval comparison")
    lines.append("=" * 110)

    for slice_tag in SLICE_ORDER:
        nice = SLICE_LABELS.get(slice_tag, slice_tag)
        lines.append("")
        lines.append(f"--- Slice: {nice} ---")
        # Column header
        lines.append(f"  {'model':<28s} {'log loss [95% CI]':<32s}  "
                     f"{'flat ROI [95% CI]':<32s}  {'matches':<22s}")
        for label, slices in by_model.items():
            if slice_tag not in slices:
                continue
            s = slices[slice_tag]
            lines.append(
                f"  {label:<28s} {_fmt_ll(s):<32s}  "
                f"{_fmt_roi(s):<32s}  {_fmt_n(s):<22s}"
            )

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Compare multiple sliced-eval JSONs side by side.")
    ap.add_argument(
        "--group", action="append", nargs="+", required=True,
        metavar=("LABEL", "PATH"),
        help="Repeated. First arg is the label, remainder are JSON paths "
             "(typically 1-3 per group: all / 50k / 100k slices).",
    )
    args = ap.parse_args()

    groups: List[Tuple[str, List[Path]]] = []
    for spec in args.group:
        if len(spec) < 2:
            sys.exit(f"--group needs LABEL + >=1 path, got {spec}")
        label = spec[0]
        paths = [Path(p) for p in spec[1:]]
        for p in paths:
            if not p.exists():
                sys.exit(f"Missing: {p}")
        groups.append((label, paths))

    print(render(groups))


if __name__ == "__main__":
    main()
