"""Aggregate blended-eval JSONs (Phase A1) into a single Markdown report.

Reads all `*_w*p*_<slice>.json` files produced by `reslice_eval_json.py`
on top of `blend_eval_json.py` outputs, builds a w × slice grid, and
emits Markdown tables with LL/Brier/ROI + 95% bootstrap CIs, win rate,
and bet counts. Also runs a per-match decomposition: how often does the
ensemble beat both components and where does it flip the bet side
relative to sim alone.

Usage:
    uv run python scripts/sim_eval/blend_report.py \\
        --sliced-dir eval_out_blend_a1/sliced \\
        --direct-json models/xgb_match_v1/test_predictions.json \\
        --out reports/blend_a1_report.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

# Hardcoded reference baselines from CLAUDE.md / TODO.md.
COINFLIP_LL = 0.6931
MARKET_LL_ALL = 0.6267  # frozen baseline LL on all-261; same number listed in TODO.md
ALWAYS_FAVORITE_ROI_PCT = 4.15

W_VALUES = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
W_LABEL = {
    0.0: "0.00 (direct alone)",
    0.2: "0.20",
    0.35: "0.35",
    0.5: "0.50",
    0.65: "0.65",
    0.8: "0.80",
    1.0: "1.00 (sim alone, v7)",
}
SLICE_LABEL = {
    "all": "all (261)",
    "min_volume_50000": "≥$50k (168)",
    "min_volume_100000": "≥$100k (110)",
}
SLICE_ORDER = ["all", "min_volume_50000", "min_volume_100000"]

W_TAG_RE = re.compile(r"_w(\d+)p(\d+)_(all|min_volume_\d+)\.json$")


def _parse_w_slice(filename: str):
    m = W_TAG_RE.search(filename)
    if not m:
        return None, None
    int_part, frac_part, slice_tag = m.groups()
    w = float(f"{int_part}.{frac_part}")
    return round(w, 2), slice_tag


def _format_row(w: float, summary: dict) -> str:
    ll = summary["avg_log_loss"]
    ll_lo = summary["avg_log_loss_ci_low"]
    ll_hi = summary["avg_log_loss_ci_high"]
    roi = summary["flat_betting_roi_pct"]
    roi_lo = summary["flat_betting_roi_ci_low"]
    roi_hi = summary["flat_betting_roi_ci_high"]
    bets = summary["flat_betting_bets_placed"]
    wr = summary["flat_betting_win_rate"] * 100
    return (
        f"| {W_LABEL[w]} "
        f"| {ll:.4f} | [{ll_lo:.4f}, {ll_hi:.4f}] "
        f"| {roi:+.2f}% | [{roi_lo:+.2f}%, {roi_hi:+.2f}%] "
        f"| {wr:.1f}% | {bets} |"
    )


def _build_grid(sliced_dir: Path) -> Dict[str, Dict[float, dict]]:
    grid: Dict[str, Dict[float, dict]] = {s: {} for s in SLICE_ORDER}
    for path in sliced_dir.glob("*.json"):
        w, slice_tag = _parse_w_slice(path.name)
        if w is None or slice_tag not in grid:
            continue
        with open(path) as f:
            data = json.load(f)
        grid[slice_tag][w] = data["summary"]
    return grid


def _per_match_decomposition(sliced_dir: Path,
                             direct_json: Path,
                             slice_tag: str = "all") -> dict:
    """Compare per-match LL between sim alone (w=1.0), direct alone
    (w=0.0), and the LL-best blend (whichever w minimized aggregate LL on
    this slice). Count wins, ties, losses.
    """
    grid = _build_grid(sliced_dir)
    if slice_tag not in grid or not grid[slice_tag]:
        return {}

    # Pick best-LL w on this slice.
    best_w = min(grid[slice_tag].keys(),
                 key=lambda w: grid[slice_tag][w]["avg_log_loss"])

    def _load(w: float) -> List[dict]:
        tag = f"w{w:.2f}".replace(".", "p")
        # Find a matching file in sliced_dir.
        for p in sliced_dir.glob(f"*_{tag}_{slice_tag}.json"):
            with open(p) as f:
                return json.load(f)["matches"]
        return []

    matches_sim = {m["match_id"]: m for m in _load(1.0)}
    matches_direct = {m["match_id"]: m for m in _load(0.0)}
    matches_best = {m["match_id"]: m for m in _load(best_w)}

    n = 0
    blend_beats_both = 0
    direct_beats_sim = 0
    blend_flips_bet_side_vs_sim = 0
    edge_above_3pct = 0
    for mid in matches_sim:
        if mid not in matches_direct or mid not in matches_best:
            continue
        ll_sim = matches_sim[mid].get("log_loss")
        ll_dir = matches_direct[mid].get("log_loss")
        ll_blend = matches_best[mid].get("log_loss")
        if any(x is None for x in (ll_sim, ll_dir, ll_blend)):
            continue
        n += 1
        if ll_blend < ll_sim and ll_blend < ll_dir:
            blend_beats_both += 1
        if ll_dir < ll_sim:
            direct_beats_sim += 1

        # Bet-side flip: highest-edge team in sim vs in best-blend.
        def _best(edge: dict):
            if not edge:
                return None
            t = max(edge, key=lambda k: edge[k])
            return t if edge[t] > 0 else None
        sim_bet = _best(matches_sim[mid].get("edge", {}))
        blend_bet = _best(matches_best[mid].get("edge", {}))
        if sim_bet != blend_bet and (sim_bet is not None or blend_bet is not None):
            blend_flips_bet_side_vs_sim += 1
        # Edge > 3% threshold on best-w blend.
        edge_dict = matches_best[mid].get("edge", {})
        if any(e > 0.03 for e in edge_dict.values()):
            edge_above_3pct += 1

    return {
        "slice": slice_tag,
        "best_w": best_w,
        "n_compared": n,
        "blend_beats_both_components": blend_beats_both,
        "direct_beats_sim_alone": direct_beats_sim,
        "blend_flips_bet_side_vs_sim": blend_flips_bet_side_vs_sim,
        "matches_with_edge_over_3pct": edge_above_3pct,
    }


def _gate_check(grid: Dict[str, Dict[float, dict]]) -> dict:
    """Apply the go/no-go gate on the ≥$50k slice. Both required:
        1. Some blended LL < market LL = 0.6267
        2. Some blended flat-ROI CI excludes zero (lower bound > 0)
    Report which w values clear each condition (none, possibly).
    """
    target = "min_volume_50000"
    sub = grid.get(target, {})
    ll_winners = [w for w, s in sub.items() if s["avg_log_loss"] < MARKET_LL_ALL]
    roi_winners = [w for w, s in sub.items()
                   if s["flat_betting_roi_ci_low"] > 0]
    return {
        "ll_clears_market": ll_winners,
        "roi_ci_excludes_zero": roi_winners,
        "both_clear": sorted(set(ll_winners) & set(roi_winners)),
    }


def render_markdown(sliced_dir: Path, direct_json: Path) -> str:
    grid = _build_grid(sliced_dir)
    out = []
    out.append("# Phase A1 — Direct + Sim Blend Report\n")
    out.append("LL/ROI by blend weight `w` and slice. "
               "`logit(P_final) = w·logit(P_sim) + (1−w)·logit(P_direct)`. "
               "Current regenerated ROI CIs use the I3 whole-competition "
               "block contract; historical reports retain the intervals "
               "recorded at generation time.\n")

    out.append(f"**Reference baselines** — coinflip LL {COINFLIP_LL:.4f}, "
               f"market LL {MARKET_LL_ALL:.4f}, "
               f"always-favorite flat ROI {ALWAYS_FAVORITE_ROI_PCT:+.2f}%.\n")

    for slice_tag in SLICE_ORDER:
        sub = grid.get(slice_tag, {})
        if not sub:
            continue
        out.append(f"\n## Slice: {SLICE_LABEL[slice_tag]}\n")
        out.append("| w | LL | LL 95% CI | Flat ROI | ROI 95% CI | Win rate | Bets |")
        out.append("|---|---|---|---|---|---|---|")
        for w in W_VALUES:
            if w in sub:
                out.append(_format_row(w, sub[w]))

    # Decision-tree characterization on ≥$50k.
    out.append("\n## Curve characterization (≥$50k slice)\n")
    sub_50k = grid.get("min_volume_50000", {})
    if sub_50k:
        ll_seq = [(w, sub_50k[w]["avg_log_loss"]) for w in W_VALUES if w in sub_50k]
        ll_seq.sort()
        is_monotone_increasing = all(
            ll_seq[i][1] <= ll_seq[i + 1][1] + 1e-6
            for i in range(len(ll_seq) - 1))
        is_monotone_decreasing = all(
            ll_seq[i][1] >= ll_seq[i + 1][1] - 1e-6
            for i in range(len(ll_seq) - 1))
        best_w = min(ll_seq, key=lambda t: t[1])[0]
        if is_monotone_increasing:
            shape = "monotone increasing in w"
            interp = (
                "**direct alone wins** — adding any sim weight worsens LL. "
                "Per the plan's decision tree: this means direct >> sim and "
                "sim is contributing only directional noise on top of direct's "
                "signal. Move to Phase A2 (richer features) — the cheap-subset "
                "is sufficient to dominate sim, but probably not enough to "
                "close the residual gap to market."
            )
        elif is_monotone_decreasing:
            shape = "monotone decreasing in w"
            interp = (
                "**sim alone wins** — direct adds nothing. Cheap features may "
                "be too thin or duplicate sim's signal. Phase A2 features are "
                "mandatory before re-evaluating."
            )
        else:
            shape = f"non-monotone — best w = {best_w}"
            interp = (
                "**Complementarity confirmed** — sim and direct error in different "
                "places, and the optimal blend is interior. This is the pattern "
                "that justifies the Phase B stacker. Phase A2 (richer features) "
                "should make this U deeper."
            )
        out.append(f"- LL-vs-w shape: **{shape}**, best LL at w = {best_w}")
        out.append(f"- Interpretation: {interp}")

    # Gate check.
    out.append("\n## Go/no-go gate check (≥$50k slice)\n")
    out.append("Required: model LL < market LL (0.6267) AND flat-ROI CI excludes zero.")
    g = _gate_check(grid)
    out.append(f"- LL < market: clears at w = {g['ll_clears_market'] or 'none'}")
    out.append(f"- ROI CI excludes 0: clears at w = {g['roi_ci_excludes_zero'] or 'none'}")
    out.append(f"- BOTH conditions: w = {g['both_clear'] or 'none'}")

    # Per-match decomposition.
    out.append("\n## Per-match decomposition (all slice)\n")
    decomp = _per_match_decomposition(sliced_dir, direct_json, "all")
    if decomp:
        out.append(f"- Compared on n = {decomp['n_compared']} matches")
        out.append(f"- Best blend w (lowest aggregate LL) = {decomp['best_w']}")
        out.append(f"- Blend beats both components per-match: {decomp['blend_beats_both_components']} / {decomp['n_compared']}")
        out.append(f"- Direct alone beats sim alone per-match: {decomp['direct_beats_sim_alone']} / {decomp['n_compared']}")
        out.append(f"- Best-blend flips bet side vs sim alone: {decomp['blend_flips_bet_side_vs_sim']}")
        out.append(f"- Best-blend matches with edge > 3%: {decomp['matches_with_edge_over_3pct']}")

    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sliced-dir", required=True, type=Path)
    ap.add_argument("--direct-json", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports/blend_a1_report.md"))
    args = ap.parse_args()

    md = render_markdown(args.sliced_dir, args.direct_json)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(md)
    print(f"\n[wrote {args.out}]")


if __name__ == "__main__":
    main()
