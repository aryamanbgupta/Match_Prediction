#!/usr/bin/env python
"""Model-vs-market evaluation for a joined women's odds set.

`eval_womens_v1.py` gates a women's arm against coinflip and an ELO-logistic
baseline, which was the only comparison available while the track had no
market data.  This script is the market half: it scores every model arm
present in a `build_womens_polymarket_odds.py` output against the Polymarket
line on the same rows, sliced by split and by volume.

**Log loss and accuracy only.**  No ROI is reported here by design: CLAUDE.md
invariant 7 requires the I3 `tournament_time_block_v1` contract for any
economic claim, and these row counts (tens, not hundreds) are far below the
10-block floor at which that contract stops being descriptive.  A model that
merely beats the market on LL has not been shown to have a betting edge.

Usage:
    uv run python scripts/eval_womens_market.py \
        --odds data/womens_polymarket/betting_odds_womens_w1.json
    uv run python scripts/eval_womens_market.py \
        --odds data/womens_polymarket_leagues/betting_odds_womens_w2.json \
        --markdown reports/womens_market_eval_leagues.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

LN2 = math.log(2.0)
EPS = 1e-9
DEFAULT_THRESHOLDS = (0, 10_000, 50_000, 100_000)


def log_loss(probabilities: list[float], labels: list[int]) -> float:
    clipped = [min(max(p, EPS), 1.0 - EPS) for p in probabilities]
    return -sum(
        y * math.log(p) + (1 - y) * math.log(1 - p)
        for p, y in zip(clipped, labels)
    ) / len(labels)


def accuracy(probabilities: list[float], labels: list[int]) -> float:
    return sum((p > 0.5) == (y == 1) for p, y in zip(probabilities, labels)) / len(
        labels
    )


def resolution(probabilities: list[float]) -> float:
    return sum(abs(p - 0.5) for p in probabilities) / len(probabilities)


def arm_columns(rows: list[dict]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for key in row:
            if key.startswith("p_team1_") and key not in names:
                names.append(key)
    return names


def evaluate(rows: list[dict], arms: list[str]) -> dict | None:
    """Metrics for one slice; None when a row lacks a prediction."""
    usable = [r for r in rows if all(r.get(a) is not None for a in arms)]
    if not usable:
        return None
    labels = [1 if r["actual_winner"] == r["team1"] else 0 for r in usable]
    market = [r["prematch_prob_team1"] for r in usable]
    market_ll = log_loss(market, labels)
    result = {
        "n": len(usable),
        "dropped_missing_prediction": len(rows) - len(usable),
        "market": {
            "ll": market_ll,
            "acc": accuracy(market, labels),
            "res": resolution(market),
        },
        # A market worse than a coinflip is not a yardstick.  Thin women's
        # league books price near 0.5 and land on the wrong side often
        # enough to score above ln2, so "the model beat the market" on such
        # a slice says nothing about the model.
        "market_beats_coinflip": market_ll < LN2,
    }
    for arm in arms:
        values = [r[arm] for r in usable]
        result[arm] = {
            "ll": log_loss(values, labels),
            "acc": accuracy(values, labels),
            "res": resolution(values),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odds", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="Volume slices in USD (0 = all rows).",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=30,
        help=(
            "Row floor below which an arm-beats-market slice is treated as "
            "non-informative in the summary."
        ),
    )
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.odds.read_text())
    rows = payload["matches"]
    arms = arm_columns(rows)
    if not arms:
        raise SystemExit(f"{args.odds} carries no p_team1_* model columns")

    splits: list[str] = []
    for row in rows:
        if row["split"] not in splits:
            splits.append(row["split"])

    slices: list[tuple[str, list[dict]]] = []
    for split in splits + ["ALL"]:
        pool = rows if split == "ALL" else [r for r in rows if r["split"] == split]
        for threshold in args.thresholds:
            selected = [
                r
                for r in pool
                if (r.get("polymarket_volume_usd") or 0.0) >= threshold
            ]
            label = split if threshold == 0 else f"{split} >=${threshold:,}"
            slices.append((label, selected))

    results = {
        label: evaluate(pool, arms) for label, pool in slices if pool
    }

    header = f"{'slice':26} {'n':>4}  {'market':>16}"
    for arm in arms:
        header += f"  {arm.replace('p_team1_', ''):>22}"
    print(f"\n{payload.get('purpose', args.odds.name)}")
    print(f"coinflip LL = {LN2:.4f}\n")
    print(header)
    print(f"{'':26} {'':>4}  {'LL':>8}{'acc':>8}" + "".join(
        f"  {'LL':>10}{'acc':>12}" for _ in arms
    ))
    for label, metrics in results.items():
        if metrics is None:
            continue
        flag = "" if metrics["market_beats_coinflip"] else " !"
        line = (
            f"{label:26} {metrics['n']:>4}  "
            f"{metrics['market']['ll']:>8.4f}{metrics['market']['acc']:>8.3f}{flag:<2}"
        )
        for arm in arms:
            line += f"  {metrics[arm]['ll']:>10.4f}{metrics[arm]['acc']:>12.3f}"
        print(line)

    beaten = [
        label
        for label, m in results.items()
        if m and any(m[a]["ll"] < m["market"]["ll"] for a in arms)
    ]
    informative = [
        label
        for label in beaten
        if results[label]["market_beats_coinflip"]
        and results[label]["n"] >= args.min_n
    ]
    print("\n  ! = market scored worse than a coinflip on that slice")
    print(
        f"\nslices where an arm beats the market on LL: {len(beaten)}/{len(results)}"
        + (f"  -> {beaten}" if beaten else "")
    )
    print(
        f"  ...of which INFORMATIVE (market beats coinflip and n >= "
        f"{args.min_n}): {len(informative)}"
        + (f"  -> {informative}" if informative else "  -> none")
    )
    print(
        "\nNo ROI reported: invariant 7 requires the I3 tournament-block "
        "contract, and these row counts sit below its descriptive floor."
    )

    if args.markdown:
        lines = [
            f"# Women's model vs market — {payload.get('purpose', '')}",
            "",
            f"Source: `{args.odds}`  ",
            f"Capture: `{payload.get('capture', {}).get('path', 'n/a')}`  ",
            f"Generated: {payload.get('generated_at', 'n/a')}",
            "",
            f"Coinflip LL = {LN2:.4f}. Log loss and accuracy only — no ROI "
            "(invariant 7 requires the I3 block contract).",
            "",
            "Rows marked **!** are slices where the market itself scored "
            "worse than a coinflip. On those, beating the market is not "
            "evidence of skill — the line carries no information to beat.",
            "",
            "| slice | n | market LL | market acc | "
            + " | ".join(
                f"{a.replace('p_team1_', '')} LL | "
                f"{a.replace('p_team1_', '')} acc"
                for a in arms
            )
            + " |",
            "|---|---:|---:|---:|" + "---:|" * (2 * len(arms)),
        ]
        for label, m in results.items():
            if m is None:
                continue
            cells = [
                label + ("" if m["market_beats_coinflip"] else " **!**"),
                str(m["n"]),
                f"{m['market']['ll']:.4f}",
                f"{m['market']['acc']:.3f}",
            ]
            for arm in arms:
                cells += [f"{m[arm]['ll']:.4f}", f"{m[arm]['acc']:.3f}"]
            lines.append("| " + " | ".join(cells) + " |")
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.markdown}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
